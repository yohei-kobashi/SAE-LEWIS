#!/usr/bin/env python3
"""LinguaLens intervention-evaluation reproduction — GENERATION side (GPU).

Faithful to the OFFICIAL REPO (github.com/THU-KEG/LinguaLens,
lingualens/intervener.py::Intervener.run_intervention_experiment), per user
instruction "論文ではなくgithubに基づいて実装して下さい":

  * intervention  = InterventionConfig(mode="set", value=0/10,
    prompt_only=False) — encode layer output with the SAE, SET the target
    latent (force-insert included), REPLACE the residual with the
    reconstruction, at every position of prompt AND every decode step.
  * reference     = repo's "control": mode="multiply", value=1.0 — the SAE
    reconstruction passthrough with no feature change ("the unmodified SAE
    model" the paper judges against).
  * generation    = model.generate(max_new_tokens=100, temperature=1.0,
    do_sample=True) on the RAW prompt string (repo applies no chat
    template), defaults of run_intervention_experiment().

What the repo does NOT contain (paper-only, flagged in the report):
  * the LLM-as-a-judge and success-rate/E_abl/E_enh/FIC math (§3.3, §4.2.2,
    App. E.2)  -> scripts/judge_ll_repro.py
  * the random-vector baseline. §3.3 says "randomly selecting 25 base
    vectors", §4.2.2 says "randomly select 50 base vector indices" for 50
    experiments — an internal discrepancy. We read it as ONE fresh random
    vector per experiment (50 indices / 50 experiments), shape-matched to
    the targeted arm (which intervenes ONE vector at a time, averaged over
    the FRC top-3 — §4.2.2). --random-nvec lets the 25-simultaneous reading
    be run as a variant.

Adaptations to our stack (documented, unavoidable):
  * Llama-3.1-8B + OpenSAE(32L)  ->  gemma-2-2b-it + Gemma Scope layer-12/16k
    (same shape as our B1; numbers anchor to Table 2 in PATTERN and RANGE,
    not digit-for-digit).
  * feature -> FRC top-3 from our code-faithful identification
    (runs/frc/identified_l12_16k_r3.json, leakage-fixed).
  * Table 2's Simile row: "simile" is not among the 99 English features
    (their own App. E.2 calls that row "the metaphor feature") — we run
    `metaphor`. `causality` has no English-99 counterpart and is skipped.

Prompts: paper App. D.1 gives the exact prompts for Politeness, Linking
Verb and Past-Tense; metaphor gets an analogously-shaped one (constructed —
flagged).

Run inside interact-g:
  python scripts/eval_ll_repro_gen.py --output-dir runs/ll_repro
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch


def stable_hash(*parts) -> int:
    """Deterministic across processes (str hash() is salted per run —
    would break resume reproducibility of seeds and random draws)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load_sae                                      # noqa: E402

# paper App. D.1 verbatim prompts; metaphor is ours (no D.1 prompt exists)
FEATURES = {
    "past_tense":   "User: Sir, tell me a story about you. Assistant:",
    "linking_verb": "User: Sir, tell me something about your ideal room. Assistant:",
    "politeness":   "User: Sir, I want to make an order offline. Assistant:",
    "metaphor":     "User: Sir, describe the sea for me. Assistant:",
}
D_SAE = 16384


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"],
                   default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--frc-sets", default="runs/frc/identified_l12_16k_r3.json")
    p.add_argument("--features", default=",".join(FEATURES))
    p.add_argument("--num-experiments", type=int, default=50,
                   help="paper: 'For each feature, we conduct 50 experiments'")
    p.add_argument("--random-nvec", type=int, default=1,
                   help="latents per random-baseline experiment (1 = shape-"
                        "matched reading; 25 = §3.3's literal count)")
    p.add_argument("--enh-value", type=float, default=10.0)
    p.add_argument("--abl-value", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=100)   # repo default
    p.add_argument("--temperature", type=float, default=1.0)    # repo default
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


class SetHook:
    """OpenSAE-faithful 'set' intervention (same semantics as B1's
    SaeClampHook in eval_clamp_baseline.py): encode -> set latent(s) ->
    replace the residual with the reconstruction, every position, prompt +
    decode steps. row_idx allows a different latent per batch row (random
    baseline). set_idx=None with enabled=True = multiply-by-1 control (the
    reconstruction passthrough)."""

    def __init__(self, sae):
        self.sae = sae
        self.enabled = False
        self.set_idx = None       # LongTensor (k,) shared across the batch
        self.row_idx = None       # list[LongTensor] one per batch row
        self.set_val = 0.0

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return None
        h = output[0] if isinstance(output, tuple) else output
        dt = h.dtype
        z = self.sae.encode(h.to(self.sae.W_enc.dtype))
        if self.row_idx is not None:
            for b, idx in enumerate(self.row_idx):
                z[b, :, idx] = float(self.set_val)
        elif self.set_idx is not None and self.set_idx.numel():
            z[..., self.set_idx] = float(self.set_val)
        h_new = self.sae.decode(z).to(dt)
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    feats = [f for f in args.features.split(",") if f]

    frc = json.loads(Path(args.frc_sets).read_text())
    top3 = {}
    for f in feats:
        if f not in frc:
            raise SystemExit(f"feature {f!r} not in {args.frc_sets}")
        top3[f] = [int(v) for v, _ in frc[f][:3]]
        print(f"[llrepro] {f}: FRC top-3 = {top3[f]}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.it_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    sae = load_sae(args.sae_type, args.sae_repo, args.sae_path,
                   sae_k=args.sae_k).to(args.device).eval()
    hook = SetHook(sae)
    model.model.layers[args.sae_layer].register_forward_hook(hook)
    print(f"[llrepro] set-hook on layers[{args.sae_layer}] output "
          f"(all positions, prompt+decode — prompt_only=False)")

    rec_path = out_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        with open(rec_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["feature"], r["cond"], r["direction"],
                              r["vec_tag"], int(r["i"])))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[llrepro] RESUME: {len(done)} generations cached")
    rec_fh = open(rec_path, "a")

    @torch.no_grad()
    def gen_batch(prompt, n, seed_key):
        """n sampled generations of the raw prompt (no chat template —
        repo passes the string straight to the tokenizer). 2026-07-18
        audit fix: the official repo decodes the FULL sequence
        (tokenizer.decode(generated_ids[0]) — PROMPT INCLUDED) and that
        full text is what the judge reads; v1 of this script decoded the
        continuation only."""
        enc = tok([prompt] * n, return_tensors="pt").to(args.device)
        torch.manual_seed(stable_hash(*seed_key) % (2**31))
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, do_sample=args.temperature > 0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return [tok.decode(o, skip_special_tokens=True) for o in out]

    def emit(feature, cond, direction, vec_tag, idxs, texts, vec_of=None):
        for i, t in zip(idxs, texts):
            rec_fh.write(json.dumps({
                "feature": feature, "cond": cond, "direction": direction,
                "vec_tag": vec_tag, "i": int(i), "output": t,
                "vec": vec_of(i) if vec_of else None,
                "prompt": FEATURES[feature]}, ensure_ascii=False) + "\n")
        rec_fh.flush()

    def run_group(feature, cond, direction, vec_tag, configure, vec_of=None):
        """Generate the missing i in 0..N-1 for one (feature, cond,
        direction, vec) cell. `configure(rows)` sets the hook for the given
        experiment rows."""
        N = args.num_experiments
        missing = [i for i in range(N)
                   if (feature, cond, direction, vec_tag, i) not in done]
        if not missing:
            return
        prompt = FEATURES[feature]
        for lo in range(0, len(missing), args.batch_size):
            rows = missing[lo:lo + args.batch_size]
            configure(rows)
            texts = gen_batch(prompt, len(rows),
                              (feature, cond, direction, vec_tag, rows[0]))
            emit(feature, cond, direction, vec_tag, rows, texts, vec_of)
        print(f"[llrepro] {feature}/{cond}/{direction}/{vec_tag}: "
              f"+{len(missing)}")

    for f in feats:
        # reference = repo control (multiply x1): recon passthrough
        def cfg_ctl(rows):
            hook.enabled, hook.set_idx, hook.row_idx = True, None, None
        run_group(f, "control", "none", "ctl", cfg_ctl)

        # targeted: one FRC vector at a time (averaged later — §4.2.2)
        for v in top3[f]:
            for direction, val in (("enh", args.enh_value),
                                   ("abl", args.abl_value)):
                def cfg_t(rows, v=v, val=val):
                    hook.enabled, hook.row_idx = True, None
                    hook.set_idx = torch.tensor([v], device=args.device)
                    hook.set_val = val
                run_group(f, "targeted", direction, str(v), cfg_t,
                          vec_of=lambda i, v=v: [v])

        # random baseline: fresh vector(s) per experiment, same layer,
        # excluding the targeted three (no blocklist — the repo has none)
        rng = np.random.default_rng(args.seed + stable_hash(f) % 10000)
        pool = np.setdiff1d(np.arange(D_SAE), np.asarray(top3[f]))
        draws = [torch.as_tensor(rng.choice(pool, size=args.random_nvec,
                                            replace=False),
                                 device=args.device)
                 for _ in range(args.num_experiments)]
        for direction, val in (("enh", args.enh_value),
                               ("abl", args.abl_value)):
            def cfg_r(rows, val=val):
                hook.enabled, hook.set_idx = True, None
                hook.set_val = val
                hook.row_idx = [draws[i] for i in rows]
            run_group(f, "random", direction, "rnd", cfg_r,
                      vec_of=lambda i: draws[i].tolist())

    rec_fh.close()
    meta = {"features": {f: top3[f] for f in feats},
            "prompts": {f: FEATURES[f] for f in feats},
            "num_experiments": args.num_experiments,
            "random_nvec": args.random_nvec,
            "generation": {"max_new_tokens": args.max_new_tokens,
                           "temperature": args.temperature,
                           "chat_template": False},
            "intervention": "OpenSAE-faithful set + full recon replacement, "
                            "prompt_only=False",
            "reference": "repo control = multiply x1 recon passthrough",
            "stack": {"model": args.it_model, "sae": args.sae_path}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[llrepro] GENERATION DONE -> {rec_path}")
    print("[llrepro] next: bash run_ll_repro_judge.sh (prepost, needs "
          "OPENAI_API_KEY)")


if __name__ == "__main__":
    main()
