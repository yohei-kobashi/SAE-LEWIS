"""
P-I: the causal test — let the INTERVENED LM's own head say what to edit.

WHY THIS EXISTS. The claim we want is causal: that the identified SAE
activations genuinely encode the linguistic phenomenon in the LM, not merely
that they carry enough information for some other model to read. Detection
alone cannot establish that (the feature may fire on a confound — a
particular word rather than the phenomenon); intervention is what breaks the
confound. Our Edit Flow editor cannot make that claim by construction: it
never modifies an activation, it takes W_dec directions as extra INPUT
tokens, so it is conditioning, not intervention (P(edit | Z=z), not
P(Y | do(Z=z))).

WHAT THIS DOES. Clamp the SAE features in gemma-2-2b's layer-12 residual
stream, then read the edit out of the LM's OWN next-token head, teacher-
forced on the source:

    Delta_i = log p_int(x_i | x_<i) - log p_recon(x_i | x_<i)

    Delta_i << 0  ->  the intervention makes this token unlikely  -> edit HERE
    argmax p_int(. | x_<i)                                        -> edit to THIS

There is NO trained readout — no rate head, no Q head, nothing fit to the
task. The only learned object in the loop is the frozen LM itself. That is
the point: "your probe just decodes what you injected" cannot be said of a
readout that is the LM's own head. It is the one form the amnesic-probing
critique does not reach.

It also fills the empty cell of the 2x2 (intervention x discrete edits):
LinguaLens/AxBench/ActAdd intervene and regenerate; our EF conditions and
edits; nobody intervenes and edits.

TWO DESIGN POINTS THAT MATTER.
1. The clean baseline is `recon` (encode->decode passthrough), NOT `raw`.
   The hook replaces the residual with the SAE reconstruction, so a raw
   baseline would make Delta mix reconstruction damage with the
   intervention. Against recon, Delta isolates the SET.
2. We intervene on gemma-2-2b (base) — the model Gemma Scope's SAE was
   actually trained on — not on gemma-2-2b-it. B1 mirrors LinguaLens's
   base-SAE + instruct-model setup on purpose; for a causal claim that
   mismatch is a confound we do not need, and teacher-forced scoring needs
   no instruct model anyway.

Controls (the claim lives or dies on these):
  empty  : no features set -> Delta == 0 everywhere -> zero edits -> copy.
           Structural, not learned.
  random : same count and magnitudes at random feature indices. If random
           clamps edit as well as the true ones, the features are not
           carrying the phenomenon.

Usage:
    python scripts/eval_clamp_readout.py \
        --output-dir runs/prod_gemma_v6/clamp_readout500 \
        --clamp-values 10 --delta-thr -1.0 --device cuda
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_lingualens import (                                    # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk, pair_metrics,
    randomize_intervention, sae_z_with_offsets,
)
from model import SAEFeatureExtractor, load_sae                  # noqa: E402
from scripts.eval_clamp_baseline import SaeClampHook             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b",
                   help="the model the SAE was trained on — intervene HERE, "
                        "not on -it, so the PT/IT mismatch is not a confound")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blocklist", default="runs/blocklist/blocklist.npy")
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--k-amp", type=int, default=32)
    p.add_argument("--k-sup", type=int, default=32)
    p.add_argument("--clamp-values", default="10",
                   help="comma-separated set values (LinguaLens uses 10)")
    p.add_argument("--delta-thr", type=float, default=-1.0,
                   help="edit position i when Delta_i < this (nats). More "
                        "negative = the intervention must object more "
                        "strongly before we touch the token.")
    p.add_argument("--conditions", default="true,empty,random")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


@torch.no_grad()
def teacher_forced_logprobs(model, ids, device):
    """log p(x_i | x_<i) for every i>=1, plus the full logits row that
    predicts each x_i (i.e. row i-1). One forward, no generation."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=x).logits[0].float()      # (T, V)
    logp = F.log_softmax(logits, dim=-1)               # (T, V)
    tgt = torch.tensor(ids[1:], dtype=torch.long, device=device)
    own = logp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)   # (T-1,)
    return own, logits[:-1]            # own[j] scores ids[j+1]; row j too


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blk = None
    if args.blocklist and Path(args.blocklist).exists():
        blk = torch.as_tensor(np.asarray(np.load(args.blocklist),
                                         dtype=np.int64))
        print(f"[readout] blocklist: {len(blk)} features masked")

    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.sample_size, len(ds)),
                     replace=False)
    print(f"[readout] {len(ds)} pairs, sampling {len(idx)}")

    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
        device=args.device)
    sae = load_sae(args.sae_repo, args.sae_path, args.sae_type,
                   args.sae_k).to(args.device)

    tok = AutoTokenizer.from_pretrained(args.llm)
    lm = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=dtype).to(args.device).eval()
    hook = SaeClampHook(sae)
    lm.model.layers[args.sae_layer].register_forward_hook(hook)
    print(f"[readout] intervening on {args.llm} layers[{args.sae_layer}]; "
          f"clean baseline = recon passthrough (isolates the SET)")

    clamp_vals = [float(v) for v in args.clamp_values.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    prng = np.random.default_rng(args.seed + 1)
    records = []

    for n, k in enumerate(idx, 1):
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        # SAE extraction and the intervened LM are the SAME model here, so
        # one tokenizer serves both — unlike B1, which mirrors LinguaLens's
        # base-SAE + instruct-model split and needs two.
        ids = tok(src, add_special_tokens=True).input_ids
        tgt_ids = tok(tgt, add_special_tokens=True).input_ids
        if len(ids) < 3:
            continue

        with torch.no_grad():
            s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
            t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
            om_s = [tuple(o) for o in tok(
                src, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            om_t = [tuple(o) for o in tok(
                tgt, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            opcodes = difflib.SequenceMatcher(
                None, ids, tgt_ids, autojunk=False).get_opcodes()
            sr, tr = edit_char_ranges(opcodes, om_s, om_t)
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
        zvar = {
            "true": (za_t, zs_t),
            "empty": (torch.zeros_like(za_t), torch.zeros_like(zs_t)),
            "random": (randomize_intervention(za_t, prng),
                       randomize_intervention(zs_t, prng)),
        }

        rec = {"idx": int(k), "src": src, "tgt": tgt, "outputs": {}}
        for cond in conditions:
            za, zs = zvar[cond]
            rec["outputs"][cond] = {}
            for cv in clamp_vals:
                # ---- clean = RECON passthrough: hook on, nothing set -----
                hook.enabled = True
                hook.amp_idx = hook.sup_idx = None
                hook.amp_val = None
                lp_recon, _ = teacher_forced_logprobs(lm, ids, args.device)
                # ---- intervened: same recon path, features SET -----------
                hook.amp_idx = torch.nonzero(za > 0).flatten().to(args.device)
                hook.amp_val = float(cv)
                hook.sup_idx = torch.nonzero(zs > 0).flatten().to(args.device)
                lp_int, logits_int = teacher_forced_logprobs(
                    lm, ids, args.device)
                hook.enabled = False

                delta = (lp_int - lp_recon)                    # (T-1,)
                fire = (delta < args.delta_thr).nonzero().flatten().tolist()
                out_ids = list(ids)
                for j in fire:                                  # j scores ids[j+1]
                    out_ids[j + 1] = int(logits_int[j].argmax())
                out_text = tok.decode(out_ids, skip_special_tokens=True)

                pm = pair_metrics(out_text, src, tgt)
                rec["outputs"][cond][f"clamp{cv:g}"] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"], "copy": pm["copy_rate"],
                    "n_fire": len(fire),
                    "delta_min": float(delta.min()) if len(delta) else 0.0,
                }
        records.append(rec)
        if n % 25 == 0:
            print(f"[readout] {n}/{len(idx)}")

    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [f"# P-I clamp-readout — the intervened LM's own head picks the edit",
             "", f"pairs: {len(records)}; intervene on {args.llm} "
             f"layers[{args.sae_layer}]; clean = recon passthrough; "
             f"Delta < {args.delta_thr} fires a substitution", "",
             "| condition | mode | exact | sim | copy | mean fires |",
             "|---|---|---|---|---|---|"]
    for cond in conditions:
        modes = sorted({m for r in records for m in r["outputs"].get(cond, {})})
        for m in modes:
            rs = [r["outputs"][cond][m] for r in records
                  if m in r["outputs"].get(cond, {})]
            if not rs:
                continue
            f_ = lambda key: sum(x[key] for x in rs) / len(rs)   # noqa: E731
            lines.append(f"| {cond} | {m} | {f_('exact'):.4f} | "
                         f"{f_('sim_target'):.4f} | {f_('copy'):.4f} | "
                         f"{f_('n_fire'):.2f} |")
    lines += ["", "Reading: `empty` MUST be exact~0 / copy 1.0 — Delta is "
              "identically 0 with nothing set, so zero fires, structurally. "
              "The claim rests on true >> random: same count and magnitudes, "
              "only the feature identities differ, so any gap is what the "
              "IDENTIFIED features causally buy. If true ~= random, the "
              "identified activations are not carrying the phenomenon."]
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.md").write_text(report + "\n")
    print(f"\n[readout] wrote {out_dir}/report.md, records.jsonl")


if __name__ == "__main__":
    main()
