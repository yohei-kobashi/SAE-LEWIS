#!/usr/bin/env python3
"""AxBench SAE/SAE-A steering reproduction — GENERATION side (GPU).

Faithful to the OFFICIAL REPO (github.com/stanfordnlp/axbench), per user
instruction. Sources (verified in the repo, 2026-07-17):

  * task        = instruction-following steering: sample 10 AlpacaEval
    instructions per concept with df.sample(10, random_state=concept_id)
    (utils/dataset.py), chat-template them (add_generation_prompt, BOS
    stripped — the tokenizer re-adds it), generate 128 tokens at
    temperature 1.0 with do_sample=True (models/model.py::predict_steer,
    sweep/wuzhengx/2b/*/no_grad.yaml).
  * steering    = AdditionIntervention: h + factor * max_act * W_dec[latent]
    at model.layers[L].output, EVERY position, prompt + decode
    (intervene_on_prompt=True; models/interventions.py).
  * factors     = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0,5.0]
    (no_grad.yaml). Factor selected per concept on one half of the
    instructions, scored on the holdout half (evaluate.py,
    winrate_split_ratio 0.5).
  * max_act     = per-latent max activation. The repo's primary path reads
    Neuronpedia's maxValue (needs NP_API_KEY, fallback 50 when <=0); its
    sanctioned alternative (disable_neuronpedia_max_act, the
    gemmascope_axbench_max_act.yaml variant) computes it from the AxBench
    dataset — we use that: max over latent_eval_data texts. Flagged in
    meta.json.
  * concepts    = axbench/concept500/prod_2b_l{10,20}_v1/generate/
    metadata.jsonl — concept text + Neuronpedia ref (the VANILLA latent).
  * SAE-A       = per concept, the latent with max detection AUROC over
    the labeled latent_eval_data (paper §: "compute AUROC over the dataset
    given true labels, and select the highest-scoring feature"). The repo
    ships the labeled data (inference/latent_eval_data.parquet: 36
    positive / 36 negative per concept); selection recomputed here with
    Mann-Whitney average-rank AUROC (tie-safe, same math as our
    select_features_auroc.py).

Extra arm (cross-protocol, OUR addition — not in AxBench):
  * ll_set10    = LinguaLens's intervention mechanics on the same vanilla
    latent and the same test data: OpenSAE-style set-10 with full
    reconstruction replacement, all positions (their enhancement op).
    Single "factor" (their protocol has none) — still split-scored.

The SAE-LEWIS (EF editor) arm is added separately after S6 settles (its
checkpoint loading is copied verbatim from prune_spec.py at that point).

Anchors (paper Table 2, gemma-2-2b): SAE L10 0.177 / L20 0.151;
SAE-A L10 0.166 / L20 0.132. (0.157 = SAE-A avg over 2B+9B configs.)

Run inside interact-g (after cloning axbench into third_party/ — the
runner does this): bash run_axbench_repro.sh
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import SAEFeatureExtractor, load_sae                 # noqa: E402

FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0,
           4.0, 5.0]                       # no_grad.yaml steering_factors
D_SAE = 16384


def stable_hash(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--axbench-dir", default="third_party/axbench")
    p.add_argument("--config", default="prod_2b_l20_v1",
                   help="concept500 config (prod_2b_l10_v1 / prod_2b_l20_v1)")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", default="",
                   help="empty = canonical 16k for --layer "
                        "(l20: average_l0_71, l10: average_l0_77)")
    p.add_argument("--llm", default="google/gemma-2-2b",
                   help="SAE activations for max_act/AUROC are computed on "
                        "the base model (Gemma Scope is pt-trained)")
    p.add_argument("--arms", default="sae,sae_a,ll_set10")
    p.add_argument("--num-concepts", type=int, default=100,
                   help="first N concept_ids (500 = full paper scale)")
    p.add_argument("--num-instructions", type=int, default=10)   # theirs
    p.add_argument("--max-new-tokens", type=int, default=128)    # theirs
    p.add_argument("--temperature", type=float, default=1.0)     # theirs
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


CANONICAL_16K = {10: "layer_10/width_16k/average_l0_77/params.npz",
                 20: "layer_20/width_16k/average_l0_71/params.npz",
                 12: "layer_12/width_16k/average_l0_82/params.npz"}


class AdditionHook:
    """AxBench steering: h + mag * max_act * W_dec[latent], every position
    (their AdditionIntervention adds to prompt and decode steps alike)."""

    def __init__(self, w_dec):
        self.w_dec = w_dec          # (d_sae, d_model)
        self.enabled = False
        self.vec = None             # (d_model,) precomputed alpha * W_dec row

    def __call__(self, module, inputs, output):
        if not self.enabled or self.vec is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h_new = h + self.vec.to(h.dtype)
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new


class SetHook:
    """LinguaLens/OpenSAE mechanics (same as eval_ll_repro_gen.py): encode
    -> set latent -> replace residual with the reconstruction."""

    def __init__(self, sae):
        self.sae = sae
        self.enabled = False
        self.set_idx = None
        self.set_val = 10.0

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return None
        h = output[0] if isinstance(output, tuple) else output
        dt = h.dtype
        z = self.sae.encode(h.to(self.sae.W_enc.dtype))
        if self.set_idx is not None:
            z[..., self.set_idx] = float(self.set_val)
        h_new = self.sae.decode(z).to(dt)
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new


def mannwhitney_auroc(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Column-wise AUROC(pos > neg) via average ranks (tie-safe).
    pos: (P, d), neg: (N, d) -> (d,)"""
    from scipy.stats import rankdata
    P, N = len(pos), len(neg)
    allv = np.concatenate([pos, neg], axis=0)
    ranks = rankdata(allv, axis=0)
    r_pos = ranks[:P].sum(axis=0)
    u = r_pos - P * (P + 1) / 2.0
    return u / (P * N)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    sae_path = args.sae_path or CANONICAL_16K[args.layer]
    ax = Path(args.axbench_dir)
    cfg_dir = ax / "axbench" / "concept500" / args.config

    # ---- their test data ---------------------------------------------------
    meta_rows = [json.loads(l) for l in
                 open(cfg_dir / "generate" / "metadata.jsonl")]
    meta_rows = meta_rows[:args.num_concepts]
    concepts = [(i, m["concept"], int(m["ref"].split("/")[-1]))
                for i, m in enumerate(meta_rows)]
    alpaca = pd.read_json(ax / "alpaca_eval.json")
    led = pd.read_parquet(cfg_dir / "inference" / "latent_eval_data.parquet")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.it_model)

    def chat_wrap(instruction: str) -> list:
        # Token IDS of the chat-templated prompt, BOS included. The v1
        # string path (decode -> re-tokenize at generate time) silently
        # produced EMPTY prompts on the GPU env — 2026-07-17 debug run:
        # identical greedy output for different instructions — so the ids
        # go straight to generate() with manual left padding, no string
        # round trip.
        ids = tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True, add_generation_prompt=True)
        if ids and isinstance(ids[0], list):     # version drift guard
            ids = ids[0]
        if hasattr(ids, "input_ids"):            # BatchEncoding variant
            ids = ids.input_ids
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        ids = [int(t) for t in ids]
        if tok.bos_token_id is not None and (
                not ids or ids[0] != tok.bos_token_id):
            ids = [tok.bos_token_id] + ids
        return ids

    # per-concept instructions: THEIR sampler, verbatim
    instr = {cid: alpaca.sample(args.num_instructions, random_state=int(cid)
                                )["instruction"].tolist()
             for cid, _, _ in concepts}

    # ---- stage A: max_act + SAE-A selection from THEIR labeled data --------
    stage_a_path = out_dir / f"stage_a_{args.config}.json"
    if stage_a_path.exists():
        stage_a = json.loads(stage_a_path.read_text())
        stage_a = {int(k): v for k, v in stage_a.items()}
        print(f"[axrepro] stage A cached: {len(stage_a)} concepts")
    else:
        stage_a = {}
    todo = [c for c in concepts if c[0] not in stage_a]
    if todo:
        extractor = SAEFeatureExtractor(
            llm_name=args.llm, sae_repo=args.sae_repo, sae_path=sae_path,
            sae_layer=args.layer, sae_type="jumprelu", sae_k=None,
        ).to(args.device).eval()
        for cid, concept, vanilla in todo:
            sub = led[led.concept_id == cid]
            pos = sub[sub.category == "positive"]["output"].tolist()
            neg = sub[sub.category == "negative"]["output"].tolist()
            feats = []
            for text in pos + neg:
                _, z = sae_z_with_offsets_compat(extractor, text, args.device)
                feats.append(z.max(dim=0).values.float().cpu().numpy())
            feats = np.stack(feats)                     # (P+N, d_sae)
            auroc = mannwhitney_auroc(feats[:len(pos)], feats[len(pos):])
            sae_a = int(auroc.argmax())
            # max_act over the SAME eval texts (repo's dataset-side path)
            max_all = feats.max(axis=0)
            stage_a[cid] = {
                "vanilla": vanilla, "sae_a": sae_a,
                "sae_a_auroc": float(auroc[sae_a]),
                "vanilla_auroc": float(auroc[vanilla]),
                "max_act_vanilla": float(max_all[vanilla]) or 50.0,
                "max_act_sae_a": float(max_all[sae_a]) or 50.0,
            }
            stage_a_path.write_text(json.dumps(stage_a))
            print(f"[axrepro] A {cid}: vanilla {vanilla} "
                  f"(auroc {stage_a[cid]['vanilla_auroc']:.3f}, "
                  f"max {stage_a[cid]['max_act_vanilla']:.1f}) | SAE-A "
                  f"{sae_a} (auroc {stage_a[cid]['sae_a_auroc']:.3f})")
        del extractor
        torch.cuda.empty_cache()

    # ---- stage B: steered generation ---------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    sae = load_sae("jumprelu", args.sae_repo, sae_path,
                   sae_k=None).to(args.device).eval()
    w_dec = sae.W_dec        # (d_sae, d_model)
    add_hook = AdditionHook(w_dec)
    set_hook = SetHook(sae)
    model.model.layers[args.layer].register_forward_hook(add_hook)
    model.model.layers[args.layer].register_forward_hook(set_hook)

    rec_path = out_dir / f"records_{args.config}.jsonl"
    done = set()
    if rec_path.exists():
        for line in open(rec_path):
            try:
                r = json.loads(line)
                done.add((r["arm"], int(r["concept_id"]), int(r["input_id"]),
                          float(r["factor"])))
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[axrepro] RESUME: {len(done)} generations")
    fh = open(rec_path, "a")
    tok.padding_side = "left"                    # their predict_steer
    _probe = chat_wrap("PING12345 sanity instruction")
    _dec = tok.decode(_probe)
    assert "PING12345" in _dec, (
        f"chat_wrap lost the instruction on this env: {_dec[:160]!r}")

    pad_id = tok.pad_token_id or tok.eos_token_id

    @torch.no_grad()
    def gen(prompt_ids, seed_key):
        # prompt_ids: list of id-lists (chat_wrap). Manual LEFT padding —
        # no re-tokenization of decoded strings (see chat_wrap).
        T = max(len(p) for p in prompt_ids)
        ids = torch.full((len(prompt_ids), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompt_ids), T), dtype=torch.long)
        for i, p in enumerate(prompt_ids):
            ids[i, T - len(p):] = torch.tensor(p, dtype=torch.long)
            mask[i, T - len(p):] = 1
        ids, mask = ids.to(args.device), mask.to(args.device)
        torch.manual_seed(stable_hash(*seed_key) % (2**31))
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=args.temperature,
            pad_token_id=pad_id)
        return [tok.decode(o[T:], skip_special_tokens=True) for o in out]

    for cid, concept, vanilla in concepts:
        sa = stage_a[cid]
        wrapped = [chat_wrap(q) for q in instr[cid]]
        jobs = []                                # (arm, factor, latent, alpha)
        if "sae" in arms:
            jobs += [("sae", f, sa["vanilla"], f * sa["max_act_vanilla"])
                     for f in FACTORS]
        if "sae_a" in arms:
            jobs += [("sae_a", f, sa["sae_a"], f * sa["max_act_sae_a"])
                     for f in FACTORS]
        if "ll_set10" in arms:
            jobs += [("ll_set10", 1.0, sa["vanilla"], None)]
        for arm, factor, latent, alpha in jobs:
            missing = [i for i in range(args.num_instructions)
                       if (arm, cid, i, float(factor)) not in done]
            if not missing:
                continue
            add_hook.enabled = set_hook.enabled = False
            if arm == "ll_set10":
                set_hook.enabled = True
                set_hook.set_idx = torch.tensor([latent], device=args.device)
                set_hook.set_val = 10.0
            else:
                add_hook.enabled = True
                add_hook.vec = float(alpha) * w_dec[latent].to(args.device)
            for lo in range(0, len(missing), args.batch_size):
                rows = missing[lo:lo + args.batch_size]
                texts = gen([wrapped[i] for i in rows],
                            (arm, cid, factor, rows[0]))
                for i, t in zip(rows, texts):
                    fh.write(json.dumps({
                        "arm": arm, "concept_id": cid, "concept": concept,
                        "input_id": int(i), "factor": float(factor),
                        "latent": int(latent),
                        "instruction": instr[cid][i], "output": t,
                    }, ensure_ascii=False) + "\n")
                fh.flush()
        print(f"[axrepro] B {cid} done ({concept[:50]}...)")
    fh.close()

    (out_dir / f"meta_{args.config}.json").write_text(json.dumps({
        "config": args.config, "layer": args.layer, "sae_path": sae_path,
        "arms": arms, "num_concepts": args.num_concepts,
        "factors": FACTORS, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_act_source": "AxBench dataset (disable_neuronpedia_max_act "
                          "path) — NOT Neuronpedia maxValue",
        "anchors_table2_2b": {"sae": {"l10": 0.177, "l20": 0.151},
                              "sae_a": {"l10": 0.166, "l20": 0.132}},
    }, indent=2))
    print(f"[axrepro] GENERATION DONE -> {rec_path}")
    print("[axrepro] next (prepost): bash run_axbench_repro_judge.sh")


def sae_z_with_offsets_compat(extractor, text, device):
    """Adapter: reuse the project's canonical activation path. Verbatim
    signature: eval_lingualens.sae_z_with_offsets returns (offsets, z) with
    z (T, d_sae), tokenization truncated at 256 tokens."""
    from eval_lingualens import sae_z_with_offsets
    return sae_z_with_offsets(extractor, text, device)


if __name__ == "__main__":
    main()
