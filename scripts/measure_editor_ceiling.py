"""
Measure the editor's headroom: gold-in-top-k curves by position type and
conditioning richness.

`repl_top1 = 0.21` alone cannot distinguish "undertrained" from "the task
is intrinsically ambiguous given corrupted context + a handful of SAE diff
features". This script separates the two:

  * gold-in-top-k (k = 1, 5, 10, 50, 100) at template REPL ([MASK]) and
    INS ([INS]) positions — if top-50 saturates near 1.0 while top-1 is
    low, the candidate set is right and ranking/conditioning has headroom;
    if even top-50 is low, the context+conditioning does not determine the
    answer and the ceiling is structural (op granularity / conditioning
    information content).
  * split by the number of nonzero conditioning features actually placed
    (0 = empty, 1-2, 3-4, 5-8) — if the curves do not improve with more
    features, the conditioning carries no usable content for the editor
    regardless of training.
  * an `empty` row for the same samples as the paired baseline.

Usage:
    python scripts/measure_editor_ceiling.py \
        --corruption-dir runs/prod_gemma_v3/corruption_dev \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --editor-ckpt runs/prod_gemma_v3/editor/editor-final.pt \
        --output-dir runs/prod_gemma_v3/ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import CorruptionCollator, CorruptionDataset            # noqa: E402
from editor import load_editor_from_checkpoint                    # noqa: E402
from intervene import diff_to_sparse                              # noqa: E402

K_LIST = (1, 5, 10, 50, 100)
FEAT_BUCKETS = ("empty", "1-2", "3-4", "5-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--editor-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--k-top", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def feat_bucket(n: int) -> str:
    if n == 0:
        return "empty"
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    return "5-8"


def main():
    args = parse_args()
    set_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    mask_id = int(tok.mask_token_id)
    ins_id = int(tok.convert_tokens_to_ids("[INS]"))

    blob = torch.load(args.editor_ckpt, map_location="cpu")
    d_sae = int(blob["d_sae"])
    del blob
    editor = load_editor_from_checkpoint(
        args.llm2vec_dir, args.editor_ckpt, d_sae=d_sae, dtype=dtype,
    ).to(args.device).eval()

    ds = CorruptionDataset(args.corruption_dir, shuffle=False,
                           seed=args.seed, infinite=False)
    coll = CorruptionCollator(
        d_sae=d_sae, pad_token_id=tok.pad_token_id,
        sep_token_id=tok.convert_tokens_to_ids("[SEP]"),
        del_token_id=tok.convert_tokens_to_ids("[DEL]"),
        bos_token_id=tok.bos_token_id,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=coll)

    rng = np.random.default_rng(args.seed)
    max_k = max(K_LIST)
    # hits[cond][pos_type][bucket][k] ; n[cond][pos_type][bucket]
    hits = {c: {pt: defaultdict(lambda: defaultdict(int))
                for pt in ("repl", "ins")} for c in ("true", "empty")}
    counts = {c: {pt: defaultdict(int) for pt in ("repl", "ins")}
              for c in ("true", "empty")}
    seen = 0

    for batch in tqdm(loader, desc="[ceiling]", unit="batch"):
        B = batch["z_X"].shape[0]
        z_amp = torch.zeros_like(batch["z_X"])
        z_sup = torch.zeros_like(batch["z_X"])
        n_feats = []
        for b in range(B):
            k_amp = int(rng.integers(1, 5))
            k_sup = int(rng.integers(1, 5))
            a, s = diff_to_sparse(
                batch["z_X"][b], batch["z_X_prime"][b],
                k_top=args.k_top, k_amp=k_amp, k_sup=k_sup,
                rng=rng, empty_conditioning_prob=0.0,
            )
            z_amp[b], z_sup[b] = a, s
            n_feats.append(int((a > 0).sum()) + int((s > 0).sum()))

        ids = batch["editor_input_ids"].to(args.device)
        attn = batch["editor_attention_mask"].to(args.device)
        tgt = batch["editor_target_ids"]
        valid = tgt != -100
        pos_masks = {
            "repl": (batch["editor_input_ids"] == mask_id) & valid,
            "ins": (batch["editor_input_ids"] == ins_id) & valid,
        }

        for cond in ("true", "empty"):
            if cond == "true":
                za, zs = z_amp, z_sup
            else:
                za = torch.zeros_like(z_amp)
                zs = torch.zeros_like(z_sup)
            with torch.no_grad():
                out = editor(ids, attn, za.to(args.device), zs.to(args.device))
                logits = out["logits"].float()
            for pt, pm in pos_masks.items():
                for b in range(B):
                    m = pm[b]
                    if not m.any():
                        continue
                    bucket = "empty" if cond == "empty" else feat_bucket(n_feats[b])
                    lg = logits[b][m.to(args.device)]
                    gold = tgt[b][m].to(args.device)
                    topk = lg.topk(max_k, dim=-1).indices          # (P, max_k)
                    match = (topk == gold.unsqueeze(-1))           # (P, max_k)
                    counts[cond][pt][bucket] += int(m.sum())
                    for k in K_LIST:
                        hits[cond][pt][bucket][k] += int(
                            match[:, :k].any(dim=-1).sum())

        seen += B
        if seen >= args.max_samples:
            break

    # ---- report --------------------------------------------------------- #
    def agg(cond, pt, buckets):
        n = sum(counts[cond][pt][b] for b in buckets)
        row = {}
        for k in K_LIST:
            h = sum(hits[cond][pt][b][k] for b in buckets)
            row[k] = h / n if n else float("nan")
        return n, row

    lines = ["# Editor headroom: gold-in-top-k", "",
             f"samples: {seen}  (cache: {args.corruption_dir})", ""]
    payload = {"n_samples": seen, "tables": {}}
    for pt in ("repl", "ins"):
        lines += [f"## {pt.upper()} positions", "",
                  "| conditioning | positions | " +
                  " | ".join(f"top-{k}" for k in K_LIST) + " |",
                  "|---|---|" + "---|" * len(K_LIST)]
        tbl = {}
        n, row = agg("true", pt, [b for b in FEAT_BUCKETS if b != "empty"])
        lines.append(f"| true (all) | {n} |" +
                     "".join(f" {row[k]:.4f} |" for k in K_LIST))
        tbl["true_all"] = {"n": n, **{f"top{k}": row[k] for k in K_LIST}}
        for b in ("1-2", "3-4", "5-8"):
            n, row = agg("true", pt, [b])
            lines.append(f"| true, {b} feats | {n} |" +
                         "".join(f" {row[k]:.4f} |" for k in K_LIST))
            tbl[f"true_{b}"] = {"n": n, **{f"top{k}": row[k] for k in K_LIST}}
        n, row = agg("empty", pt, list(FEAT_BUCKETS))
        lines.append(f"| empty | {n} |" +
                     "".join(f" {row[k]:.4f} |" for k in K_LIST))
        tbl["empty"] = {"n": n, **{f"top{k}": row[k] for k in K_LIST}}
        lines.append("")
        payload["tables"][pt] = tbl

    lines += [
        "Reading guide:",
        "- top-50 high & top-1 low → ranking/conditioning headroom "
        "(scale training, improve selection); top-50 low → structural "
        "ceiling (context + features underdetermine the answer; op design "
        "or richer conditioning needed).",
        "- true vs empty gap growing with feature count → the editor "
        "extracts content from the conditioning; flat → it does not.",
        "",
    ]
    (out_dir / "ceiling_report.md").write_text("\n".join(lines))
    (out_dir / "ceiling_metrics.json").write_text(json.dumps(payload, indent=2))
    print("\n".join(lines))
    print(f"[ceiling] wrote {out_dir}/ceiling_report.md")


if __name__ == "__main__":
    main()
