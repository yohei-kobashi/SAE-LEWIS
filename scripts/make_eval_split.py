"""Canonical eval/identification split of LinguaLens-Data (English).

User decision 2026-07-22: the seed-42 stdlib-shuffle 500 (the sample every
exact eval has always used) is THE eval set; the remaining ~4,451 pairs are
the identification pool from which per-feature interventions are built for
ALL arms (ef feature-spec, LinguaLens FRC, AxBench AUROC). This file is the
single source of truth — every identification/spec script must read it
instead of re-deriving the sample (fixes the np-rng/stdlib recipe mismatch:
the two 500s overlapped only 48/500).

Usage:
    python scripts/make_eval_split.py --out runs/tables/eval_split.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/tables/eval_split.json")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--dev-size", type=int, default=500,
                   help="v2 (2026-07-22): a disjoint dev set for ALL "
                        "hyperparameter selection — the next dev-size "
                        "indices of the SAME shuffle after the eval "
                        "prefix. train = the rest (identification only).")
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    n = len(ds)
    order = list(range(n))
    random.Random(args.seed).shuffle(order)
    eval_idx = sorted(order[: min(args.sample_size, n)])
    dev_idx = sorted(order[args.sample_size:
                           args.sample_size + args.dev_size])
    per_feat = {}
    for i in eval_idx:
        f = ds[int(i)]["feature"] or "?"
        per_feat[f] = per_feat.get(f, 0) + 1
    out = {
        "dataset": args.dataset,
        "language": args.language,
        "n_total": n,
        "recipe": f"random.Random({args.seed}).shuffle(range(n))"
                  f"[:{args.sample_size}] — identical to every exact eval",
        "eval_idx": eval_idx,
        "dev_idx": dev_idx,
        "n_eval": len(eval_idx),
        "n_dev": len(dev_idx),
        "n_pool": n - len(eval_idx),
        "n_train": n - len(eval_idx) - len(dev_idx),
        "eval_per_feature": dict(sorted(per_feat.items())),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"[split] {n} pairs -> test {len(eval_idx)} / dev "
          f"{len(dev_idx)} / train {n - len(eval_idx) - len(dev_idx)}; "
          f"wrote {p}")


if __name__ == "__main__":
    main()
