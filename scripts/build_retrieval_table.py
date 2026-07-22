"""Retrieval table for improvement A: per feature, every POOL pair's
source-side SAE signature (top max-act entries, both sides) + its delta.
Consumed by eval_ef_bare --fspec-retrieve.

Usage:
    python scripts/build_retrieval_table.py \
        --pairs runs/feature_specs/l12_pairs.jsonl \
        --split runs/tables/eval_split.json \
        --out runs/feature_specs/l12_retrieve.json --top-acts 64
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-acts", type=int, default=64,
                   help="max-act entries kept per side for the similarity")
    args = p.parse_args()

    eval_idx = set(json.loads(Path(args.split).read_text())["eval_idx"])
    tab = defaultdict(list)
    n = 0
    with open(args.pairs) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["idx"]) in eval_idx:
                continue

            def trim(d):
                it = sorted(d.items(), key=lambda kv: -kv[1])
                return {str(k): round(float(v), 4)
                        for k, v in it[:args.top_acts]}
            tab[r["feature"]].append({
                "m1": trim(r["max_s"]),          # sentence1 side (sup src)
                "m2": trim(r["max_t"]),          # sentence2 side (amp src)
                "d": {str(k): round(float(v), 5)
                      for k, v in r["delta"].items()},
            })
            n += 1
    Path(args.out).write_text(json.dumps(dict(tab)))
    print(f"[retrieve] {n} pool pairs across {len(tab)} features "
          f"-> {args.out}")
    print("RETRIEVE-TABLE-BUILT")


if __name__ == "__main__":
    main()
