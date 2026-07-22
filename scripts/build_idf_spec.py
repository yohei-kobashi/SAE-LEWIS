"""Improvement ③ (user-approved 2026-07-22): sensitivity-normalized specs.

Generic-corpus SAEs give large raw deltas to frequent/high-variance
latents (topic, register) that carry little feature information. This
script reweights the per-feature pool-mean delta with GLOBAL per-latent
statistics computed over ALL pool pairs before re-taking top-k:

  idf:    w_i = log((N + 1) / (df_i + 1)),  df_i = #pairs with delta_i != 0
  invstd: w_i = 1 / (std_i + eps),          std over all pool pairs
                                            (zeros included)

Output schema matches l{L}_spec.json (mean values are reweighted; the
norm-median rescale target is passed through) so eval_ef_bare
--feature-spec consumes it unchanged.

Usage:
    python scripts/build_idf_spec.py \
        --pairs runs/feature_specs/l12_pairs.jsonl \
        --split runs/tables/eval_split.json \
        --base runs/feature_specs/l12_spec.json \
        --out-prefix runs/feature_specs/l12_spec
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--top-store", type=int, default=128)
    p.add_argument("--d-sae", type=int, default=16384)
    args = p.parse_args()

    eval_idx = set(json.loads(Path(args.split).read_text())["eval_idx"])
    base = json.loads(Path(args.base).read_text())

    df = np.zeros(args.d_sae)
    s1 = np.zeros(args.d_sae)
    s2 = np.zeros(args.d_sae)
    acc = defaultdict(lambda: defaultdict(float))
    npairs = defaultdict(int)
    N = 0
    with open(args.pairs) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["idx"]) in eval_idx:
                continue
            N += 1
            ph = r["feature"]
            npairs[ph] += 1
            for i, v in r["delta"].items():
                i = int(i)
                v = float(v)
                df[i] += 1
                s1[i] += v
                s2[i] += v * v
                acc[ph][i] += v
    mean_g = s1 / N
    var_g = np.maximum(s2 / N - mean_g ** 2, 0.0)
    std_g = np.sqrt(var_g)
    w_idf = np.log((N + 1) / (df + 1))
    w_idf = w_idf / w_idf.max()
    eps = np.median(std_g[std_g > 0])
    w_std = eps / (std_g + eps)          # 1 at median-std, <1 for noisy

    for tag, w in (("_idf", w_idf), ("_invstd", w_std)):
        out = {}
        for ph, comp in acc.items():
            n = npairs[ph]
            weighted = sorted(((i, (s / n) * w[i])
                               for i, s in comp.items()),
                              key=lambda x: -abs(x[1]))
            keep = [(i, v) for i, v in weighted[:args.top_store]
                    if abs(v) > 0]
            mn = float(math.sqrt(sum(v * v for _, v in keep)))
            out[ph] = {
                "n": n,
                "spec": {int(i): round(float(v), 6) for i, v in keep},
                "mean_norm": round(mn, 6),
                "norm_median": base[ph]["norm_median"],
                "splithalf_cos": base[ph].get("splithalf_cos"),
            }
        path = f"{args.out_prefix}{tag}.json"
        Path(path).write_text(json.dumps(out))
        print(f"[idf] wrote {path} ({len(out)} features)")
    print("IDF-SPECS-BUILT")


if __name__ == "__main__":
    main()
