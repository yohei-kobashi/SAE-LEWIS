"""Per-latent max activation over the identification POOL (AxBench's
max_act, computed dataset-side — their sanctioned alternative to
Neuronpedia — but on the pool only, so the eval 500 never contributes).

Reads the build_feature_specs sidecar (max_s/max_t sparse dicts per pair)
and emits {phenomenon: {latent: max_act}} for the latents named in the
given selection JSON (e.g. l{L}_auroc_r1.json).

Usage:
    python scripts/extract_pool_maxact.py \
        --pairs runs/feature_specs/l12_pairs.jsonl \
        --sets runs/feature_specs/l12_auroc_r1.json \
        --split runs/tables/eval_split.json \
        --out runs/feature_specs/l12_auroc_r1_maxact.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    eval_idx = set(json.loads(Path(args.split).read_text())["eval_idx"])
    sets = json.loads(Path(args.sets).read_text())
    want = {ph: {int(f) for f, _ in lst} for ph, lst in sets.items()}

    mx = defaultdict(lambda: defaultdict(float))
    n = 0
    with open(args.pairs) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["idx"]) in eval_idx:
                continue
            ph = r["feature"]
            if ph not in want:
                continue
            for side in ("max_s", "max_t"):
                for f, v in r[side].items():
                    fi = int(f)
                    if fi in want[ph]:
                        mx[ph][fi] = max(mx[ph][fi], float(v))
            n += 1
    out = {ph: {str(f): round(v, 5) for f, v in d.items()}
           for ph, d in mx.items()}
    Path(args.out).write_text(json.dumps(out))
    zero = [ph for ph in want if not out.get(ph)]
    print(f"[maxact] {n} pool pairs; {len(out)} phenomena; "
          f"no-activation phenomena: {zero if zero else 'none'}")
    print(f"[maxact] wrote {args.out}")


if __name__ == "__main__":
    main()
