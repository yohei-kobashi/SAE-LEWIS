"""Build a judge-ready records file for a NEW ef-arm variant.

Takes the layer's original probe records (which carry the empty/raw
reference repeats — model-independent) and replaces the ef outputs with
the given variant's outputs (e.g. the adopted T2 model or the T4-adapted
model). All other arms are STRIPPED so the judge scores only the ef rows
(steer/prompting/clamp/axbench have their own judge dirs).

Usage:
    python scripts/merge_ef_records.py \
        --base runs/prod_gemma_v6/fs_probe_l12/records.jsonl \
        --ef   runs/prod_gemma_v6/fs_v6t2_l12/records.jsonl \
        --out  runs/prod_gemma_v6/fic_ad_l12/records_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--ef", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ef = {}
    for line in open(args.ef):
        if line.strip():
            r = json.loads(line)
            ef[int(r["idx"])] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = n_miss = 0
    with open(out, "w") as w:
        for line in open(args.base):
            if not line.strip():
                continue
            r = json.loads(line)
            src = ef.get(int(r["idx"]))
            if src is None:
                n_miss += 1
                continue
            outputs = {}
            raw = r["outputs"].get("empty", {}).get("raw")
            if raw is None:
                n_miss += 1
                continue
            outputs["empty"] = {"raw": raw}
            ok = True
            for cond in ("true", "random"):
                t = src["outputs"].get(cond, {}).get("ef")
                if t is None:
                    ok = False
                    break
                outputs[cond] = {"ef": t}
            if not ok:
                n_miss += 1
                continue
            r2 = {k: r[k] for k in ("idx", "src", "tgt", "feature")
                  if k in r}
            r2["outputs"] = outputs
            w.write(json.dumps(r2, ensure_ascii=False) + "\n")
            n += 1
    print(f"[merge] {n} rows -> {out} ({n_miss} skipped)")


if __name__ == "__main__":
    main()
