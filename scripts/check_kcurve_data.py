"""
What k values are already on disk? (CPU, instant — no model loading)

run_paper_todo.sh swept --k-grid 1,2,4,8,16,32,64 over the original block, so
the small-k arm of the "how many features does it take to command an edit?"
curve may already exist and cost nothing to report. run_confirm1000.sh later
extended only k32 to the fresh pairs, so expect k1..k64 on ~499 and k32 on 997.

Usage:
    python scripts/check_kcurve_data.py \
        --records runs/prod_gemma_v6/ksweep500/records.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--condition", default="true")
    args = p.parse_args()

    n, per_mode = 0, defaultdict(list)
    with open(args.records) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            for mode, node in (r.get("outputs", {}).get(args.condition)
                               or {}).items():
                if isinstance(node, dict) and "exact" in node:
                    per_mode[mode].append(float(node["exact"]))

    print(f"[kcurve] {n} records in {args.records} (condition={args.condition})")
    print(f"\n| mode | pairs | exact |")
    print("|---|---|---|")

    def kof(m):
        return int(m[1:]) if m.startswith("k") and m[1:].isdigit() else 10**6
    for mode in sorted(per_mode, key=kof):
        v = per_mode[mode]
        print(f"| {mode} | {len(v)} | {sum(v)/len(v):.4f} |")

    ks = sorted((kof(m) for m in per_mode if kof(m) < 10**6))
    have = [k for k in (1, 2, 4, 8, 16, 32, 64) if k in ks]
    missing = [k for k in (1, 2, 4, 8, 16, 32, 64) if k not in ks]
    print(f"\n[kcurve] exact already on disk for k = {have}")
    if missing:
        print(f"[kcurve] MISSING (needs GPU re-run): k = {missing}")
    else:
        print("[kcurve] the whole exact curve is free — no GPU needed.")
    print("[kcurve] FRR still needs judging per k (gold cache is shared, so "
          "only the system side costs calls).")


if __name__ == "__main__":
    main()
