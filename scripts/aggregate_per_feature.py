"""Per-feature aggregation over probe records (user request 2026-07-19:
every result should also be readable per feature; the optimal top-k
likely differs by feature). Pure stdlib -> runs anywhere (login node).

Modes:
  * single records file: per-feature exact/sim/copy per arm x condition
      python3 scripts/aggregate_per_feature.py \
          --records runs/prod_gemma_v6/eflm_l12_v5f/probe500/records.jsonl \
          --out runs/tables/perfeature_v5f_l12.md
  * k-sweep: label=path pairs; per-feature exact-vs-k matrix (ef/true),
    per-feature argmax k, and the oracle per-feature-k aggregate
      python3 scripts/aggregate_per_feature.py --out ... \
          --sweep k1=...:k4=...:k8=...:k16=...:k32=...:k64=...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def per_feature_table(records, min_n=1):
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        f = r.get("feature") or "?"
        for c, arms_d in r.get("outputs", {}).items():
            for a, m in arms_d.items():
                by[f][(a, c)].append(m)
    lines = ["| feature | n | " + " | ".join(
        f"{a}/{c}" for (a, c) in sorted(
            {k for f in by for k in by[f]})) + " |"]
    keys = sorted({k for f in by for k in by[f]})
    lines.append("|" + "---|" * (len(keys) + 2))
    for f in sorted(by):
        n = max(len(v) for v in by[f].values())
        if n < min_n:
            continue
        cells = []
        for k in keys:
            ms = by[f].get(k, [])
            cells.append(f"{mean([m['exact'] for m in ms]):.3f}"
                         if ms else "—")
        lines.append(f"| {f} | {n} | " + " | ".join(cells) + " |")
    return lines


def sweep_table(sweep_spec):
    labels, paths = [], []
    for part in sweep_spec.split(":"):
        lab, p = part.split("=", 1)
        labels.append(lab)
        paths.append(p)
    per = {}                      # feature -> label -> [exact]
    for lab, p in zip(labels, paths):
        for r in load(p):
            f = r.get("feature") or "?"
            m = r.get("outputs", {}).get("true", {}).get("ef")
            if m:
                per.setdefault(f, defaultdict(list))[lab].append(
                    m["exact"])
    lines = ["| feature | n | " + " | ".join(labels)
             + " | best | best_exact |",
             "|" + "---|" * (len(labels) + 4)]
    oracle_vals, fixed_last = [], []
    for f in sorted(per):
        vals = {lab: mean(per[f].get(lab, [])) for lab in labels}
        n = max((len(v) for v in per[f].values()), default=0)
        best = max((lab for lab in labels
                    if vals[lab] == vals[lab]),
                   key=lambda x: vals[x], default="—")
        row = [f"{vals[lab]:.3f}" if vals[lab] == vals[lab] else "—"
               for lab in labels]
        lines.append(f"| {f} | {n} | " + " | ".join(row)
                     + f" | {best} | {vals.get(best, float('nan')):.3f} |")
        if best != "—":
            oracle_vals.extend([max(v for v in
                                    (vals[lab] for lab in labels)
                                    if v == v)] * n)
        last = vals.get(labels[-1], float("nan"))
        if last == last:
            fixed_last.extend([last] * n)
    lines += ["",
              f"**oracle per-feature k** (weighted): "
              f"{mean(oracle_vals):.4f}  vs fixed {labels[-1]}: "
              f"{mean(fixed_last):.4f}"]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="")
    p.add_argument("--sweep", default="",
                   help="label=path[:label=path...] of k-sweep records")
    p.add_argument("--out", required=True)
    p.add_argument("--min-n", type=int, default=1)
    args = p.parse_args()

    lines = ["# Per-feature aggregation", ""]
    if args.records:
        lines += [f"source: {args.records} (exact per arm/condition)", ""]
        lines += per_feature_table(load(args.records), args.min_n)
    if args.sweep:
        lines += ["", "## exact vs k per feature (arm=ef, cond=true)", ""]
        lines += sweep_table(args.sweep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
