"""
Per-phenomenon FRR aggregation (CPU seconds) — the LinguaLens-basis
companion of the per-feature exact table.

Reads FRR judgment jsonl files (scripts/judge_feature_realization.py
output; each row carries feature / realized / copy) and aggregates FRR
per LinguaLens feature per system. When a matching random-condition
file is given (label_rnd=...), the per-feature net-FRR (true - random,
the specificity) is included.

Usage:
    python scripts/frr_per_feature.py \
        --frr routed=runs/frr_final/.../routed.jsonl \
        --frr routed_rnd=runs/frr_final/.../routed_rnd.jsonl \
        --frr ef32=... --frr steer=... \
        --out runs/tables/frr_per_feature
Systems ending in `_rnd` are treated as the random control of the
same-named base system.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frr", action="append", required=True,
                   help="label=judgments.jsonl (label ending `_rnd` = "
                        "random control of the base label)")
    p.add_argument("--min-n", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = {}
    for spec in args.frr:
        label, path = spec.split("=", 1)
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        data[label] = rows
        print(f"[frr-feat] {label}: {len(rows)} judgments")

    bases = [l for l in data if not l.endswith("_rnd")]

    def frr_of(rows):
        scored = [r for r in rows if r.get("realized") is not None]
        if not scored:
            return None, 0
        return (sum(1 for r in scored if r["realized"]) / len(scored),
                len(scored))

    # system-level summary
    lines = ["# Per-phenomenon FRR (LinguaLens-basis)", "",
             "| system | FRR | net-FRR (true - random) | scored |",
             "|---|---|---|---|"]
    for b in bases:
        f_t, n_t = frr_of(data[b])
        rnd = data.get(b + "_rnd")
        f_r = frr_of(rnd)[0] if rnd else None
        net = f"{f_t - f_r:.4f}" if (f_t is not None and f_r is not None) \
            else "—"
        lines.append(f"| {b} | {f_t:.4f} | {net} | {n_t} |")

    # per-feature
    feats = defaultdict(lambda: defaultdict(list))
    for label, rows in data.items():
        for r in rows:
            feats[r.get("feature") or "?"][label].append(r)
    order = sorted(feats, key=lambda f: -len(next(
        (feats[f][b] for b in bases if b in feats[f]), [])))

    csv = ["feature,n," + ",".join(
        f"{b}_frr,{b}_net" for b in bases)]
    lines += ["", f"## Per-feature FRR (n >= {args.min_n}; full table in "
                  f"the CSV)", "",
              "| feature | n | " + " | ".join(
                  f"{b} (net)" for b in bases) + " |",
              "|---" * (2 + len(bases)) + "|"]
    for f in order:
        n_ref = max((len(feats[f][b]) for b in bases if b in feats[f]),
                    default=0)
        cells_md, cells_csv = [], [f'"{f}"', str(n_ref)]
        for b in bases:
            f_t, _ = frr_of(feats[f].get(b, []))
            rnd_rows = feats[f].get(b + "_rnd", [])
            f_r = frr_of(rnd_rows)[0] if rnd_rows else None
            if f_t is None:
                cells_md.append("—")
                cells_csv.extend(["", ""])
                continue
            net = (f"{f_t - f_r:.3f}" if f_r is not None else "—")
            cells_md.append(f"{f_t:.3f} ({net})")
            cells_csv.append(f"{f_t:.4f}")
            cells_csv.append(f"{f_t - f_r:.4f}" if f_r is not None else "")
        csv.append(",".join(cells_csv))
        if n_ref >= args.min_n:
            lines.append(f"| {f} | {n_ref} | " + " | ".join(cells_md)
                         + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    print(report)
    Path(str(out) + ".md").write_text(report + "\n")
    Path(str(out) + ".csv").write_text("\n".join(csv) + "\n")
    print(f"[frr-feat] wrote {out}.md, {out}.csv")


if __name__ == "__main__":
    main()
