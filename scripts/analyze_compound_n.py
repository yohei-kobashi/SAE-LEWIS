"""
scripts/analyze_compound_n.py — summarise calibration.jsonl per N.

Reads `<run_dir>/n{N}/calibration.jsonl` for each N value and produces:

  * Per-N distribution stats for ppl_ratio and sae_shift (count, mean,
    median, percentiles).
  * Per-N op-type histogram (REPL / INS / DEL).
  * Yield simulation under the default sqrt(N) gate AND a sweep of
    alternate ppl_per_op_factor / sae_per_op_min / sae_per_op_max
    settings, so the user can see how the threshold-fit shifts yield
    across N.

Output:
  * Tab-separated `report.tsv` with one row per N.
  * Markdown `report.md` with the same table plus the gate-sweep
    section. Both files are written under <run_dir>.

Usage:
  python scripts/analyze_compound_n.py <run_dir>
  python scripts/analyze_compound_n.py <run_dir> --n-values 0 1 2 3 4 5
  python scripts/analyze_compound_n.py <run_dir> --report-tsv X.tsv --report-md X.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def percentiles(values: List[float], qs: List[float]) -> List[float]:
    """Linear-interpolation percentiles (no numpy dependency). Returns `nan`
    for empty input or all-`nan` input.
    """
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return [float("nan")] * len(qs)
    out: List[float] = []
    for q in qs:
        if q <= 0:
            out.append(clean[0])
            continue
        if q >= 1:
            out.append(clean[-1])
            continue
        pos = q * (len(clean) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(clean[lo])
        else:
            frac = pos - lo
            out.append(clean[lo] * (1 - frac) + clean[hi] * frac)
    return out


def mean(values: List[float]) -> float:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(clean) if clean else float("nan")


# --------------------------------------------------------------------------- #
# Calibration record loader
# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Per-N summary
# --------------------------------------------------------------------------- #
def summarise_one_N(records: List[Dict]) -> Dict:
    ppl_ratios = [r.get("ppl_ratio") for r in records]
    sae_shifts = [r.get("sae_shift") for r in records]
    op_counter: Counter = Counter()
    for r in records:
        for t in r.get("op_types", []):
            op_counter[t] += 1
    n_total = len(records)
    ppl_pcts = percentiles(ppl_ratios, [0.25, 0.50, 0.75, 0.95])
    sae_pcts = percentiles(sae_shifts, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n_records": n_total,
        "ppl_mean": mean(ppl_ratios),
        "ppl_p25": ppl_pcts[0],
        "ppl_p50": ppl_pcts[1],
        "ppl_p75": ppl_pcts[2],
        "ppl_p95": ppl_pcts[3],
        "sae_mean": mean(sae_shifts),
        "sae_p05": sae_pcts[0],
        "sae_p25": sae_pcts[1],
        "sae_p50": sae_pcts[2],
        "sae_p75": sae_pcts[3],
        "sae_p95": sae_pcts[4],
        "op_counts": dict(op_counter),
    }


# --------------------------------------------------------------------------- #
# Gate simulation
# --------------------------------------------------------------------------- #
def simulate_gate_yield(
    records: List[Dict], N: int,
    ppl_per_op_factor: float,
    sae_per_op_min: float,
    sae_per_op_max: float,
) -> Dict:
    """Compute the fraction of `records` that would pass the gate at the
    given scale constants.

    For N=0 (identity) the gate is no-op and yield is 100% by definition.
    """
    if N <= 0 or not records:
        return {"n": len(records), "pass_frac": 1.0,
                "rej_ppl_inf": 0, "rej_ppl_high": 0,
                "rej_sae_low": 0, "rej_sae_high": 0}
    s = math.sqrt(N)
    ppl_max = ppl_per_op_factor ** s
    sae_min = sae_per_op_min * s
    sae_max = sae_per_op_max * s
    passed = 0
    r_ppl_inf = r_ppl_high = r_sae_low = r_sae_high = 0
    for r in records:
        ppl_ratio = r.get("ppl_ratio")
        shift = r.get("sae_shift")
        if ppl_ratio is None or not math.isfinite(ppl_ratio):
            r_ppl_inf += 1
            continue
        if ppl_ratio > ppl_max:
            r_ppl_high += 1
            continue
        if shift is None or not math.isfinite(shift):
            r_sae_low += 1
            continue
        if shift < sae_min:
            r_sae_low += 1
            continue
        if shift > sae_max:
            r_sae_high += 1
            continue
        passed += 1
    total = len(records)
    return {
        "n": total,
        "pass_frac": passed / total,
        "rej_ppl_inf": r_ppl_inf,
        "rej_ppl_high": r_ppl_high,
        "rej_sae_low": r_sae_low,
        "rej_sae_high": r_sae_high,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
TSV_COLUMNS = [
    "N", "records",
    "ppl_mean", "ppl_p25", "ppl_p50", "ppl_p75", "ppl_p95",
    "sae_mean", "sae_p05", "sae_p25", "sae_p50", "sae_p75", "sae_p95",
    "op_REPL", "op_INS", "op_DEL",
    "yield_default", "ppl_max_default", "sae_min_default", "sae_max_default",
]


def format_tsv_row(N: int, summary: Dict, gate: Dict,
                   ppl_per_op_factor: float,
                   sae_per_op_min: float, sae_per_op_max: float) -> List[str]:
    s = math.sqrt(N) if N > 0 else 0.0
    return [
        str(N),
        str(summary["n_records"]),
        f"{summary['ppl_mean']:.3f}",
        f"{summary['ppl_p25']:.3f}",
        f"{summary['ppl_p50']:.3f}",
        f"{summary['ppl_p75']:.3f}",
        f"{summary['ppl_p95']:.3f}",
        f"{summary['sae_mean']:.3f}",
        f"{summary['sae_p05']:.3f}",
        f"{summary['sae_p25']:.3f}",
        f"{summary['sae_p50']:.3f}",
        f"{summary['sae_p75']:.3f}",
        f"{summary['sae_p95']:.3f}",
        str(summary["op_counts"].get("REPL", 0)),
        str(summary["op_counts"].get("INS", 0)),
        str(summary["op_counts"].get("DEL", 0)),
        f"{gate['pass_frac']:.3f}",
        f"{ppl_per_op_factor ** s:.3f}" if N > 0 else "inf",
        f"{sae_per_op_min * s:.3f}",
        f"{sae_per_op_max * s:.3f}",
    ]


def render_md(
    rows: List[Dict],
    sweeps: List[Dict],
    ppl_per_op_factor: float,
    sae_per_op_min: float,
    sae_per_op_max: float,
) -> str:
    out: List[str] = []
    out.append("# Compound-N corruption measurement\n")
    out.append("Default gates (calibrated knobs at the corruption.py defaults):")
    out.append(f"  * `ppl_per_op_factor` = {ppl_per_op_factor}  →  ppl_max(N) = factor^√N")
    out.append(f"  * `sae_per_op_min`    = {sae_per_op_min}  →  sae_min(N) = min · √N")
    out.append(f"  * `sae_per_op_max`    = {sae_per_op_max}  →  sae_max(N) = max · √N\n")

    # Main table — distributions per N
    out.append("## Per-N distributions\n")
    out.append("| N | records | PPL p25 | PPL p50 | PPL p75 | PPL p95 | shift p25 | shift p50 | shift p75 | shift p95 |")
    out.append("|---|---------|---------|---------|---------|---------|-----------|-----------|-----------|-----------|")
    for row in rows:
        s = row["summary"]
        out.append(
            f"| {row['N']} | {s['n_records']} | "
            f"{s['ppl_p25']:.2f} | {s['ppl_p50']:.2f} | {s['ppl_p75']:.2f} | {s['ppl_p95']:.2f} | "
            f"{s['sae_p25']:.2f} | {s['sae_p50']:.2f} | {s['sae_p75']:.2f} | {s['sae_p95']:.2f} |"
        )

    # Default gate yields
    out.append("\n## Yield under the default gate\n")
    out.append("| N | ppl_max(N) | sae_min(N) | sae_max(N) | yield | rej PPL inf | rej PPL high | rej SAE low | rej SAE high |")
    out.append("|---|-----------|------------|------------|-------|-------------|--------------|-------------|--------------|")
    for row in rows:
        N = row["N"]
        s = math.sqrt(N) if N > 0 else 0.0
        ppl_max = ppl_per_op_factor ** s if N > 0 else float("inf")
        sae_min = sae_per_op_min * s
        sae_max = sae_per_op_max * s
        g = row["gate_default"]
        ppl_max_str = "∞" if not math.isfinite(ppl_max) else f"{ppl_max:.2f}"
        out.append(
            f"| {N} | {ppl_max_str} | {sae_min:.2f} | {sae_max:.2f} | "
            f"{g['pass_frac']*100:.1f}% | {g['rej_ppl_inf']} | {g['rej_ppl_high']} | "
            f"{g['rej_sae_low']} | {g['rej_sae_high']} |"
        )

    # Op-type histogram
    out.append("\n## Per-N op-type histogram\n")
    out.append("Total op counts (sum of all ops across all records at each N).\n")
    out.append("| N | REPL | INS | DEL |")
    out.append("|---|------|-----|-----|")
    for row in rows:
        ops = row["summary"]["op_counts"]
        out.append(
            f"| {row['N']} | {ops.get('REPL', 0)} | "
            f"{ops.get('INS', 0)} | {ops.get('DEL', 0)} |"
        )

    # Sweep
    if sweeps:
        out.append("\n## Gate sweep (yield under alternate scale constants)\n")
        out.append("Each row varies one scale constant while keeping the others at the default.")
        out.append("Shown is the per-N yield (% passing the gate) for the sampled records.\n")
        # Header from the union of N values in any sweep
        n_values_seen = sorted({row["N"] for row in rows})
        header = "| sweep | " + " | ".join(f"N={n}" for n in n_values_seen) + " |"
        sep = "|" + "|".join("---" for _ in range(len(n_values_seen) + 1)) + "|"
        out.append(header)
        out.append(sep)
        for sweep in sweeps:
            yields = sweep["per_n_pass"]
            row = "| " + sweep["label"] + " | " + \
                  " | ".join(
                      f"{yields[n]*100:.1f}%" if n in yields else "-"
                      for n in n_values_seen
                  ) + " |"
            out.append(row)

    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Output of scripts/measure_compound_n.sh "
                                    "(containing n{N}/calibration.jsonl files)")
    ap.add_argument("--n-values", nargs="*", type=int, default=None,
                    help="N values to look for. Default = autodetect from "
                         "directories named nK under run_dir.")
    ap.add_argument("--ppl-per-op-factor", type=float, default=1.8)
    ap.add_argument("--sae-per-op-min", type=float, default=0.30)
    ap.add_argument("--sae-per-op-max", type=float, default=2.50)
    ap.add_argument("--report-tsv", help="Path to write TSV summary "
                                          "(default: <run_dir>/report.tsv)")
    ap.add_argument("--report-md", help="Path to write Markdown report "
                                         "(default: <run_dir>/report.md)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    if args.n_values is None:
        n_values: List[int] = []
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("n") and (sub / "calibration.jsonl").exists():
                try:
                    n_values.append(int(sub.name[1:]))
                except ValueError:
                    pass
        n_values.sort()
    else:
        n_values = sorted(set(args.n_values))
    if not n_values:
        raise SystemExit(f"no calibration.jsonl files found under {run_dir}")
    print(f"[analyze-N] N values: {n_values}")

    # Load + summarise per-N
    rows: List[Dict] = []
    for N in n_values:
        cal_path = run_dir / f"n{N}" / "calibration.jsonl"
        if not cal_path.exists():
            print(f"[analyze-N] skip N={N}: {cal_path} missing")
            continue
        records = load_calibration(cal_path)
        summary = summarise_one_N(records)
        gate = simulate_gate_yield(
            records, N,
            args.ppl_per_op_factor, args.sae_per_op_min, args.sae_per_op_max,
        )
        rows.append({"N": N, "records": records, "summary": summary,
                     "gate_default": gate})

    # Gate sweep — vary each scale knob independently.
    sweeps: List[Dict] = []
    ppl_factor_grid = [1.4, 1.6, 1.8, 2.0, 2.4]
    sae_min_grid = [0.15, 0.30, 0.50]
    sae_max_grid = [1.50, 2.50, 4.00]
    for fac in ppl_factor_grid:
        per_n_pass = {}
        for row in rows:
            g = simulate_gate_yield(
                row["records"], row["N"],
                fac, args.sae_per_op_min, args.sae_per_op_max,
            )
            per_n_pass[row["N"]] = g["pass_frac"]
        sweeps.append({"label": f"ppl_factor={fac}", "per_n_pass": per_n_pass})
    for smin in sae_min_grid:
        per_n_pass = {}
        for row in rows:
            g = simulate_gate_yield(
                row["records"], row["N"],
                args.ppl_per_op_factor, smin, args.sae_per_op_max,
            )
            per_n_pass[row["N"]] = g["pass_frac"]
        sweeps.append({"label": f"sae_min={smin}", "per_n_pass": per_n_pass})
    for smax in sae_max_grid:
        per_n_pass = {}
        for row in rows:
            g = simulate_gate_yield(
                row["records"], row["N"],
                args.ppl_per_op_factor, args.sae_per_op_min, smax,
            )
            per_n_pass[row["N"]] = g["pass_frac"]
        sweeps.append({"label": f"sae_max={smax}", "per_n_pass": per_n_pass})

    # TSV
    tsv_path = Path(args.report_tsv) if args.report_tsv else (run_dir / "report.tsv")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "wt", encoding="utf-8") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            cells = format_tsv_row(
                row["N"], row["summary"], row["gate_default"],
                args.ppl_per_op_factor, args.sae_per_op_min, args.sae_per_op_max,
            )
            f.write("\t".join(cells) + "\n")
    print(f"[analyze-N] wrote {tsv_path}")

    # Markdown
    md_path = Path(args.report_md) if args.report_md else (run_dir / "report.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_md(
        rows, sweeps,
        args.ppl_per_op_factor, args.sae_per_op_min, args.sae_per_op_max,
    ))
    print(f"[analyze-N] wrote {md_path}")

    # Console summary
    print()
    print("Per-N summary:")
    print(f"  {'N':>3} | {'records':>7} | {'PPL p50':>8} | {'PPL p95':>8} | "
          f"{'shift p50':>9} | {'shift p95':>9} | {'yield':>7}")
    for row in rows:
        s = row["summary"]
        g = row["gate_default"]
        print(f"  {row['N']:>3} | {s['n_records']:>7} | "
              f"{s['ppl_p50']:>8.3f} | {s['ppl_p95']:>8.3f} | "
              f"{s['sae_p50']:>9.3f} | {s['sae_p95']:>9.3f} | "
              f"{g['pass_frac']*100:>6.1f}%")


if __name__ == "__main__":
    main()
