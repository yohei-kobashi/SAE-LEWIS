"""
Judge-agreement readout: compare two FRR judgment files for the SAME
system under DIFFERENT judges (idx-joined). Reports pairwise agreement
on the realized verdict and on the raw direction (sys field), plus
each judge's FRR — the robustness line for the paper.

Usage:
    python scripts/judge_agreement.py \
        --a runs/frr_final/hf_google_gemma-2-9b-it/routed.jsonl \
        --b runs/frr_final/openai_gpt-4o/routed.jsonl
"""

from __future__ import annotations

import argparse
import json


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["idx"])] = r
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    args = p.parse_args()
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    print(f"[agree] common judged pairs: {len(common)} "
          f"(a {len(A)}, b {len(B)})")

    both = [(A[k], B[k]) for k in common]
    scored = [(a, b) for a, b in both
              if a["realized"] is not None and b["realized"] is not None]
    if scored:
        agree_r = sum(1 for a, b in scored
                      if a["realized"] == b["realized"]) / len(scored)
        frr_a = sum(1 for a, _ in scored if a["realized"]) / len(scored)
        frr_b = sum(1 for _, b in scored if b["realized"]) / len(scored)
        print(f"[agree] realized-verdict agreement: {agree_r:.4f} "
              f"(n={len(scored)}; FRR a={frr_a:.4f} b={frr_b:.4f})")
    agree_s = sum(1 for a, b in both if a["sys"] == b["sys"]) / len(both)
    agree_g = sum(1 for a, b in both if a["gold"] == b["gold"]) / len(both)
    print(f"[agree] raw direction agreement: sys {agree_s:.4f}, "
          f"gold {agree_g:.4f}")


if __name__ == "__main__":
    main()
