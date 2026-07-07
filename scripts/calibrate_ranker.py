"""
Offline RankerWeights calibration over cached candidate components.

Input: records.jsonl produced by `eval_lingualens.py --dump-details` on a
DEV slice (use a --seed different from the evaluation slice so the tuned
weights are not fitted to the reported benchmark sample).

Every candidate's raw ranker components (sae_align / fluency / content /
ins_slots) and text are stored per pair, so re-scoring under different
weights is pure arithmetic — the full grid runs in seconds with no GPU.

For each weight combination the script simulates the ranker's pick per
pair (condition `true`), then reports mean sim_target (word-level
SequenceMatcher vs sentence2), copy rate, and edit-location IoU, sorted
by sim_target. The chosen weights go back in via
`eval_lingualens.py --ranker-weights "a,b,c,e"` (and the same flag can be
mirrored in evaluate_intervention.py / RankerWeights defaults once
confirmed).

Usage:
    python eval_lingualens.py ... --seed 123 --sample-size 200 \
        --dump-details --output-dir runs/.../lingua_dev
    python scripts/calibrate_ranker.py \
        --records runs/.../lingua_dev/records.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="records.jsonl from eval_lingualens --dump-details.")
    p.add_argument("--condition", default="true")
    p.add_argument("--grid-sae", default="0.5,1.0,2.0,4.0,8.0")
    p.add_argument("--grid-fluency", default="0.0,0.1,0.3,1.0")
    p.add_argument("--grid-content", default="0.0,0.1,0.2,0.5")
    p.add_argument("--grid-lenpen", default="0.0,0.05,0.2")
    p.add_argument("--fluency-gate", type=float, default=0.0,
                   help="Mirror of the inference-time gate (nats/token; "
                        "0 = off): candidates with components['fluency'] < "
                        "tanh(-gate) are excluded before scoring, identity "
                        "excepted. Only valid on dumps produced AFTER the "
                        "fluency component became a delta (ranker.py); "
                        "older dumps store saturated absolute values.")
    p.add_argument("--top", type=int, default=15)
    return p.parse_args()


def _words(s: str) -> List[str]:
    return s.strip().split()


def _norm(s: str) -> str:
    return " ".join(s.strip().split()).casefold()


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _words(a), _words(b),
                                   autojunk=False).ratio()


def edited_positions(src_words, other_words):
    pos = set()
    sm = difflib.SequenceMatcher(None, src_words, other_words, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "equal":
            pos.update(range(i1, max(i2, i1 + 1)))
    return pos


def main():
    args = parse_args()
    pairs = []
    with open(args.records, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out = r.get("outputs", {}).get(args.condition)
            if not out or "candidates" not in out:
                continue
            pairs.append({
                "source": r["source"],
                "target": r["target"],
                "cands": out["candidates"],
            })
    if not pairs:
        raise SystemExit(
            f"no candidates for condition={args.condition!r} in "
            f"{args.records} — was eval_lingualens run with --dump-details?")
    print(f"[calibrate] {len(pairs)} pairs, "
          f"{sum(len(p['cands']) for p in pairs)} cached candidates")

    grid = list(itertools.product(
        (float(x) for x in args.grid_sae.split(",")),
        (float(x) for x in args.grid_fluency.split(",")),
        (float(x) for x in args.grid_content.split(",")),
        (float(x) for x in args.grid_lenpen.split(",")),
    ))
    print(f"[calibrate] grid size: {len(grid)}")

    # Precompute per-candidate metrics once.
    gate_floor = math.tanh(-args.fluency_gate) if args.fluency_gate > 0 else None
    n_gated = 0
    for p_ in pairs:
        sw = _words(p_["source"])
        gold = edited_positions(sw, _words(p_["target"]))
        for c in p_["cands"]:
            c["_sim_t"] = sim(c["text"], p_["target"])
            c["_is_copy"] = float(_norm(c["text"]) == _norm(p_["source"]))
            pred = edited_positions(sw, _words(c["text"]))
            union = gold | pred
            c["_iou"] = (len(gold & pred) / len(union)) if union else 1.0
            c["_gated"] = (gate_floor is not None
                           and not c.get("is_identity", False)
                           and c["components"]["fluency"] < gate_floor)
            n_gated += int(c["_gated"])
    if gate_floor is not None:
        print(f"[calibrate] fluency gate {args.fluency_gate}: "
              f"{n_gated} candidates excluded")

    results = []
    for a, b, cw, e in grid:
        st, cp, io = 0.0, 0.0, 0.0
        for p_ in pairs:
            best, best_s = None, -1e30
            for c in p_["cands"]:
                if c["_gated"]:
                    continue
                comp = c["components"]
                sc = (a * comp["sae_align"] + b * comp["fluency"]
                      + cw * comp["content"] - e * comp["ins_slots"])
                if sc > best_s:
                    best_s, best = sc, c
            st += best["_sim_t"]
            cp += best["_is_copy"]
            io += best["_iou"]
        n = len(pairs)
        results.append(((a, b, cw, e), st / n, cp / n, io / n))

    results.sort(key=lambda r: -r[1])
    copy_base = sum(sim(p_["source"], p_["target"]) for p_ in pairs) / len(pairs)
    print(f"\ninput-copy baseline sim_target = {copy_base:.4f}\n")
    print(f"{'sae':>5} {'flu':>5} {'cont':>5} {'lpen':>5} | "
          f"{'sim_target':>10} {'copy_rate':>9} {'iou':>7}")
    for (a, b, cw, e), st, cp, io in results[:args.top]:
        marker = "  ← beats copy" if st > copy_base else ""
        print(f"{a:>5.2f} {b:>5.2f} {cw:>5.2f} {e:>5.2f} | "
              f"{st:>10.4f} {cp:>9.3f} {io:>7.4f}{marker}")
    best_w = results[0][0]
    print(f"\nbest: --ranker-weights \"{best_w[0]},{best_w[1]},"
          f"{best_w[2]},{best_w[3]}\"")


if __name__ == "__main__":
    main()
