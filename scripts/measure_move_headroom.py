"""
MOVE headroom pre-measurement — the no-GPU gate before MOVE-in-EF.

Reinterprets DEL+INS op pairs with identical token content as MOVE
(the V7 A2 gold-derivation rule, applied to the same alignment the EF
probe uses), then joins the flags with an existing probe records.jsonl
to measure how much exact the current champion loses on exactly those
pairs. Runs in seconds on CPU.

Decision rule (pre-registered): MOVE-in-EF is worth building only if
(a) strict-MOVE pairs are a non-trivial share of the 500-pair sample,
AND (b) the champion's exact on them is well below its non-MOVE exact —
i.e. the headroom line at the bottom is a meaningful fraction of the
remaining error budget.

Usage (on miyabi):
    python scripts/measure_move_headroom.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --records runs/prod_gemma_v6/editflow_s3/probe500/records.jsonl

Self-test (no deps beyond the repo):
    python scripts/measure_move_headroom.py --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editflow_ops import align_pair, slot_ops  # noqa: E402


# ---------------------------------------------------------------------------
# MOVE detection on the slot alignment
# ---------------------------------------------------------------------------
def del_ins_runs(slots) -> Tuple[List[Tuple[int, Tuple[int, ...]]],
                                 List[Tuple[int, Tuple[int, ...]]]]:
    """Maximal contiguous runs of pure-DEL / pure-INS slots.

    Returns two lists of (start_slot, content_tuple). SUB/KEEP slots
    break runs, so a run is exactly one side of a difflib delete/insert
    region (or the remainder of an unequal replace).
    """
    d_runs, i_runs = [], []
    kind_prev: Optional[str] = None
    start = 0
    buf: List[int] = []

    def flush():
        if not buf:
            return
        (d_runs if kind_prev == "del" else i_runs).append(
            (start, tuple(buf)))

    for k, (a0, a1) in enumerate(slots):
        if a0 is not None and a1 is None:
            kind, tok = "del", int(a0)
        elif a0 is None and a1 is not None:
            kind, tok = "ins", int(a1)
        else:
            kind, tok = None, None
        if kind != kind_prev:
            flush()
            buf, start, kind_prev = [], k, kind
        if kind is not None:
            buf.append(tok)
    flush()
    return d_runs, i_runs


def match_moves(d_runs, i_runs) -> List[Dict]:
    """Greedy 1:1 pairing of DEL runs with content-identical INS runs,
    longest content first (a long span match is stronger evidence of
    movement than a stray single-token coincidence, so it gets first
    pick of the partners)."""
    moves: List[Dict] = []
    used = set()
    order = sorted(range(len(d_runs)), key=lambda i: -len(d_runs[i][1]))
    for di in order:
        d_start, d_content = d_runs[di]
        for ii, (i_start, i_content) in enumerate(i_runs):
            if ii in used or i_content != d_content:
                continue
            used.add(ii)
            moves.append({"del_slot": d_start, "ins_slot": i_start,
                          "toks": list(d_content)})
            break
    return moves


def loose_overlap(d_runs, i_runs) -> int:
    """Token-multiset overlap between everything deleted and everything
    inserted — an upper bound on movement-ish mass (strict runs miss
    moves whose span is broken up or partially rewritten)."""
    dc = Counter(t for _, c in d_runs for t in c)
    ic = Counter(t for _, c in i_runs for t in c)
    return sum(min(dc[t], ic[t]) for t in dc)


# ---------------------------------------------------------------------------
def bname(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 8:
        return "4-8"
    return "9+"


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", default="runs/mcgill_gemma_repro_3k/final")
    p.add_argument("--records", required=True,
                   help="editflow_probe records.jsonl (e.g. the S3 probe500)")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--condition", default="true")
    p.add_argument("--examples", type=int, default=10,
                   help="how many strict-MOVE example pairs to print")
    p.add_argument("--out", default="",
                   help="optional output .md path (default: "
                        "move_headroom.md next to --records)")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:min(args.sample_size, len(order))]

    with open(args.records) as f:
        recs = {r["idx"]: r for r in
                (json.loads(l) for l in f if l.strip())}
    modes = sorted({m for r in recs.values()
                    for m, v in r["outputs"][args.condition].items()
                    if isinstance(v, dict)})

    pairs = []
    for k in chosen:
        ex = ds[int(k)]
        src_ids = tokenizer(ex["sentence1"],
                            add_special_tokens=True).input_ids
        tgt_ids = tokenizer(ex["sentence2"],
                            add_special_tokens=True).input_ids
        slots = align_pair(src_ids, tgt_ids)
        ops = slot_ops(slots)
        if not ops:
            continue                      # probe skips these too
        d_runs, i_runs = del_ins_runs(slots)
        moves = match_moves(d_runs, i_runs)
        covered = 2 * sum(len(m["toks"]) for m in moves)
        pairs.append({
            "idx": int(k),
            "n_ops": len(ops),
            "moves": moves,
            "move_toks": sum(len(m["toks"]) for m in moves),
            "residual_ops": len(ops) - covered,
            "loose": loose_overlap(d_runs, i_runs),
            "src": ex["sentence1"], "tgt": ex["sentence2"],
        })

    n_all = len(pairs)
    strict = [q for q in pairs if q["moves"]]
    pure = [q for q in strict if q["residual_ops"] == 0]
    loose_only = [q for q in pairs if not q["moves"] and q["loose"] > 0]

    L: List[str] = []
    L.append(f"# MOVE headroom — records: {args.records}")
    L.append("")
    L.append(f"pairs with ops: {n_all} (sampled {len(chosen)}, "
             f"records matched below)")
    L.append(f"- strict MOVE pairs (>=1 content-identical DEL/INS run "
             f"pair): **{len(strict)}** ({len(strict)/n_all:.1%})")
    L.append(f"  - pure reorders (every op covered by a MOVE): "
             f"{len(pure)}")
    L.append(f"- loose-only pairs (token-multiset overlap but no strict "
             f"run match): {len(loose_only)} "
             f"(upper bound on additional movement-ish mass)")
    L.append("")
    L.append("strict MOVE pairs by n_ops bucket:")
    bc = Counter(bname(q["n_ops"]) for q in strict)
    ba = Counter(bname(q["n_ops"]) for q in pairs)
    L.append("| bucket | move pairs | all pairs | share |")
    L.append("|---|---|---|---|")
    for b in ("1", "2-3", "4-8", "9+"):
        if ba[b]:
            L.append(f"| {b} | {bc[b]} | {ba[b]} | {bc[b]/ba[b]:.1%} |")
    L.append("")

    joined = [q for q in pairs if q["idx"] in recs]
    j_move = [q for q in joined if q["moves"]]
    j_rest = [q for q in joined if not q["moves"]]
    L.append(f"## Joined with records ({len(joined)} pairs: "
             f"{len(j_move)} MOVE / {len(j_rest)} non-MOVE)")
    L.append("")
    L.append("| mode | MOVE exact | MOVE sim | non-MOVE exact | "
             "non-MOVE sim | MOVE missed |")
    L.append("|---|---|---|---|---|---|")

    def metr(subset, mode, key):
        vals = []
        for q in subset:
            o = recs[q["idx"]]["outputs"][args.condition].get(mode)
            if isinstance(o, dict):
                vals.append(o[key])
        return vals

    headroom = {}
    for m in modes:
        me, ms = metr(j_move, m, "exact"), metr(j_move, m, "sim_target")
        re_, rs = metr(j_rest, m, "exact"), metr(j_rest, m, "sim_target")
        missed = sum(1 for v in me if v == 0)
        headroom[m] = missed
        L.append(f"| {m} | {mean(me):.4f} | {mean(ms):.4f} | "
                 f"{mean(re_):.4f} | {mean(rs):.4f} | "
                 f"{missed}/{len(me)} |")
    L.append("")
    L.append("headroom upper bound (every missed MOVE pair -> exact):")
    for m in modes:
        cur = mean(metr(joined, m, "exact"))
        gain = headroom[m] / len(joined)
        L.append(f"- {m}: {cur:.4f} -> {cur + gain:.4f} (+{gain:.4f})")
    L.append("")

    if strict and args.examples:
        L.append(f"## Examples (first {args.examples} strict MOVE pairs)")
        for q in strict[:args.examples]:
            moved = "; ".join(
                repr(tokenizer.decode(m["toks"])) for m in q["moves"])
            got = ""
            if q["idx"] in recs:
                marks = []
                for m in modes:
                    o = recs[q["idx"]]["outputs"][args.condition].get(m)
                    if isinstance(o, dict):
                        marks.append(f"{m}:{'O' if o['exact'] else 'X'}")
                got = "  [" + " ".join(marks) + "]"
            L.append(f"- idx {q['idx']} (n_ops {q['n_ops']}, residual "
                     f"{q['residual_ops']}): moved {moved}{got}")
            L.append(f"  - src: {q['src']}")
            L.append(f"  - tgt: {q['tgt']}")

    report = "\n".join(L)
    print(report)
    out = Path(args.out) if args.out else (
        Path(args.records).parent / "move_headroom.md")
    out.write_text(report + "\n")
    print(f"\n[move-headroom] wrote {out}")


# ---------------------------------------------------------------------------
def _selftest():
    # 1. particle shift: [1,2,3,4] -> [1,3,4,2] — "2" moves right
    slots = align_pair([1, 2, 3, 4], [1, 3, 4, 2])
    d, i = del_ins_runs(slots)
    mv = match_moves(d, i)
    assert len(mv) == 1 and mv[0]["toks"] == [2], mv
    assert loose_overlap(d, i) == 1

    # 2. substitution only: no runs, no moves
    slots = align_pair([1, 2, 3], [1, 9, 3])
    d, i = del_ins_runs(slots)
    assert not d and not i and not match_moves(d, i)

    # 3. del/ins with different content: runs but no strict move;
    #    loose overlap 0
    slots = align_pair([1, 2, 3], [1, 3, 9])
    d, i = del_ins_runs(slots)
    assert not match_moves(d, i)
    assert loose_overlap(d, i) == 0

    # 4. multi-token span move: [1,2,3,4,5] -> [4,5,1,2,3]
    slots = align_pair([1, 2, 3, 4, 5], [4, 5, 1, 2, 3])
    d, i = del_ins_runs(slots)
    mv = match_moves(d, i)
    assert len(mv) == 1 and (mv[0]["toks"] in ([4, 5], [1, 2, 3])), mv

    # 5. partial rewrite breaks the strict match but loose catches it:
    #    del (5,6) somewhere, ins (6,7) elsewhere
    slots = align_pair([1, 5, 6, 2, 3], [1, 2, 3, 6, 7])
    d, i = del_ins_runs(slots)
    assert not match_moves(d, i)
    assert loose_overlap(d, i) == 1

    # 6. longest-first pairing: two del runs (7,) and (7,8) and one ins
    #    run (7,8) — the 2-token run must claim the partner
    d = [(0, (7,)), (3, (7, 8))]
    i = [(9, (7, 8))]
    mv = match_moves(d, i)
    assert len(mv) == 1 and mv[0]["toks"] == [7, 8] and \
        mv[0]["del_slot"] == 3, mv

    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
