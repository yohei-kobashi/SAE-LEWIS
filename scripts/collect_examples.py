"""
Qualitative example collection: per LinguaLens feature, success / near-miss
/ failure cases with every system's output side by side on the SAME pair.

The three categories are the paper's residual-frontier decomposition
(outline section 6c) turned into examples, so the qualitative table lines
up one-to-one with the quantitative claim:

  success : focus system exact-matched the target.
  near    : exact miss, but the FRR judge says the feature WAS realized in
            the commanded direction — "directionally realizable, not
            exactly editable" (the metaphor / personification cluster).
  fail    : exact miss AND not realized — the genuinely unreachable end.

Without --frr for the focus system only success/fail (by exact) are split.

Systems are shown on the same idx so the comparison is controlled; outputs
are rendered as a word diff against the SOURCE (**added/substituted**,
~~removed~~) because what matters is which edit each system made.

Usage (miyabi; B1/B2 cover ~499 pairs, so --require-all intersects to that):
    python scripts/collect_examples.py \
        --sys ours=runs/prod_gemma_v6/routed_system/records.jsonl,routed \
        --sys ef32=runs/prod_gemma_v6/ksweep500/records.jsonl,k32 \
        --sys lingualens=runs/prod_gemma_v6/clamp_baseline500/records.jsonl,clamp10 \
        --sys steer=runs/prod_gemma_v6/steer_baseline500/records.jsonl,steer0.5 \
        --frr ours=runs/frr_final/openai_gpt-4o/routed.jsonl \
        --frr ef32=runs/frr_final/openai_gpt-4o/ef32.jsonl \
        --frr steer=runs/frr_final/openai_gpt-4o/steer.jsonl \
        --focus ours --per-feature 1 --out runs/tables/examples_gpt-4o
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path

CATS = ("success", "near", "fail")
CAT_DESC = {
    "success": "focus system exact-matched the target",
    "near": "exact miss, but the judge saw the feature realized in the "
            "commanded direction — directionally realizable, not exactly "
            "editable",
    "fail": "exact miss and not realized — the unreachable end",
}


def norm(s: str) -> str:
    return " ".join(s.split())


def mark(src: str, out: str) -> str:
    """Word diff of `out` against `src`: **added/substituted**, ~~removed~~."""
    s, o = src.split(), out.split()
    parts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, s, o, autojunk=False).get_opcodes():
        if tag == "equal":
            parts.extend(o[j1:j2])
            continue
        if tag in ("delete", "replace") and i1 != i2:
            parts.append("~~" + " ".join(s[i1:i2]) + "~~")
        if tag in ("insert", "replace") and j1 != j2:
            parts.append("**" + " ".join(o[j1:j2]) + "**")
    return " ".join(parts)


def load_sys(path: str, mode: str, condition: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            o = (r.get("outputs") or {}).get(condition)
            if o is None:
                continue
            node = o if not mode else o.get(mode)
            if not isinstance(node, dict) or "text" not in node:
                continue
            out[int(r["idx"])] = {
                "src": r.get("src") or r.get("source"),
                "tgt": r.get("tgt") or r.get("target"),
                "text": node["text"],
                "exact": float(node.get("exact", 0.0)),
                "picked": r.get("picked"),
            }
    return out


def load_frr(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["idx"])] = r.get("realized")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sys", action="append", required=True,
                   help="label=records.jsonl,mode[,condition] "
                        "(condition defaults to 'true')")
    p.add_argument("--frr", action="append", default=[],
                   help="label=judgments.jsonl — attaches the judge's "
                        "realized verdict; the focus system's verdict is "
                        "what splits near from fail")
    p.add_argument("--focus", default="",
                   help="system whose success/failure defines the split "
                        "(default: first --sys)")
    p.add_argument("--per-feature", type=int, default=1)
    p.add_argument("--features", default="",
                   help="comma-separated allowlist (default: all)")
    p.add_argument("--require-all", action="store_true", default=True,
                   help="only use pairs where EVERY system has an output, "
                        "so each example is a controlled comparison")
    p.add_argument("--any-system", dest="require_all", action="store_false")
    p.add_argument("--max-words", type=int, default=0,
                   help="skip sources longer than this (0 = no limit); "
                        "short pairs read better in a paper")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    systems, modes = {}, {}
    for spec in args.sys:
        label, rest = spec.split("=", 1)
        parts = rest.split(",")
        path, mode = parts[0], parts[1] if len(parts) > 1 else ""
        cond = parts[2] if len(parts) > 2 else "true"
        systems[label] = load_sys(path, mode, cond)
        modes[label] = mode or cond
        print(f"[ex] {label} ({mode}): {len(systems[label])} records")

    frr = {}
    for spec in args.frr:
        label, path = spec.split("=", 1)
        frr[label] = load_frr(path)
        print(f"[ex] frr {label}: {len(frr[label])} judgments")

    labels = list(systems)
    focus = args.focus or labels[0]
    if focus not in systems:
        raise SystemExit(f"--focus {focus!r} is not among {labels}")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)

    keys = set(systems[focus])
    for lab in labels:
        keys = (keys & set(systems[lab])) if args.require_all \
            else (keys | set(systems[lab]))
    keys = sorted(keys)
    print(f"[ex] {len(keys)} pairs "
          f"({'all systems present' if args.require_all else 'union'})")

    allow = {f.strip() for f in args.features.split(",") if f.strip()}
    buckets = defaultdict(lambda: defaultdict(list))
    for k in keys:
        f = systems[focus].get(k)
        if not f:
            continue
        if args.max_words and len(f["src"].split()) > args.max_words:
            continue
        ex = ds[k]
        feat = ex.get("feature") or ex.get("categories") or "?"
        if allow and feat not in allow:
            continue
        realized = frr.get(focus, {}).get(k)
        if f["exact"] >= 1.0:
            cat = "success"
        elif realized is True:
            cat = "near"
        elif realized is False:
            cat = "fail"
        else:
            continue          # gold-indecisive, or no FRR for the focus
        # rank: prefer examples that CONTRAST with LinguaLens, then short
        score = 0.0
        ll = next((l for l in labels if l != focus and "lingua" in l), None)
        if ll and ll in systems and k in systems[ll]:
            if cat == "success" and systems[ll][k]["exact"] < 1.0:
                score += 4.0          # we succeed where LinguaLens does not
            if cat == "fail" and systems[ll][k]["exact"] < 1.0:
                score += 4.0          # a frontier nobody reaches
        if cat == "near":
            score += 2.0              # every near case is illustrative
        score -= len(f["src"].split()) / 100.0
        buckets[feat][cat].append((-score, k))

    lines = [f"# LinguaLens transformation examples — focus: {focus}", "",
             f"Systems: " + ", ".join(f"`{l}` ({modes[l]})" for l in labels)
             + f". Pairs where every system has an output: {len(keys)}.",
             "", "Categories (= the residual-frontier decomposition):", ""]
    lines += [f"- **{c}** — {CAT_DESC[c]}" for c in CATS]
    lines += ["", "Outputs are word-diffed against the SOURCE: "
                  "**added/substituted**, ~~removed~~. `exact` is against "
                  "the target; `FRR` is the judge's realized verdict "
                  "(— = not judged for that system).", ""]

    dump, counts = [], defaultdict(int)
    for feat in sorted(buckets):
        picked_any = False
        for cat in CATS:
            cands = sorted(buckets[feat].get(cat, []))[:args.per_feature]
            for _, k in cands:
                counts[cat] += 1
                if not picked_any:
                    lines.append(f"## {feat}")
                    picked_any = True
                f = systems[focus][k]
                lines += ["", f"### {cat} — idx {k}", "",
                          f"- **source**: {f['src']}",
                          f"- **target**: {mark(f['src'], f['tgt'])}", ""]
                row = {"feature": feat, "cat": cat, "idx": k,
                       "src": f["src"], "tgt": f["tgt"], "systems": {}}
                for lab in labels:
                    s = systems[lab].get(k)
                    if not s:
                        lines.append(f"- `{lab}`: —")
                        continue
                    r = frr.get(lab, {}).get(k)
                    rs = "—" if r is None else ("✓" if r else "✗")
                    es = "✓" if s["exact"] >= 1.0 else "✗"
                    star = " ←focus" if lab == focus else ""
                    pick = (f" via {s['picked']}"
                            if lab == focus and s.get("picked") else "")
                    body = ("*(copy — no edit)*"
                            if norm(s["text"]) == norm(f["src"])
                            else mark(f["src"], s["text"]))
                    lines.append(
                        f"- `{lab}`{pick} [exact {es} · FRR {rs}]{star}: "
                        f"{body}")
                    row["systems"][lab] = {
                        "text": s["text"], "exact": s["exact"],
                        "realized": r, "picked": s.get("picked")}
                lines.append("")
                dump.append(row)

    head = ["", "**Coverage**: " + ", ".join(
        f"{c} {counts[c]}" for c in CATS) +
        f" over {len(buckets)} features.", ""]
    lines[2:2] = head

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    with open(str(out) + ".jsonl", "w") as f:
        for r in dump:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ex] " + ", ".join(f"{c} {counts[c]}" for c in CATS)
          + f" over {len(buckets)} features")
    print(f"[ex] wrote {out}.md, {out}.jsonl")


if __name__ == "__main__":
    main()
