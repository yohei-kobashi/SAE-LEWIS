"""
Literature-standard edit metrics over existing records — the paper's
main-table generator (CPU only, no generation).

Per system (--cand name=records.jsonl:mode, probe format; pipeline
format via mode ""):
  exact / sim_target / copy  (from the records)
  SARI                       (set-based, n=1..4, single reference —
                              keep/add F1 + del precision, the standard
                              edit metric of the simplification /
                              instruction-editing literature)
  BLEU / chrF vs reference   (corpus-level, sacrebleu if installed —
                              LEWIS-style lexical overlap)
  self-BLEU vs source        (LEWIS's content-preservation analog)
Plus two pseudo-systems:
  oracle  — per-pair best exact across cands (the Pass@K analog:
            report as exact@oracle next to the router's exact@1)
  routed  — the confirmed count-rule router (--router-head/-fallback/-T:
            head if its own output has <= T hunks vs src, else fallback)
And the per-phenomenon breakdown (LinguaLens `feature` column):
  per feature x system: n / exact / SARI  -> per_feature.csv + top table.

Usage (miyabi):
    python scripts/score_edit_metrics.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --cand routedEF=runs/prod_gemma_v6/ksweep500/records.jsonl:k32 \
        --cand steer=runs/prod_gemma_v6/steer_baseline500/records.jsonl:steer0.5 \
        --cand clamp=runs/prod_gemma_v6/clamp_baseline500/records.jsonl:clamp10 \
        --cand prompt=runs/prod_gemma_v6/prompt_baseline500/records.jsonl:prompt8 \
        --cand pipeline=runs/prod_gemma_v6/eval_lingualens_final/records.jsonl: \
        --router-head routedEF --router-fallback steer --router-T 1 \
        --out runs/tables/main_metrics
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# SARI (set-based, n=1..4, single reference). Components: keep F1, add
# F1, del precision; an operation whose sys and ref sets are BOTH empty
# counts 1.0 (correctly doing nothing).
# ---------------------------------------------------------------------------
def _ngram_set(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _pr(hit, sys_n, ref_n):
    if sys_n == 0 and ref_n == 0:
        return 1.0, 1.0
    p = hit / sys_n if sys_n else 0.0
    r = hit / ref_n if ref_n else 0.0
    return p, r


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def sari_sentence(src: str, out: str, ref: str) -> float:
    s_w, o_w, r_w = src.lower().split(), out.lower().split(), ref.lower().split()
    scores = []
    for n in range(1, 5):
        S, O, R = _ngram_set(s_w, n), _ngram_set(o_w, n), _ngram_set(r_w, n)
        k_sys, k_ref = S & O, S & R
        kp, kr = _pr(len(k_sys & k_ref), len(k_sys), len(k_ref))
        a_sys, a_ref = O - S, R - S
        ap, ar = _pr(len(a_sys & a_ref), len(a_sys), len(a_ref))
        d_sys, d_ref = S - O, S - R
        dp, _ = _pr(len(d_sys & d_ref), len(d_sys), len(d_ref))
        scores.append((_f1(kp, kr) + _f1(ap, ar) + dp) / 3.0)
    return float(np.mean(scores)) * 100.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", action="append", required=True,
                   help="name=records.jsonl:mode (mode '' = pipeline "
                        "format)")
    p.add_argument("--condition", default="true")
    p.add_argument("--router-head", default="",
                   help="cand name for the count-rule router head")
    p.add_argument("--router-fallback", default="")
    p.add_argument("--router-T", type=int, default=1)
    p.add_argument("--llm2vec-dir", default="",
                   help="tokenizer for router hunk counts (required with "
                        "--router-head)")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--out", required=True,
                   help="output prefix: <out>.md, <out>_per_feature.csv, "
                        "<out>.json")
    p.add_argument("--emit-routed-records", default="",
                   help="materialize the routed system as a probe-format "
                        "records.jsonl (mode 'routed', + 'picked' field) "
                        "so FRR/SLOR/compare treat it like any system")
    return p.parse_args()


def main():
    args = parse_args()
    cands = []
    for spec in args.cand:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        with open(path) as f:
            recs = {r["idx"]: r for r in
                    (json.loads(l) for l in f if l.strip())}
        cands.append((name, recs, mode))
        print(f"[metrics] {name}: {len(recs)} records (mode "
              f"{mode or '(pipeline)'})")
    common = sorted(set.intersection(*[set(r) for _, r, _ in cands]))
    print(f"[metrics] common pairs: {len(common)}")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    feat_of = {k: (ds[k].get("feature") or "?") for k in common}

    tok = None
    if args.router_head:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.llm2vec_dir)

    def get_node(name_recs_mode, k):
        _, recs, mode = name_recs_mode
        o = recs[k]["outputs"].get(args.condition)
        if o is None:
            return None
        node = o if not mode else o.get(mode)
        return node if isinstance(node, dict) and "text" in node else None

    def st(rec):
        return (rec.get("src") or rec.get("source"),
                rec.get("tgt") or rec.get("target"))

    # per-pair rows: {sys: {"text","exact","sim","sari"}}
    pair_rows = {}
    for k in common:
        row = {}
        src, tgt = st(cands[0][1][k])
        for c in cands:
            node = get_node(c, k)
            if node is None:
                continue
            row[c[0]] = {
                "text": node["text"],
                "exact": float(node.get("exact",
                                        node.get("exact_match", 0.0))),
                "sim": float(node["sim_target"]),
                "sari": sari_sentence(src, node["text"], tgt),
            }
        # oracle pseudo-system (Pass@K analog over the cand set)
        if row:
            best = max(row.values(), key=lambda v: (v["exact"], v["sari"]))
            row["oracle"] = dict(best)
        pair_rows[k] = row

    # routed pseudo-system (the confirmed count-rule)
    picked_of = {}
    if args.router_head and args.router_fallback:
        head = dict((n, (r, m)) for n, r, m in cands)[args.router_head]
        for k in common:
            base = head[0][k]
            src, _ = st(base)
            text = base["outputs"][args.condition][head[1]]["text"]
            a = tok(src, add_special_tokens=False).input_ids
            b = tok(text, add_special_tokens=False).input_ids
            hunks = sum(1 for t, *_ in difflib.SequenceMatcher(
                None, a, b, autojunk=False).get_opcodes() if t != "equal")
            pick = args.router_head if hunks <= args.router_T \
                else args.router_fallback
            if pick in pair_rows[k]:
                pair_rows[k]["routed"] = dict(pair_rows[k][pick])
                picked_of[k] = pick

    if args.emit_routed_records and picked_of:
        rp = Path(args.emit_routed_records)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f:
            for k in common:
                if k not in picked_of:
                    continue
                base = cands[0][1][k]
                src, tgt = st(base)
                v = pair_rows[k]["routed"]
                f.write(json.dumps({
                    "idx": int(k), "src": src, "tgt": tgt,
                    "n_ops": base.get("n_ops", 1),
                    "picked": picked_of[k],
                    "outputs": {args.condition: {"routed": {
                        "text": v["text"], "exact": v["exact"],
                        "sim_target": v["sim"],
                    }}}}, ensure_ascii=False) + "\n")
        print(f"[metrics] routed records -> {rp} ({len(picked_of)})")

    sys_names = [n for n, _, _ in cands]
    if args.router_head:
        sys_names.append("routed")
    sys_names.append("oracle")

    # corpus BLEU/chrF (sacrebleu, optional) + self-BLEU vs source
    def corpus_scores(name):
        try:
            import sacrebleu
        except ImportError:
            return None
        outs, tgts, srcs = [], [], []
        for k in common:
            if name in pair_rows[k]:
                src, tgt = st(cands[0][1][k])
                outs.append(pair_rows[k][name]["text"])
                tgts.append(tgt)
                srcs.append(src)
        if not outs:
            return None
        return {
            "bleu": sacrebleu.corpus_bleu(outs, [tgts]).score,
            "chrf": sacrebleu.corpus_chrf(outs, [tgts]).score,
            "sbleu": sacrebleu.corpus_bleu(outs, [srcs]).score,
        }

    lines = ["# Main edit-metrics table (literature-standard)", "",
             f"pairs: {len(common)}; condition {args.condition}; SARI = "
             f"set-based n1-4 single-ref (keep/add F1 + del P); BLEU/chrF "
             f"vs reference, self-BLEU vs source (LEWIS-style); 'oracle' "
             f"= per-pair best exact across systems (Pass@K analog); "
             f"'routed' = count-rule T={args.router_T} "
             f"({args.router_head} if own hunks<=T else "
             f"{args.router_fallback})", "",
             "| system | exact | SARI | sim | BLEU | chrF | selfBLEU |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for name in sys_names:
        rows = [pair_rows[k][name] for k in common if name in pair_rows[k]]
        if not rows:
            continue
        cs = corpus_scores(name) or {}
        summary[name] = {
            "n": len(rows),
            "exact": float(np.mean([r["exact"] for r in rows])),
            "sari": float(np.mean([r["sari"] for r in rows])),
            "sim": float(np.mean([r["sim"] for r in rows])),
            **cs,
        }
        m = summary[name]
        lines.append(
            f"| {name} | {m['exact']:.4f} | {m['sari']:.2f} | "
            f"{m['sim']:.4f} | {m.get('bleu', float('nan')):.2f} | "
            f"{m.get('chrf', float('nan')):.2f} | "
            f"{m.get('sbleu', float('nan')):.2f} |")

    # ---- per-phenomenon breakdown ----------------------------------------
    byf = defaultdict(list)
    for k in common:
        byf[feat_of[k]].append(k)
    csv_lines = ["feature,n," + ",".join(
        f"{n}_exact,{n}_sari" for n in sys_names)]
    feats_sorted = sorted(byf, key=lambda f: -len(byf[f]))
    for f in feats_sorted:
        ks = byf[f]
        cells = [f'"{f}"', str(len(ks))]
        for name in sys_names:
            rows = [pair_rows[k][name] for k in ks if name in pair_rows[k]]
            if rows:
                cells.append(f"{np.mean([r['exact'] for r in rows]):.4f}")
                cells.append(f"{np.mean([r['sari'] for r in rows]):.2f}")
            else:
                cells.extend(["", ""])
        csv_lines.append(",".join(cells))

    lines += ["", "## Per-phenomenon exact (features with n >= 8; full "
                  "table in the CSV)", ""]
    show = [n for n in ("routed", args.router_head or sys_names[0],
                        "steer", "pipeline") if n in sys_names]
    lines.append("| feature | n | " + " | ".join(show) + " |")
    lines.append("|---" * (2 + len(show)) + "|")
    for f in feats_sorted:
        ks = byf[f]
        if len(ks) < 8:
            continue
        cells = []
        for name in show:
            rows = [pair_rows[k][name] for k in ks if name in pair_rows[k]]
            cells.append(f"{np.mean([r['exact'] for r in rows]):.3f}"
                         if rows else "—")
        lines.append(f"| {f} | {len(ks)} | " + " | ".join(cells) + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    print(report)
    Path(str(out) + ".md").write_text(report + "\n")
    Path(str(out) + "_per_feature.csv").write_text("\n".join(csv_lines) + "\n")
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1))
    print(f"[metrics] wrote {out}.md, {out}_per_feature.csv, {out}.json")


if __name__ == "__main__":
    main()
