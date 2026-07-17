#!/usr/bin/env python3
"""LinguaLens intervention-evaluation reproduction — JUDGE + METRICS (CPU).

The official repo contains NO judging code (confirmed 2026-07-14), so this
side follows the PAPER exactly (§3.3, §4.2.2, App. E.2):

  * judge      = GPT-4o, comparing each intervened output against the
    "unmodified SAE model" output (= the repo control, multiply x1),
    judging whether the target feature's prominence INCREASED / DECREASED /
    stayed the SAME. Presentation order randomized per comparison.
  * success    = enhancement: prominence increased; ablation: decreased.
  * P^T (per direction) = mean of the three per-vector success rates
    ("compute the average results across these three interventions").
  * P^B = the random-baseline arm's success rate.
  * E_abl = (P(Y=0|IT) - P(Y=0|IB)) / P(Y=0|IT)
    E_enh = (P(Y=1|IT) - P(Y=1|IB)) / (1 - P(Y=1|IB))
  * FIC   = harmonic mean of the penalized effects, E' = E if E>=0 else
    w*|E| with w = 0.5 (App. E.2).

Report anchors against the paper's Table 2 (Llama-3.1-8B + OpenSAE): the
comparison is of PATTERN and RANGE — the base model, SAE and layer differ.

Usage (prepost, needs OPENAI_API_KEY):
  python scripts/judge_ll_repro.py --run-dir runs/ll_repro
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_paper_metrics import Judge                           # noqa: E402

# glosses verbatim from LinguaLens App. F.2 (self-extracted PDF)
GLOSS = {
    "past_tense": "verb form that locates an event before speech time",
    "linking_verb": "verb that links subject to a complement",
    "politeness": "linguistic strategies that mitigate imposition or "
                  "face threat",
    "metaphor": "describing one thing as if it were another, non-literal "
                "figurative language",
}

# paper Table 2 (self-extracted): feature -> (enh_exp, enh_ctr, abl_exp,
# abl_ctr, FIC) in percent. Simile row: their App. E.2 calls it "the
# metaphor feature"; causality has no English-99 counterpart.
TABLE2 = {
    "past_tense":   (12.0,  4.0, 48.0, 44.0,  8.3),
    "linking_verb": (52.0, 24.0, 48.0, 40.0, 22.9),
    "metaphor":     (72.0, 52.0, 48.0, 52.0,  6.9),   # their "Simile" row
    "politeness":   (60.0, 32.0, 44.0, 20.0, 46.9),
}

PROMPT = """You are evaluating the prominence of a linguistic feature in \
generated text.

Feature: {name} — {gloss}.

TEXT A:
{a}

TEXT B:
{b}

Compared with TEXT A, does TEXT B exhibit this feature MORE prominently, \
LESS prominently, or about the SAME? Answer with exactly one word: MORE, \
LESS, or SAME."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--judge-model", default="gpt-4o")   # paper's judge
    p.add_argument("--order-seed", type=int, default=1234)
    p.add_argument("--penalty-w", type=float, default=0.5)  # App. E.2
    return p.parse_args()


def parse_rel(text: str):
    t = text.strip().upper()
    for w in ("MORE", "LESS", "SAME"):
        if w in t:
            return w
    return None


def main():
    args = parse_args()
    run = Path(args.run_dir)
    recs = [json.loads(l) for l in open(run / "records.jsonl")
            if l.strip()]
    ctl = defaultdict(dict)          # feature -> i -> text
    ivd = []                         # intervened records
    for r in recs:
        if r["cond"] == "control":
            ctl[r["feature"]][int(r["i"])] = r["output"]
        else:
            ivd.append(r)
    print(f"[llrepro-judge] {len(ivd)} intervened outputs, "
          f"{sum(len(v) for v in ctl.values())} controls")

    cache_path = run / f"judge_cache_{args.judge_model}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                c = json.loads(line)
                cache[c["key"]] = c["rel"]
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[llrepro-judge] RESUME: {len(cache)} judgments cached")
    cfh = open(cache_path, "a")
    judge = Judge(args.judge_model, max_tokens=6)

    n_new = 0
    for r in ivd:
        key = f"{r['feature']}|{r['cond']}|{r['direction']}|{r['vec_tag']}|{r['i']}"
        if key in cache:
            continue
        ref = ctl[r["feature"]].get(int(r["i"]))
        if ref is None:
            continue
        flip = random.Random(f"{args.order_seed}|{key}").random() < 0.5
        a, b = (r["output"], ref) if flip else (ref, r["output"])
        ans = parse_rel(judge(PROMPT.format(
            name=r["feature"], gloss=GLOSS[r["feature"]], a=a, b=b)))
        if ans is None:
            rel = "SAME"             # unparseable -> conservative
        elif ans == "SAME":
            rel = "SAME"
        else:
            # rel = intervened relative to control
            rel = ans if not flip else ("LESS" if ans == "MORE" else "MORE")
        cache[key] = rel
        cfh.write(json.dumps({"key": key, "rel": rel}) + "\n")
        n_new += 1
        if n_new % 50 == 0:
            cfh.flush()
            print(f"[llrepro-judge] +{n_new}")
    cfh.close()

    # ---- metrics ----------------------------------------------------------
    def rate(feature, cond, direction, vec_tag, want):
        keys = [k for k in cache
                if k.startswith(f"{feature}|{cond}|{direction}|{vec_tag}|")]
        if not keys:
            return float("nan"), 0
        hit = sum(cache[k] == want for k in keys)
        return hit / len(keys), len(keys)

    def eff(pt, pb, kind):
        if kind == "abl":
            return (pt - pb) / pt if pt > 0 else float("nan")
        return (pt - pb) / (1.0 - pb) if pb < 1.0 else float("nan")

    def penal(e):
        return e if e >= 0 else args.penalty_w * abs(e)

    meta = json.loads((run / "meta.json").read_text())
    lines = ["# LinguaLens intervention-evaluation reproduction (repo-faithful)",
             "",
             f"stack: {meta['stack']} | judge: {args.judge_model} "
             f"(paper's) | reference = repo control (multiply x1 recon "
             f"passthrough) | random arm: {meta['random_nvec']} fresh "
             f"vector(s)/experiment",
             "",
             "| feature | enh exp | enh ctr | abl exp | abl ctr | E_enh | "
             "E_abl | FIC | Table2 (enh e/c, abl e/c, FIC) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in meta["features"]:
        vecs = [str(v) for v in meta["features"][f]]
        per_enh = [rate(f, "targeted", "enh", v, "MORE")[0] for v in vecs]
        per_abl = [rate(f, "targeted", "abl", v, "LESS")[0] for v in vecs]
        pt_enh = sum(per_enh) / len(per_enh)
        pt_abl = sum(per_abl) / len(per_abl)
        pb_enh, _ = rate(f, "random", "enh", "rnd", "MORE")
        pb_abl, _ = rate(f, "random", "abl", "rnd", "LESS")
        e_enh, e_abl = eff(pt_enh, pb_enh, "enh"), eff(pt_abl, pb_abl, "abl")
        fic = (2 * penal(e_abl) * penal(e_enh)
               / (penal(e_abl) + penal(e_enh))
               if penal(e_abl) + penal(e_enh) > 0 else float("nan"))
        t2 = TABLE2.get(f)
        t2s = (f"{t2[0]:.0f}/{t2[1]:.0f}, {t2[2]:.0f}/{t2[3]:.0f}, {t2[4]}"
               if t2 else "—")
        lines.append(
            f"| {f} | {pt_enh*100:.1f} | {pb_enh*100:.1f} | "
            f"{pt_abl*100:.1f} | {pb_abl*100:.1f} | {e_enh:+.3f} | "
            f"{e_abl:+.3f} | {fic*100:.1f} | {t2s} |")
        pv = ", ".join(f"{v}:{e*100:.0f}/{a*100:.0f}"
                       for v, e, a in zip(vecs, per_enh, per_abl))
        lines.append(f"|   per-vector enh/abl | {pv} |  |  |  |  |  |  |  |")
    lines += [
        "",
        "Anchor reading: the paper's pattern is (1) enhancement exp >> ctr "
        "for every feature, (2) ablation nearly flat (single-feature "
        "ablation compensated by parallel paths), (3) FIC positive but "
        "modest. Digit-for-digit equality is not expected — different base "
        "model (Llama-3.1-8B+OpenSAE 32L vs gemma-2-2b-it+Gemma Scope "
        "L12).",
        "",
        "Known discrepancies inherited from the paper itself: §3.3 says 25 "
        "random baseline vectors, §4.2.2 says 50 indices (we run 1 fresh "
        "vector per experiment, shape-matched to the targeted arm); Table 2 "
        "labels the Semantics row 'Simile' while App. E.2 calls it 'the "
        "metaphor feature' (we run `metaphor`); the paper says 6 features "
        "but Table 2 lists 5.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[llrepro-judge] wrote {run/'report.md'}")


if __name__ == "__main__":
    main()
