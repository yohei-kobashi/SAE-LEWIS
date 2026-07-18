#!/usr/bin/env python3
"""AxBench steering reproduction — JUDGE + METRICS (CPU, prepost).

Judge prompts are the OFFICIAL ones, verbatim from
axbench/evaluators/prompt_templates.py (concept / instruction / fluency,
each 0-2, "Rating: [[score]]"). Judge model = gpt-4o-mini (their yaml).
Overall = harmonic mean of the three subscores (0 if any is 0).

Protocol (evaluate.py, winrate_split_ratio=0.5): per (concept, arm), the
best factor is chosen by mean overall score on the SELECTION half of the
instructions (input_id < n/2), then the reported score is the mean overall
on the HOLDOUT half (input_id >= n/2) at that factor. Judging is lazy:
selection half over all factors first, then holdout at the best factor.

Anchors (paper Table 2, gemma-2-2b): SAE L10 0.177 / L20 0.151;
SAE-A L10 0.166 / L20 0.132.

Usage: python scripts/judge_axbench_repro.py --run-dir runs/axbench_repro \
           --config prod_2b_l20_v1
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_paper_metrics import Judge                           # noqa: E402

# ---- verbatim from axbench/evaluators/prompt_templates.py ------------------
T_CONCEPT = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

T_INSTRUCT = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

T_FLUENCY = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""
# ---------------------------------------------------------------------------

ANCHORS = {"prod_2b_l10_v1": {"sae": 0.177, "sae_a": 0.166},
           "prod_2b_l20_v1": {"sae": 0.151, "sae_a": 0.132}}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default="prod_2b_l20_v1")
    p.add_argument("--judge-model", default="gpt-4o-mini")   # their yaml
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def parse_rating(text: str):
    """Official-parser-equivalent (axbench lm_judge._get_rating_from_
    completion): split on 'Rating:', strip brackets IF present. The v1
    regex REQUIRED the literal [[x]] wrapper, but gpt-4o-mini answers
    'Rating: 0' without brackets -> every score silently parsed to None
    -> 0 (the 2026-07-18 all-zero report)."""
    if "Rating:" in text:
        tail = text.split("Rating:")[-1].strip().split("\n")[0].strip()
        tail = (tail.replace("[", "").replace("]", "")
                .rstrip(".").strip('"').strip("'").strip("*").strip())
        try:
            v = float(tail)
        except ValueError:
            return None
        return int(v) if v in (0.0, 1.0, 2.0) else None
    m = re.findall(r"\[\[([0-2])\]\]", text)
    return int(m[-1]) if m else None


def hmean3(a, b, c):
    if min(a, b, c) == 0:
        return 0.0
    return 3.0 / (1.0 / a + 1.0 / b + 1.0 / c)


def main():
    args = parse_args()
    run = Path(args.run_dir)
    recs = [json.loads(l)
            for l in open(run / f"records_{args.config}.jsonl") if l.strip()]
    n_instr = max(int(r["input_id"]) for r in recs) + 1
    half = n_instr // 2          # 0..half-1 = selection, half.. = holdout
    print(f"[axjudge] {len(recs)} generations, {n_instr} instructions "
          f"(selection {half} / holdout {n_instr - half})")

    cache_path = run / f"judge_cache_{args.config}_{args.judge_model}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                c = json.loads(line)
                cache[c["key"]] = c["scores"]
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[axjudge] RESUME: {len(cache)} cached")
    cfh = open(cache_path, "a")
    judge = Judge(args.judge_model, max_tokens=400)

    def key_of(r):
        return f"{r['arm']}|{r['concept_id']}|{r['input_id']}|{r['factor']}"

    def score_one(r):
        k = key_of(r)
        if k in cache:
            return k, cache[k]
        frag = r["output"][:2000]
        sc = {}
        sc["concept"] = parse_rating(judge(
            T_CONCEPT.format(concept=r["concept"], sentence=frag)))
        sc["instruct"] = parse_rating(judge(
            T_INSTRUCT.format(instruction=r["instruction"], sentence=frag)))
        sc["fluency"] = parse_rating(judge(T_FLUENCY.format(sentence=frag)))
        sc = {k2: (0 if v is None else v) for k2, v in sc.items()}
        return k, sc

    def run_batch(rows, tag):
        todo = [r for r in rows if key_of(r) not in cache]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, sc in ex.map(score_one, todo):
                cache[k] = sc
                cfh.write(json.dumps({"key": k, "scores": sc}) + "\n")
        cfh.flush()
        print(f"[axjudge] {tag}: +{len(todo)}")

    def overall(r):
        sc = cache[key_of(r)]
        return hmean3(sc["concept"], sc["instruct"], sc["fluency"])

    by = defaultdict(list)
    for r in recs:
        by[(r["arm"], int(r["concept_id"]))].append(r)

    # phase 1: judge selection halves; pick best factor; judge holdout
    results = defaultdict(dict)     # arm -> concept -> holdout score
    subscores = defaultdict(lambda: defaultdict(list))
    for (arm, cid), rows in sorted(by.items()):
        sel = [r for r in rows if int(r["input_id"]) < half]
        run_batch(sel, f"{arm}/c{cid}/sel")
        by_factor = defaultdict(list)
        for r in sel:
            by_factor[float(r["factor"])].append(overall(r))
        best_f = max(by_factor, key=lambda f: sum(by_factor[f]) /
                     len(by_factor[f]))
        hold = [r for r in rows if int(r["input_id"]) >= half
                and float(r["factor"]) == best_f]
        run_batch(hold, f"{arm}/c{cid}/hold@{best_f}")
        if hold:
            results[arm][cid] = sum(overall(r) for r in hold) / len(hold)
            for r in hold:
                sc = cache[key_of(r)]
                for k2 in ("concept", "instruct", "fluency"):
                    subscores[arm][k2].append(sc[k2])
    cfh.close()

    lines = [f"# AxBench steering reproduction — {args.config} "
             f"(judge {args.judge_model}, factor selection half / holdout "
             f"half per evaluate.py)",
             "",
             "| arm | mean overall (holdout) | paper Table 2 | concept | "
             "instruct | fluency | n concepts |",
             "|---|---|---|---|---|---|---|"]
    for arm in sorted(results):
        vals = list(results[arm].values())
        mean = sum(vals) / len(vals)
        anch = ANCHORS.get(args.config, {}).get(arm, "—")
        subs = {k2: sum(v) / len(v) for k2, v in subscores[arm].items()}
        lines.append(f"| {arm} | **{mean:.3f}** | {anch} | "
                     f"{subs['concept']:.2f} | {subs['instruct']:.2f} | "
                     f"{subs['fluency']:.2f} | {len(vals)} |")
    lines += ["",
              "ll_set10 = LinguaLens's set-10 + full recon replacement on "
              "the same vanilla latent and the same AlpacaEval test data — "
              "cross-protocol arm, not part of AxBench.",
              "Anchor reading: pattern + range (their runs use Neuronpedia "
              "maxValue for max_act and 500 concepts; ours recomputes "
              "max_act from their eval set and runs the first "
              "N concepts)."]
    out = run / f"report_{args.config}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[axjudge] wrote {out}")


if __name__ == "__main__":
    main()
