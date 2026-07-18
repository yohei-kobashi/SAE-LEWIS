"""One-shot diagnosis of the all-zero AxBench judge: call the REAL Judge
with the REAL templates on a few REAL records and print the full raw
completions plus parse_rating's verdict. Pure stdlib — runs on the miyabi
login node (bash -lc so OPENAI_API_KEY comes from .bashrc).

    cd ~/SAE-LEWIS && bash -lc "python3 scripts/debug_axjudge.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_axbench_repro import (T_CONCEPT, T_FLUENCY, T_INSTRUCT,  # noqa
                                 parse_rating)
from judge_paper_metrics import Judge  # noqa

REC = "runs/axbench_repro/records_prod_2b_l20_v1.jsonl"


def main():
    recs = [json.loads(l) for l in open(REC)]
    picks = []
    for arm, factor in (("sae", 1.0), ("sae", 0.2), ("ll_set10", 1.0)):
        picks.append(next(r for r in recs
                          if r["arm"] == arm and r["factor"] == factor))
    judge = Judge("gpt-4o-mini", max_tokens=400)
    for r in picks:
        frag = r["output"][:2000]
        print("=" * 70)
        print(f"arm={r['arm']} factor={r['factor']} concept={r['concept']!r}")
        print(f"output[:120]: {frag[:120]!r}")
        for name, prompt in (
                ("concept", T_CONCEPT.format(concept=r["concept"],
                                             sentence=frag)),
                ("instruct", T_INSTRUCT.format(instruction=r["instruction"],
                                               sentence=frag)),
                ("fluency", T_FLUENCY.format(sentence=frag))):
            raw = judge(prompt)
            print(f"--- {name}: parse={parse_rating(raw)!r}")
            print(f"RAW FULL: {raw!r}")


if __name__ == "__main__":
    main()
