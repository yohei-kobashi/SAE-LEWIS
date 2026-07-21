#!/bin/bash
# FIC judge for the ef repeat-frame arm at ALL layers x {40k, 80k}
# (user 2026-07-21: FICはL4/L12/L20の全てで40kと80k両方).
# L12 40k lives in fic_judge_l12 (complete). Each run judges the layer's
# probe500 records (ef+steer x true/random, ~2k judgments) into its own
# cache — resume-safe, so walltime kills just mean "run me again".

cd ~/SAE-LEWIS
source env-c/bin/activate

set -eo pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

for CFG in \
    "eflm_l12_v5f_80k  fic_judge_l12_80k" \
    "eflm_l4_v5f2      fic_judge_l4" \
    "eflm_l4_v5f2_80k  fic_judge_l4_80k" \
    "eflm_l20_v5f2     fic_judge_l20" \
    "eflm_l20_v5f2_80k fic_judge_l20_80k"; do
  set -- $CFG
  echo "==== judging $1 -> $2 ===="
  python scripts/eval_fic_judge.py \
      --repeat-probe500 "runs/prod_gemma_v6/$1/probe500/records.jsonl" \
      --dir-map runs/tables/lingualens_dirmap_en.json \
      --output-dir "runs/prod_gemma_v6/$2"
done

echo "==================== FIC-JUDGE-ALL-DONE ===================="
