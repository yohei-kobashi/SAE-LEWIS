#!/bin/bash
# FIC judge for the L12 80k-extension probe outputs (repeat frame, ef+steer)
# — the 40k-vs-80k FIC comparison (user 2026-07-21). steer rows are
# byte-identical interventions re-judged: doubles as a judge-stability
# check against fic_judge_l12's steer E_abl.

cd ~/SAE-LEWIS
source env-c/bin/activate

set -eo pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

python scripts/eval_fic_judge.py \
    --repeat-probe500 runs/prod_gemma_v6/eflm_l12_v5f_80k/probe500/records.jsonl \
    --dir-map runs/tables/lingualens_dirmap_en.json \
    --output-dir runs/prod_gemma_v6/fic_judge_l12_80k
