#!/bin/bash
# AxBench steering reproduction — JUDGE (CPU, prepost).
#   qsub -I -l select=1 -W group_list=go25 -q prepost
#   cd SAE-LEWIS && git pull && source env-c/bin/activate && bash run_axbench_repro_judge.sh
# Judge = gpt-4o-mini with the official rubric templates (verbatim).
# Lazy judging: selection half over all 14 factors, holdout at best factor.
set -eo pipefail
cd "$(dirname "$0")"
CONFIG=${CONFIG:-prod_2b_l20_v1}
[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }
[ -f "runs/axbench_repro/records_${CONFIG}.jsonl" ] || { echo "run run_axbench_repro.sh first"; exit 1; }

python scripts/judge_axbench_repro.py \
    --run-dir runs/axbench_repro --config "$CONFIG"

echo "==================== AXBENCH-REPRO JUDGE DONE ===================="
