#!/bin/bash
# LinguaLens intervention-evaluation reproduction — JUDGE (CPU, prepost).
#   qsub -I -l select=1 -W group_list=go25 -q prepost
#   cd SAE-LEWIS && git pull && source env-c/bin/activate && bash run_ll_repro_judge.sh
# Needs OPENAI_API_KEY (judge = gpt-4o, the paper's choice). Resumable
# (judge_cache_gpt-4o.jsonl). ~1600 pairwise judgments.
set -eo pipefail
cd "$(dirname "$0")"
[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }
[ -f runs/ll_repro/records.jsonl ] || { echo "run run_ll_repro.sh (GPU) first"; exit 1; }

python scripts/judge_ll_repro.py --run-dir runs/ll_repro

echo "==================== LL-REPRO JUDGE DONE ===================="
