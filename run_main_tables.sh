#!/bin/bash
# Paper main tables (CPU, minutes — run in any session, no qsub needed):
#   (1) main_metrics_499  — ALL systems on the original 499 pairs
#       (clamp/prompt/pipeline only exist there); the cross-system table.
#   (2) main_metrics_997  — ef32 / steer / routed / oracle on the full
#       997 pairs INCLUDING the untouched confirmation block; the
#       headline table for the routed system.
# Usage: bash run_main_tables.sh
set -eo pipefail
cd "$(dirname "$0")"

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final

echo "==== (1) all systems, original 499 pairs ===================="
python scripts/score_edit_metrics.py \
    --llm2vec-dir "$LLM2VEC" \
    --cand ef32="$V6/ksweep500/records.jsonl":k32 \
    --cand ef64="$V6/editflow_s3/probe500/records.jsonl":thr0.1 \
    --cand steer="$V6/steer_baseline500/records.jsonl":steer0.5 \
    --cand clamp="$V6/clamp_baseline500/records.jsonl":clamp10 \
    --cand prompt="$V6/prompt_baseline500/records.jsonl":prompt8 \
    --cand pipeline="$V6/eval_lingualens_final/records.jsonl": \
    --router-head ef32 --router-fallback steer --router-T 1 \
    --out runs/tables/main_metrics_499

echo
echo "==== (2) routed system, full 997 pairs (incl. confirmation) =="
python scripts/score_edit_metrics.py \
    --llm2vec-dir "$LLM2VEC" \
    --cand ef32="$V6/ksweep500/records.jsonl":k32 \
    --cand steer="$V6/steer_baseline500/records.jsonl":steer0.5 \
    --router-head ef32 --router-fallback steer --router-T 1 \
    --out runs/tables/main_metrics_997

echo
echo "==================== MAIN TABLES DONE ===================="
echo "outputs: runs/tables/main_metrics_499{.md,_per_feature.csv,.json}"
echo "         runs/tables/main_metrics_997{.md,_per_feature.csv,.json}"
echo "Reading: table (1) places every system on the common original"
echo "  sample (cross-system columns incl. SARI/BLEU); table (2) is the"
echo "  headline — routed exact@1 vs oracle exact@K on 997 pairs, of"
echo "  which ~498 are the untouched confirmation block. Per-feature"
echo "  CSVs carry the 99-phenomenon breakdown for both."
