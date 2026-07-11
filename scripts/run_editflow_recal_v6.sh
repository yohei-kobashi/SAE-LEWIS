#!/bin/bash
# SAE-EF re-probe after the pilot verdict (README §13.8): gate (a) passed
# big (λ-IoU 0.73 vs tagger ~0.30) but the decode barely fired — the rate
# head is globally under-scaled vs the w(t) target. This rerun tests the
# zero-training fix: calibrated thresholding (fire when λ ≥ F·w(t)) at
# three F values, plus the tagger COUNT-ORACLE IoU on the same pairs so
# gate (a) is apples-to-apples. No det/stoch rerun (already measured).
# ~30-40 min on one GPU:  bash scripts/run_editflow_recal_v6.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
OUT=$V6/editflow_pilot/probe_recal

python scripts/editflow_probe.py \
    --llm2vec-dir "$LLM2VEC" \
    --editflow-ckpt "$V6/editflow_pilot/editflow-final.pt" \
    --output-dir "$OUT" \
    --cond-scope local --blocklist "$BLOCKLIST" \
    --k-amp 64 --k-sup 64 --sample-size 200 \
    --steps 48 --decode thr0.05,thr0.1,thr0.25 --steer-lambda 1 \
    --tagger-ckpt "$V6/tagger/tagger-final.pt" \
    --device cuda

echo "==================== EF RECAL DONE ===================="
sed -n '/## Gate (a)/,/## Multi-site/p' "$OUT/probe_report.md"
echo "Reading:"
echo "  - tagger count-oracle row = the honest gate-(a) bar for lambda-IoU 0.73"
echo "  - rate-calibration table: ratio << 1 confirms the under-scaled head"
echo "  - thr rows: if exact/sim recover with F in {0.05..0.25}, the decode"
echo "    failure was pure calibration -> fixable at inference; if not,"
echo "    retrain with a larger rate-head LR / longer schedule."