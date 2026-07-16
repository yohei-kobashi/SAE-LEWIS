#!/bin/bash
# S6 = S3 + v7 top-up (P4 geometry + variety) + P5 mismatched-z nulls
#      + M1b MOVE riding along (the ledger's condition is now met:
#      "M1b only rides a corruption top-up that includes P4 + P5").
#
# Run INSIDE interact-g (2h cap; ~4 sessions for 50k steps, --resume picks
# up from the last checkpoint):
#   qsub -I -l select=1 -W group_list=go25 -q interact-g
#   source start_gpu_nodes.sh && cd SAE-LEWIS && git pull && bash train_editflow_s6.sh
#
# Gates at the end (same battery M1 failed, judged on probe500):
#   (i) MOVE pairs exact > 0 (target >= 4/12)   (ii) thr0.1 >= 0.2104 (S3)
#   (iii) empty no_edit 1.00 AND random no_edit >= 0.87 <- P5's whole point
#   (iv) lambda-IoU >= 0.74; empty ranking must NOT rise (M1's leak was
#        0.15 -> 0.39; the mismatch nulls exist to hold it down)
set -eo pipefail
cd "$(dirname "$0")"
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
OUT=$V6/editflow_s6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BUDGET=${BUDGET:-6600}

mkdir -p "$OUT"
set +e
timeout "$BUDGET" python train_editflow.py \
    --corruption-dir "$V4/corruption" \
    --dev-corruption-dir "$V4/corruption_seldev" \
    --llm2vec-dir "$LLM2VEC" \
    --output-dir "$OUT" \
    --init-from-editflow "$V6/editflow_s3/editflow-final.pt" \
    --rate-param hazard \
    --cond-mode feature-tokens \
    --true-align \
    --lora-r 32 \
    --lam-prop 4.0 \
    --move-ops \
    --mismatch-null-prob 0.12 \
    --max-steps 50000 \
    --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
    --dev-batches 96 --eval-steps 4000 \
    --batch-size 8 --num-workers 2 \
    --resume --device cuda
RC=$?
set -e
if [ $RC -eq 124 ]; then
    echo "[s6] BUDGET hit — resume next session (bash train_editflow_s6.sh)"
    exit 0
fi

# training finished -> probe immediately (fold the measurement into the run)
if [ -f "$OUT/editflow-final.pt" ] && [ ! -f "$OUT/probe/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$OUT/editflow-final.pt" \
        --output-dir "$OUT/probe" \
        --cond-scope local --blocklist runs/blocklist/blocklist.npy \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --steps 48 --steer-lambda 1 \
        --decode thr0.1,thr0.5 \
        --conditions true,empty,random \
        --device cuda
fi
[ -f "$OUT/probe/probe_report.md" ] && cat "$OUT/probe/probe_report.md"
echo "==================== S6 SESSION DONE ===================="
