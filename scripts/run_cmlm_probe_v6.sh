#!/bin/bash
# C1-0 (V7_PLAN): inference-only iterative Mask-Predict probe on the v6
# editor — does iterative refinement coordinate multi-site fills WITHOUT
# reveal-curriculum training? Same 190 pairs / seed as probe_local, so
# rows are directly comparable. parallel + lens1 rerun as in-table
# baselines; cmlm8 and cmlm8+lens1 are the new evidence. ~25 min.
# Run on miyabi with a GPU session:  bash scripts/run_cmlm_probe_v6.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
OUT=$V6/probe_cmlm

python scripts/lingualens_gold_template_probe.py \
    --llm2vec-dir "$LLM2VEC" \
    --editor-ckpt "$V6/editor/editor-final.pt" \
    --output-dir "$OUT" \
    --cond-scope local --blocklist "$BLOCKLIST" \
    --k-amp 64 --k-sup 64 --sample-size 200 \
    --modes parallel,cmlm8,cmlm8+lens1 \
    --steer-lambdas 1 \
    --device cuda

echo "==================== C1-0 VERDICT ===================="
sed -n '/## Fill accuracy/,/Reading guide:/p' "$OUT/probe_report.md"
echo "Decision rule (V7_PLAN C1-0):"
echo "  cmlm8 > parallel at n_edit>=2 (esp. exact) -> C1 expected value HIGH"
echo "  cmlm8 ~ parallel -> inconclusive (editor never saw revealed gold);"
echo "                      C1 GO/NO-GO still needs the trained comparison,"
echo "                      but its priority stays DEMOTED (README 13.7)."
