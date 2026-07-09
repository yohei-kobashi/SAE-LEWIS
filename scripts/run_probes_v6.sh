#!/bin/bash
# v6 verdict probes — fills the 2x2 evidence matrix in three runs:
#   data hypothesis    : probe_treated vs probe_control (parallel rows)
#   geometry hypothesis: lens{λ} rows vs parallel, per editor
# Run on miyabi with a GPU session:  bash scripts/run_probes_v6.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V5=./runs/prod_gemma_v5
LLM2VEC=runs/mcgill_gemma_repro_3k/final
LAMBDAS="0.5,1,2,4,8"

probe () {  # $1 = editor ckpt, $2 = output dir
    python scripts/lingualens_gold_template_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editor-ckpt "$1" \
        --output-dir "$2" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --modes parallel \
        --steer-lambdas "$LAMBDAS" \
        --device cuda
}

probe "$V5/pilot_v6/editor_control/editor-final.pt" "$V5/pilot_v6/probe_control"
probe "$V5/pilot_v6/editor_treated/editor-final.pt" "$V5/pilot_v6/probe_treated"
probe "$V5/editor/editor-final.pt"                  "$V5/gold_template_probe_lens"

echo "==================== 2x2 VERDICT ===================="
for d in "$V5/pilot_v6/probe_control" "$V5/pilot_v6/probe_treated" \
         "$V5/gold_template_probe_lens"; do
    echo "---- $d ----"
    sed -n '/## Fill accuracy/,/^$/p' "$d/probe_report.md"
done
