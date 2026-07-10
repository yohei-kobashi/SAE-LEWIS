#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

# SAE-EF prototype (EDIT_FLOWS_PLAN.md §4-§5): 30k-step edit-flow training
# on the v6 cache + LinguaLens probe with the promotion-gate measurements.
# Same data / conditioning spec / budget class as the v6 pilot arms, so the
# probe rows are directly comparable with probe_local / probe_cmlm.
#   Phase 0  train_editflow 30k (init from v6 editor; seldev monitor)
#   Phase 1  editflow_probe (λ-IoU + det/stoch decode + empty no-edit)
# ~2.5h total. Resumable: re-submit until it prints "EF PILOT DONE".
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
OUT=$V6/editflow_pilot

if [ ! -f "$OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --init-from-editor "$V6/editor/editor-final.pt" \
        --max-steps 30000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 2000 \
        --batch-size 8 --num-workers 2 \
        --resume \
        --device cuda
fi

if [ ! -f "$OUT/probe/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$OUT/editflow-final.pt" \
        --output-dir "$OUT/probe" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --steps 48 --decode det,stoch --steer-lambda 1 \
        --device cuda
fi

echo "==================== EF PILOT DONE ===================="
echo "results: $OUT/probe/probe_report.md"
echo ""
echo "Promotion gates (EDIT_FLOWS_PLAN §5) — reference numbers:"
echo "  (a) true λ-IoU vs tagger OOD span IoU ~0.30 (and >> empty/random)"
echo "  (b) det exact/sim >= stoch"
echo "  (c) empty no_edit >= 0.99"
echo "  overall: bucket exact vs editor pipeline (e2e exact 0.114;"
echo "           gold-site ceiling cmlm8+lens1 0.447 — EF has no gold"
echo "           sites, so e2e is the honest comparison)"
