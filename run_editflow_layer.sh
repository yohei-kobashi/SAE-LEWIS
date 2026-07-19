#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Layer-native EF foundation (user decision 2026-07-20: L4/L20 must be
# built EXACTLY like L12 — first train the layer's own token-output EF
# champion, then warm-start the v5f editor from ITS encoder, instead of
# transplanting the L12-native S3 encoder).
# Stages (guards + resume; resubmit until "EF-LAYER L$LAYER DONE"):
#   1. S2-config from scratch (cold feature-token conditioning, 100k —
#      the S2 budget that fixed Z1b's cold-start lag; no editor warm
#      because no layer-native editor exists)
#   2. S3 localized warm (+50k, lam_prop 4.0) — the champion config
#   3. probe200 (lambda-IoU gate diagnostics)
# Then: qsub -v LAYER=$LAYER,FRAME=repeat,... FLOW_INIT=<this final>
# Batch: qsub -v LAYER=4 run_editflow_layer.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

LAYER=${LAYER:?set LAYER=4 or 20}
case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy ;;
    *)  echo "L12 already has editflow_s3"; exit 1 ;;
esac

LLM2VEC=runs/mcgill_gemma_repro_3k/final
CACHE=runs/prod_gemma_v4/corruption_z_l$LAYER
DEV=runs/prod_gemma_v4/corruption_seldev_z_l$LAYER
S2OUT=runs/prod_gemma_v6/editflow_s2_l$LAYER
S3OUT=runs/prod_gemma_v6/editflow_s3_l$LAYER

[ -f "$CACHE/meta.json" ] || { echo "sidecar missing: $CACHE"; exit 1; }

if [ ! -f "$S2OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$CACHE" \
        --dev-corruption-dir "$DEV" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$S2OUT" \
        --sae-path "$SAE" \
        --rate-param hazard \
        --cond-mode feature-tokens \
        --true-align \
        --lora-r 32 \
        --max-steps 100000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 4000 \
        --batch-size 8 --num-workers 2 \
        --resume --device cuda
fi

if [ -f "$S2OUT/editflow-final.pt" ] \
   && [ ! -f "$S3OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$CACHE" \
        --dev-corruption-dir "$DEV" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$S3OUT" \
        --sae-path "$SAE" \
        --init-from-editflow "$S2OUT/editflow-final.pt" \
        --rate-param hazard \
        --cond-mode feature-tokens \
        --true-align \
        --lora-r 32 \
        --lam-prop 4.0 \
        --max-steps 50000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 4000 \
        --batch-size 8 --num-workers 2 \
        --resume --device cuda
fi

if [ -f "$S3OUT/editflow-final.pt" ] \
   && [ ! -f "$S3OUT/probe/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$S3OUT/editflow-final.pt" \
        --output-dir "$S3OUT/probe" \
        --sae-path "$SAE" --sae-layer "$LAYER" \
        --cond-scope local --blocklist "$BLK" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --steps 48 --steer-lambda 1 \
        --decode det,thr0.1,thr0.5 \
        --device cuda
fi

if [ -f "$S3OUT/probe/probe_report.md" ]; then
    echo "==================== EF-LAYER L$LAYER DONE ===================="
    echo "next: qsub -v LAYER=$LAYER,FRAME=repeat,EDIT_ONLY=1,LAM_SUP=0.2,FLOW_INIT=$S3OUT/editflow-final.pt,OUT_SUFFIX=_v5f2,MAX_STEPS=40000 run_ef_editor.sh"
fi
