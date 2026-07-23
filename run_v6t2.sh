#!/bin/bash -l

# v6 ablation study, arm T2 (user 2026-07-23, scratch-unified form):
# champion nb recipe trained on a CTX-measured z sidecar — the training
# cache's z_X/z_X' re-measured inside the repeat prompt on gemma-2-2b-it
# (the (7) operating point), so train-time conditioning matches the
# eval-time in-context specs. No T1/T3 additions. 40k from scratch.
# Run inside interact-g (2h sessions; every stage is resumable):
#   bash run_v6t2.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

SRC=runs/prod_gemma_v4
CTX=$SRC/corruption_zctx_l12
CTXDEV=$SRC/corruption_seldev_zctx_l12

# Stage 1 — ctx sidecar for the training cache (per-shard resume)
if [ ! -f $CTX/build.done ]; then
    python scripts/make_z_sidecar_ctx.py \
        --cache-dir $SRC/corruption_z_l12 --out-dir $CTX --layer 12 \
        && touch $CTX/build.done
fi

# Stage 2 — ctx sidecar for the dev cache
if [ ! -f $CTXDEV/build.done ]; then
    python scripts/make_z_sidecar_ctx.py \
        --cache-dir $SRC/corruption_seldev_z_l12 --out-dir $CTXDEV \
        --layer 12 && touch $CTXDEV/build.done
fi

# Stage 3 — scratch 40k on the ctx cache (champion recipe, no T1/T3)
CACHE_DIR=$CTX DEV_DIR=$CTXDEV \
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
    FLOW_INIT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
    NORM_REG_W=0.0 NULL_NORM_W=0.0 \
    OUT_SUFFIX=_v6t2 MAX_STEPS=40000 bash run_ef_editor.sh

# Stage 4 — feature-spec verdict evals (ctx spec, both directions)
P=runs/prod_gemma_v6
FS=runs/feature_specs
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
    if [ ! -f $P/fs_v6t2_l12$DIRX/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_specctx.json \
            --fspec-scale 3.5 --conditions true,random --arms ef \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v6t2/eflm-final.pt \
            --sample-size 500 --device cuda \
            --output-dir $P/fs_v6t2_l12$DIRX $EXTRA
    fi
done

echo "==================== V6T2-DONE ===================="
