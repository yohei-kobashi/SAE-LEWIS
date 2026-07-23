#!/bin/bash -l

# T2+T3 composition (user 2026-07-23): champion nb recipe from scratch on
# the CTX-measured z sidecar (T2, cache already built) + insertion-loss
# boost 1.5 (T3). Orthogonal mechanisms (cache vs loss weight) — aiming
# for T3's ablation (0.150) and T2's enhancement (0.140) together.
# Then (7)-spec evals, both directions.
# Run inside interact-g (resumable): bash run_v6t23.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

SRC=runs/prod_gemma_v4
CTX=$SRC/corruption_zctx_l12
CTXDEV=$SRC/corruption_seldev_zctx_l12
[ -f $CTX/build.done ] || { echo "ctx cache missing"; exit 1; }

CACHE_DIR=$CTX DEV_DIR=$CTXDEV \
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
    FLOW_INIT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
    NORM_REG_W=0.0 NULL_NORM_W=0.0 INS_BOOST=1.5 \
    OUT_SUFFIX=_v6t23 MAX_STEPS=40000 bash run_ef_editor.sh

P=runs/prod_gemma_v6
FS=runs/feature_specs
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
    if [ ! -f $P/fs_v6t23_l12$DIRX/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_specctx.json \
            --fspec-scale 3.5 --conditions true,random --arms ef \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v6t23/eflm-final.pt \
            --sample-size 500 --device cuda \
            --output-dir $P/fs_v6t23_l12$DIRX $EXTRA
    fi
done

echo "==================== V6T23-DONE ===================="
