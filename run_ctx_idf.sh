#!/bin/bash
# ⑦x③ stack: invstd/idf reweighting ON THE CTX sidecar -> dev mode pick
# (both dirs) -> eval500 both dirs at the ctx scales (3.5/3.5).
# interact-g chain, marker CTXIDF-DONE.
cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail
git pull || true
P=runs/prod_gemma_v6
FS=runs/feature_specs
SPLIT=runs/tables/eval_split.json

if [ ! -f $FS/l12_specctx_invstd.json ]; then
    python scripts/build_idf_spec.py \
        --pairs $FS/l12_pairs_ctx.jsonl --split $SPLIT \
        --base $FS/l12_specctx.json --out-prefix $FS/l12_specctx
fi

EF () {  # EF <outdir> <spec> <extra...>
    OUT=$1; SPEC=$2; shift 2
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $SPEC \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v5f_nobudget/eflm-final.pt \
            --arms ef --device cuda --output-dir $P/$OUT "$@"
    fi
}
exact_of () {
    grep -E "^\| true \| ef " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}
for MODE in idf invstd; do
    EF fs_cxn_l12_$MODE       $FS/l12_specctx_$MODE.json --pool-dev $SPLIT \
       --sample-size 100 --conditions true --fspec-scale 3.5
    EF fs_cxn_l12_${MODE}_amp $FS/l12_specctx_$MODE.json --pool-dev $SPLIT \
       --sample-size 100 --conditions true --fspec-scale 3.5 --reverse-pairs
done
for D in "" "_amp"; do
    F=$P/fs_cxn_mode_l12$D.txt
    if [ ! -f $F ]; then
        BEST=$(for MODE in idf invstd; do
            printf "%s %s\n" "$MODE" \
              "$(exact_of $P/fs_cxn_l12_$MODE$D/report.md)"
        done | sort -k2 -gr | head -1 | cut -d' ' -f1)
        echo "$BEST" > $F
    fi
done
MS=$(cat $P/fs_cxn_mode_l12.txt)
MA=$(cat $P/fs_cxn_mode_l12_amp.txt)
echo "[ctxidf] modes: sup=$MS amp=$MA"
EF fs_ctxidf_l12     $FS/l12_specctx_$MS.json --sample-size 500 \
   --conditions true,random --fspec-scale 3.5
EF fs_ctxidf_l12_amp $FS/l12_specctx_$MA.json --sample-size 500 \
   --conditions true,random --fspec-scale 3.5 --reverse-pairs
echo "==================== CTXIDF-DONE ===================="
