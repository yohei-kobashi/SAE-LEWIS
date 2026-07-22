#!/bin/bash
# amp-direction improvements, order B -> A -> C (user 2026-07-22):
#   B. amp-specific fspec-scale dev selection (amp inherited sup's 3.5)
#      -> eval500 amp at the amp-selected scale
#   A. retrieval spec: build table, dev-sweep m in {1,5,15} (amp, at B's
#      scale) -> eval500 amp for dev-best m; sup dev check for regression
#   C. amp-only shaping: dev probe only (eval held until A/B judged)
# Run inside interact-g (chain marker AMP-IMPROVE-DONE). All guarded.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SAE=layer_12/width_16k/average_l0_82/params.npz
BLK=runs/blocklist/blocklist.npy
CKPT=$P/eflm_l12_v5f_nobudget/eflm-final.pt
SPLIT=runs/tables/eval_split.json

EF () {  # EF <outdir> <extra...>
    OUT=$1; shift
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$CKPT" --arms ef --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}
exact_of () {
    grep -E "^\| true \| ef " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}

# ---- B. amp scale selection --------------------------------------------
for S in 1.5 2.5 3.5 5.0; do
    EF fs_adev_l12_s${S/./} --pool-dev $SPLIT --sample-size 100 \
       --conditions true --reverse-pairs --fspec-scale $S
done
if [ ! -f $P/fs_amp_scale_l12.txt ]; then
    BESTS=$(for S in 1.5 2.5 3.5 5.0; do
        printf "%s %s\n" "$S" \
          "$(exact_of $P/fs_adev_l12_s${S/./}/report.md)"
    done | sort -k2 -gr | head -1 | cut -d' ' -f1)
    echo "$BESTS" > $P/fs_amp_scale_l12.txt
fi
SCA=$(cat $P/fs_amp_scale_l12.txt)
echo "[ampimp] B: amp-selected scale = $SCA"
EF fs_ampB_l12_amp --sample-size 500 --conditions true,random \
   --reverse-pairs --fspec-scale $SCA

# ---- A. retrieval spec --------------------------------------------------
if [ ! -f $FS/l12_retrieve.json ]; then
    python scripts/build_retrieval_table.py \
        --pairs $FS/l12_pairs.jsonl --split $SPLIT \
        --out $FS/l12_retrieve.json
fi
for M in 1 5 15; do
    EF fs_adev_l12_m$M --pool-dev $SPLIT --sample-size 100 \
       --conditions true --reverse-pairs --fspec-scale $SCA \
       --fspec-retrieve $FS/l12_retrieve.json --retrieve-m $M
done
if [ ! -f $P/fs_amp_m_l12.txt ]; then
    BESTM=$(for M in 1 5 15; do
        printf "%s %s\n" "$M" "$(exact_of $P/fs_adev_l12_m$M/report.md)"
    done | sort -k2 -gr | head -1 | cut -d' ' -f1)
    echo "$BESTM" > $P/fs_amp_m_l12.txt
fi
BESTM=$(cat $P/fs_amp_m_l12.txt)
echo "[ampimp] A: dev-best m = $BESTM"
EF fs_ampA_l12_amp --sample-size 500 --conditions true,random \
   --reverse-pairs --fspec-scale $SCA \
   --fspec-retrieve $FS/l12_retrieve.json --retrieve-m $BESTM
# sup regression check (dev only)
EF fs_adev_l12_m${BESTM}_sup --pool-dev $SPLIT --sample-size 100 \
   --conditions true --fspec-scale 3.5 \
   --fspec-retrieve $FS/l12_retrieve.json --retrieve-m $BESTM

# ---- C. amp-only shaping (dev only) ------------------------------------
EF fs_adev_l12_amponly --pool-dev $SPLIT --sample-size 100 \
   --conditions true --reverse-pairs --fspec-scale $SCA --amp-only

echo "==================== AMP-IMPROVE-DONE ===================="
