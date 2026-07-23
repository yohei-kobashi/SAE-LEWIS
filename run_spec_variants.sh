#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

# Improvements ①x② JOINT selection (user 2026-07-22: ①が有効なら①前提で
# ②を選ぶ — 逐次でなく同時選択):
#  A. build 8 spec variants (k8/16/32/64 x nofilter/c70)
#  B. pool-dev probe (100 pairs, sup) over {base+8 variants} x {src-gate
#     off/on} = 18 configs -> joint dev-best
#  C. eval-500 sup: narrowing curve k8-64 (gate off, ToDo① analysis,
#     ALL reported) + joint-best config
#  D. eval-500 amp: joint-best config
# Batch: qsub -N specvar run_spec_variants.sh

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
SC=$(cat $P/fs_scale_l12.txt)

if [ ! -f $FS/l12_spec_k8_c70.json ]; then
    python scripts/build_spec_variants.py \
        --pairs $FS/l12_pairs.jsonl --split $SPLIT \
        --base $FS/l12_spec.json --out-prefix $FS/l12_spec
fi

EF () {  # EF <outdir> <specjson> <extra...>
    OUT=$1; SPEC=$2; shift 2
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $SPEC --fspec-scale $SC \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$CKPT" --arms ef --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}
exact_of () {
    grep -E "^\| true \| ef " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}

# ---- B. dev probes: {base+8 variants} x {gate off/on} -------------------
VARS="base k8 k16 k32 k64 k8_c70 k16_c70 k32_c70 k64_c70"
specof () { [ "$1" = base ] && echo $FS/l12_spec.json || echo $FS/l12_spec_$1.json; }
for V in $VARS; do
    EF fs_kdev_l12_$V "$(specof $V)" \
       --pool-dev $SPLIT --sample-size 100 --conditions true
    EF fs_kdev_l12_${V}_g "$(specof $V)" --src-gate \
       --pool-dev $SPLIT --sample-size 100 --conditions true
done
if [ ! -f $P/fs_kbest_l12.txt ]; then
    BEST=$(for V in $VARS; do
        printf "%s %s\n" "$V" "$(exact_of $P/fs_kdev_l12_$V/report.md)"
        printf "%s_g %s\n" "$V" "$(exact_of $P/fs_kdev_l12_${V}_g/report.md)"
    done | sort -k2 -gr | head -1 | cut -d' ' -f1)
    echo "$BEST" > $P/fs_kbest_l12.txt
fi
BEST=$(cat $P/fs_kbest_l12.txt)
echo "[specvar] joint dev-best config: $BEST"
BV=${BEST%_g}
BGATE=""
case "$BEST" in *_g) BGATE=--src-gate ;; esac

# ---- C. eval narrowing curve (sup) --------------------------------------
for K in 8 16 32 64; do
    EF fs_k${K}_l12 $FS/l12_spec_k$K.json \
       --sample-size 500 --conditions true,random
done
EF fs_kbestv_l12 "$(specof $BV)" $BGATE \
   --sample-size 500 --conditions true,random

# ---- D. joint-best amp --------------------------------------------------
EF fs_kbestv_l12_amp "$(specof $BV)" $BGATE \
   --sample-size 500 --conditions true,random --reverse-pairs

echo "==================== SPEC-VARIANTS-DONE ===================="
