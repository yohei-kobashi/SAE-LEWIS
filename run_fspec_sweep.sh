#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

# Feature-spec strength diagnostic (pilot follow-up, 2026-07-22):
# ef true copy=0.60 under feature-spec (vs 0.30 oracle) suggests the editor
# under-fires on the averaged spec. Sweep BOTH knobs at L12/sup:
#   input side  --fspec-scale 1.5 / 2.5 (spec magnitude fed to editor+dvec)
#   output side --ef-scale    1.5 / 2.5 (editor delta multiplier)
# ef arm only (steer's own alpha sweep exists in the pilot outputs).
# Batch: qsub -N fssweep run_fspec_sweep.sh

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

run1 () {  # run1 <outdir> <extra args...>
    OUT=$1; shift
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$CKPT" --arms ef --conditions true,random \
            --sample-size 500 --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}

run1 fspec_sw_in15  --fspec-scale 1.5
run1 fspec_sw_in25  --fspec-scale 2.5
run1 fspec_sw_out15 --ef-scale 1.5
run1 fspec_sw_out25 --ef-scale 2.5

# extension (2026-07-22): input-side monotone to 2.5x — find the peak
run1 fspec_sw_in35 --fspec-scale 3.5
run1 fspec_sw_in50 --fspec-scale 5.0

echo "==================== FSPEC-SWEEP-DONE ===================="

