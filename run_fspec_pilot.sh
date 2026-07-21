#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Feature-spec pilot at L12 (user 2026-07-22: identification on the 4,451
# pool for ALL arms; eval on the canonical 500):
#   1. ef + steer + raw with --feature-spec (spec = pool-mean delta;
#      steer dvec = same spec @ W_dec — mechanism-matched)
#   2. LinguaLens complete protocol: FRC top-3 (pool) + clamp
#   3. AxBench complete protocol: AUROC top-1 (pool) + steer
# All guarded; resubmit past walltime kills.
# Batch: qsub -N fspilot run_fspec_pilot.sh

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

[ -f $FS/l12_spec.json ] || { echo "missing $FS/l12_spec.json"; exit 1; }

if [ ! -f $P/fspec_probe_l12/report.md ]; then
    python scripts/eval_ef_bare.py \
        --frame repeat --feature-spec $FS/l12_spec.json \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --k-amp 64 --k-sup 64 --steer-alpha 0.5 \
        --ef-ckpt "$CKPT" --arms ef,steer,raw \
        --sample-size 500 --device cuda \
        --output-dir $P/fspec_probe_l12
fi
if [ ! -f $P/fspec_clamp_frc3_l12/report.md ]; then
    python scripts/eval_clamp_baseline.py \
        --feature-sets $FS/l12_frc_r3.json \
        --intervention clamp --clamp-values 5,10,20 \
        --conditions true,empty,random \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --sample-size 500 \
        --output-dir $P/fspec_clamp_frc3_l12 --device cuda
fi
if [ ! -f $P/fspec_steer_auroc1_l12/report.md ]; then
    python scripts/eval_clamp_baseline.py \
        --intervention steer --feature-sets $FS/l12_auroc_r1.json \
        --clamp-values 0.5,1 --conditions true,empty,random \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --sample-size 500 \
        --output-dir $P/fspec_steer_auroc1_l12 --device cuda
fi

echo "==================== FSPEC-PILOT-DONE ===================="
