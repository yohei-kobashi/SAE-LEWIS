#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

# LinguaLens-clamp AMP fix (audit 2026-07-22: fs_clamp_l*_amp were no-ops
# — suppress-at-source magnitudes are ~0 when src is the counterfactual;
# verified true=0.1122 <= raw floor 0.1202 at L20). Their ENHANCEMENT
# protocol is "SET the identified latent to 10" (force-insert, OpenSAE
# semantics — already what SaeClampHook does for amp members), so amp
# cells put the FRC-r3 latents in the amp set via --fsets-enhance;
# clamp10 = their faithful value, 5/20 robustness, clampZ = pool max_act.
# Batch: qsub -N fsclampa run_fspec_clamp_amp.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SPLIT=runs/tables/eval_split.json

for LAYER in 12 4 20; do
  case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy ;;
    12) SAE="layer_12/width_16k/average_l0_82/params.npz"
        BLK=runs/blocklist/blocklist.npy ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy ;;
  esac
  MX=$FS/l${LAYER}_frc_r3_maxact.json
  if [ ! -f $MX ]; then
      python scripts/extract_pool_maxact.py \
          --pairs $FS/l${LAYER}_pairs.jsonl \
          --sets $FS/l${LAYER}_frc_r3.json \
          --split $SPLIT --out $MX
  fi
  if [ ! -f $P/fs_clampE_l${LAYER}_amp/report.md ]; then
      python scripts/eval_clamp_baseline.py \
          --intervention clamp --feature-sets $FS/l${LAYER}_frc_r3.json \
          --fsets-enhance --fsets-maxact $MX \
          --clamp-values 5,10,20 --conditions true,empty,random \
          --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer $LAYER \
          --blocklist "$BLK" --sample-size 500 --device cuda \
          --output-dir $P/fs_clampE_l${LAYER}_amp --reverse-pairs
  fi
done

echo "==================== FS-CLAMP-AMP-DONE ===================="
