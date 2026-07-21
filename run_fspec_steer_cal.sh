#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Per-arm fairness fix (2026-07-22): the fs_probe runs drove steer at the
# EF-selected fspec-scale, which over-drives it (L12 scale3.5: steer 0.026
# vs 0.086 at 1.0). AxBench's own convention selects the steering factor
# per method — so give steer its own pool-dev scale selection and rerun
# the steer arm at that scale, all layers x both directions.
# Batch: qsub -N fscal run_fspec_steer_cal.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SPLIT=runs/tables/eval_split.json

steer_of () {
    grep -E "^\| true \| steer " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}

for LAYER in 12 4 20; do
  case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy ;;
    12) SAE="layer_12/width_16k/average_l0_82/params.npz"
        BLK=runs/blocklist/blocklist.npy ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy ;;
  esac
  ST () {  # ST <outdir> <extra...>
      OUT=$1; shift
      if [ ! -f $P/$OUT/report.md ]; then
          python scripts/eval_ef_bare.py \
              --frame repeat --feature-spec $FS/l${LAYER}_spec.json \
              --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer $LAYER \
              --blocklist "$BLK" --k-amp 64 --k-sup 64 --steer-alpha 0.5 \
              --arms steer --device cuda \
              --output-dir $P/$OUT "$@"
      fi
  }
  SCALE_F=$P/fs_steer_scale_l$LAYER.txt
  if [ ! -f $SCALE_F ]; then
      for S in 0.5 1.0 2.0; do
          ST fs_sdev_l${LAYER}_s${S/./} --pool-dev $SPLIT --sample-size 100 \
             --conditions true --fspec-scale $S
      done
      BEST=$(for S in 0.5 1.0 2.0; do
          printf "%s %s\n" "$S" \
            "$(steer_of $P/fs_sdev_l${LAYER}_s${S/./}/report.md)"
      done | sort -k2 -gr | head -1 | cut -d' ' -f1)
      echo "$BEST" > $SCALE_F
  fi
  SC=$(cat $SCALE_F)
  echo "[steer-cal] L$LAYER steer scale=$SC"
  ST fs_steer_l$LAYER       --conditions true,random --sample-size 500 --fspec-scale $SC
  ST fs_steer_l${LAYER}_amp --conditions true,random --sample-size 500 --fspec-scale $SC --reverse-pairs
done

echo "==================== FS-STEER-CAL-DONE ===================="
