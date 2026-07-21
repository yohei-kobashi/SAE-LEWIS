#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# AxBench-FAITHFUL enhancement arm (user audit 2026-07-22: the fs_axb amp
# cells were near no-ops — suppress-at-source on a source without the
# feature; and 0.5/1 is not their factor grid):
#   h + factor * max_act * W_dec[AUROC-r1 latent], factors from their
#   no_grad.yaml grid (subset 0.2/0.6/1/2/4 of [0.2..5]), max_act from the
#   identification POOL (their dataset-side path, eval-clean), direction =
#   amp (their protocol IS enhancement). Per-layer factor selected on
#   pool-dev; the 500-pair run records ALL factors so their per-concept
#   tune-on-eval convention can also be reported (labelled as such).
# Batch: qsub -N fsaxb run_fspec_axb.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SPLIT=runs/tables/eval_split.json
GRID=0.2,0.6,1,2,4

for LAYER in 12 4 20; do
  case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy ;;
    12) SAE="layer_12/width_16k/average_l0_82/params.npz"
        BLK=runs/blocklist/blocklist.npy ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy ;;
  esac
  MX=$FS/l${LAYER}_auroc_r1_maxact.json
  if [ ! -f $MX ]; then
      python scripts/extract_pool_maxact.py \
          --pairs $FS/l${LAYER}_pairs.jsonl \
          --sets $FS/l${LAYER}_auroc_r1.json \
          --split $SPLIT --out $MX
  fi
  AX () {  # AX <outdir> <extra...>
      OUT=$1; shift
      if [ ! -f $P/$OUT/report.md ]; then
          python scripts/eval_clamp_baseline.py \
              --intervention steer --feature-sets $FS/l${LAYER}_auroc_r1.json \
              --fsets-enhance --fsets-maxact $MX \
              --clamp-values $GRID \
              --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer $LAYER \
              --blocklist "$BLK" --device cuda \
              --output-dir $P/$OUT --reverse-pairs "$@"
      fi
  }
  # dev: pick the per-layer factor without touching the eval 500
  AX fs_axbdev_l$LAYER --pool-dev $SPLIT --sample-size 100 --conditions true
  if [ ! -f $P/fs_axbE_scale_l$LAYER.txt ]; then
      for F in 0.2 0.6 1 2 4; do
          printf "%s %s\n" "$F" \
            "$(grep -E "^\| true \| steer$F " $P/fs_axbdev_l$LAYER/report.md \
               | head -1 | awk -F'|' '{print $4}' | tr -d ' ')"
      done | sort -k2 -gr | head -1 | cut -d' ' -f1 \
           > $P/fs_axbE_scale_l$LAYER.txt
  fi
  echo "[axb] L$LAYER dev-selected factor: $(cat $P/fs_axbE_scale_l$LAYER.txt)"
  # eval: all factors recorded (dev-selected column = the clean number)
  AX fs_axbE_l${LAYER}_amp --sample-size 500 --conditions true,empty,random
done

echo "==================== FS-AXB-DONE ===================="
