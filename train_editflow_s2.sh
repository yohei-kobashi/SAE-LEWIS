#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

# S2 (EDIT_FLOWS_ZERO §5): full co-adapted build at adequate budget —
# the anti-greedy step. Everything the S/Z series validated goes in at
# once: hazard rate head (S1: ranking record + structural premise
# safety), feature-token conditioning (Z1b: fixes magnitude + premise
# structurally; its ranking lag was a 30k budget artifact), true
# alignment (Z2 teacher), LoRA r=32 (capacity up; editor's r=16 LoRA is
# NOT transferable so LoRA starts fresh — conditioning/Proj_A still
# warm-starts from the editor), 100k steps. Probe folds in the S1b
# F sweep + bo-K so no separate decode job is needed.
# Gates: lambda-IoU >= 0.77 (beat S1), some thr F reaches the S0
# champion (exact 0.1859 / sim 0.6622), empty no_edit 1.00.
# ~7h; resubmit until "S2 DONE".
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
OUT=$V6/editflow_s2

if [ ! -f "$OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --init-from-editor "$V6/editor/editor-final.pt" \
        --rate-param hazard \
        --cond-mode feature-tokens \
        --true-align \
        --lora-r 32 \
        --max-steps 100000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 4000 \
        --batch-size 8 --num-workers 2 \
        --resume --device cuda
fi

if [ ! -f "$OUT/probe/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$OUT/editflow-final.pt" \
        --output-dir "$OUT/probe" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --steps 48 --steer-lambda 1 \
        --decode det,thr0.05,thr0.1,thr0.25,thr0.5,bo4@temp0.7 \
        --device cuda
fi

echo "==================== S2 DONE ===================="
sed -n '/## Gate (a)/,$p' "$OUT/probe/probe_report.md" | head -80
echo "Gates: (i) lambda-IoU >= 0.77 (S1's record; recovery proves Z1b's"
echo "  ranking lag was budget, not design); (ii) best thr F reaches the"
echo "  S0 champion exact 0.1859 / sim 0.6622; (iii) empty no_edit 1.00"
echo "  at every F, random >= 0.88 at the chosen operating point;"
echo "  (iv) check whether mean p's t-decline (S1: 0.19->0.08) flattened"
echo "  with feature tokens + 100k. Winner -> S3 (Localized CTMC)."
