#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Intervener L12 (INTERVENER_PLAN.md): learned intervention generator.
# Batch:      cd ~/SAE-LEWIS && qsub run_intervener_l12.sh   (login node)
# Interactive: qsub -I -l select=1 -W group_list=go25 -q interact-g
#              then: cd ~/SAE-LEWIS && git pull && bash run_intervener_l12.sh
#
# Multi-session: training checkpoints every 1000 steps and resumes; rerun /
# resubmit until "INTERVENER L12 DONE". When training completes, the probe
# runs automatically: the SAME 500-pair frame as the steer champion
# (blocklist, k 64/64, seed 42), conditions true/empty/random.
# Bar = steer0.5 exact 0.2385@499; floors raw 0.0601 / random 0.0521;
# empty must stay ~copy (premise protection).

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
OUT=$V6/intervener_l12

if [ ! -f "$OUT/intervener-final.pt" ]; then
    python train_intervener.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --inject-layer 12 \
        --max-steps 40000 \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 \
        --norm-alpha 0.5 --norm-reg-w 0.05 --null-norm-w 0.1 \
        --dev-batches 48 --eval-steps 2000 \
        --resume --device cuda
fi

if [ -f "$OUT/intervener-final.pt" ] && [ ! -f "$OUT/probe500/report.md" ]; then
    python scripts/eval_clamp_baseline.py \
        --llm2vec-dir "$LLM2VEC" --blocklist "$BLOCKLIST" \
        --output-dir "$OUT/probe500" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --intervention learned \
        --intervener-ckpt "$OUT/intervener-final.pt" \
        --device cuda
fi

if [ -f "$OUT/probe500/report.md" ]; then
    echo "==================== INTERVENER L12 DONE ===================="
    cat "$OUT/probe500/report.md"
    echo
    echo "Bar: steer0.5 exact 0.2385@499 (same frame). Controls: empty"
    echo "learned ~ copy (premise protection), random ~ floor 0.05."
fi
