#!/bin/bash
# B-1: INTERVENTION k-sweep — the localization spectrum, population level.
# (🔵 2026-07-17 reframing: how many activations must be intervened on for
#  the edit to succeed? Few = feature locally expressed; many = distributed.
#  This is the INTERVENTION counterpart of M0's conditioning k-sweep
#  0.060/0.126/0.210/0.190 @ k=8/16/32/64.)
#
# Run inside interact-g:
#   qsub -I -l select=1 -W group_list=go25 -q interact-g
#   source start_gpu_nodes.sh && cd SAE-LEWIS && git pull && bash run_ksweep_steer.sh
#
# Mechanics: B3 steer0.5 (the C1' causal-editing champion, greedy rewrite)
# with the instance-level spec truncated to k_amp=k_sup=K. K=64 ties the
# sweep to the published champion (steer_baseline500 ran with the 64/64
# defaults -> 0.2337@997). conditions true,empty,random so every K carries
# its own specificity floor. ~3 rewrites/pair x 500 x 7K ~= 4-5h — the
# script is per-K and per-pair resumable; rerun to extend.
set -eo pipefail
cd "$(dirname "$0")"

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final

for K in 1 2 4 8 16 32 64; do
    OUT=$V6/ksweep_steer/k$K
    if [ ! -f "$OUT/report.md" ]; then
        echo "-------- steer k_amp=k_sup=$K"
        python scripts/eval_clamp_baseline.py \
            --llm2vec-dir "$LLM2VEC" \
            --output-dir "$OUT" \
            --intervention steer --scope all \
            --clamp-values 0.5 \
            --k-amp "$K" --k-sup "$K" \
            --conditions true,empty,random \
            --sample-size 500 --device cuda
    fi
done

echo "==================== STEER K-SWEEP DONE ===================="
echo "| K | exact(true) | copy(true) | exact(random) |"
for K in 1 2 4 8 16 32 64; do
    R=$V6/ksweep_steer/k$K/report.md
    [ -f "$R" ] && echo "--- k=$K ---" && grep -E "steer0.5|exact" "$R" | head -6
done
echo "curve reading: exact vs K = how many intervened activations the"
echo "minimal-pair edit needs (population level); per-instance minimal sets"
echo "come from prune_spec.py --effector steer (P-O intervention version)."
