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

# P-D: "how many features does it take to COMMAND an edit?"
#
# SCOPE (corrected 2026-07-15 — read this before reading the numbers):
# This curve does NOT answer AxBench's detection 0.695. That number is about
# concept->latent lookup, which this pipeline never performs: our conditioning
# is top-k of (z_tgt - z_src), an instance-level encoding of an observed
# difference. The answer to AxBench's detection axis is P-B plus a scoped
# claim (see PAPER_OUTLINE section 8), not this sweep.
#
# What this curve DOES answer is "why k=32?" — currently an arbitrary-looking
# default. It measures how many INSTANCE-LEVEL features the editor needs, and
# folds three scattered results onto one axis: P-A (k selection), M0 (narrowing
# refuted: k8/k16/k32 FRR 0.7753/0.7895/0.8169 with exact 0.060/0.126/0.210),
# and the k=1 endpoint we never measured.
#
# The separate, sharper contrast — instance-level encoding vs concept-level
# identification — is P-B (FRC phenomenon features collapse editing ~10x) and
# lives on a different axis from this one. Do not conflate them: a small-k
# point is still instance-level.
#
# exact is FREE: run_paper_todo.sh already swept --k-grid 1,2,4,8,16,32,64 over
# the original block. Only the FRR arm costs judge calls, and gold is cached.
#
# Usage: JUDGE=openai:gpt-4o qsub -v JUDGE run_kcurve.sh   (or bash, CPU+API only)
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
KS=${KS:-1 2 4 8 16 32 64}
JUDGE=${JUDGE:-openai:gpt-4o}
TAG=$(echo "$JUDGE" | tr ':/' '__')
FRR=runs/frr_final/$TAG
GOLD=$FRR/gold.jsonl
REC=$V6/ksweep500/records.jsonl
mkdir -p "$FRR" runs/tables

echo "==================== WHAT IS ALREADY ON DISK ===================="
python scripts/check_kcurve_data.py --records "$REC"

echo
echo "==================== EXACT / SARI / sim BY k (free) ===================="
CAND=()
for k in $KS; do CAND+=(--cand "k$k=$REC:k$k"); done
python scripts/score_edit_metrics.py \
    --llm2vec-dir "$LLM2VEC" "${CAND[@]}" \
    --out runs/tables/kcurve_exact

echo
echo "==================== FRR BY k (judge: $JUDGE) ===================="
# M0 already judged k8/k16/k32 under gemma; this redoes the whole arm under the
# PRIMARY judge so every point on the curve is comparable, and extends it down
# to k=1 (the AxBench regime) and up to k=64.
for k in $KS; do
    if [ ! -f "$FRR/.done.kcurve_k$k" ]; then
        echo "---- k=$k"
        python scripts/judge_feature_realization.py \
            --records "$REC" --mode "k$k" --condition true \
            --gold-cache "$GOLD" --judge "$JUDGE" \
            --n-ops-ref "$REC" \
            --out "$FRR/kcurve_k$k.jsonl" --device cuda
        touch "$FRR/.done.kcurve_k$k"
    fi
done

FR=()
for k in $KS; do FR+=(--frr "k$k=$FRR/kcurve_k$k.jsonl"); done
python scripts/frr_per_feature.py "${FR[@]}" \
    --out runs/tables/kcurve_frr_$TAG

echo
echo "==================== PAIRED TESTS ALONG THE CURVE ===================="
# is k=1 (the AxBench regime) significantly worse than k=32, on the SAME pairs?
python scripts/frr_paired_test.py --label "kcurve_$TAG" "${FR[@]}" \
    --out runs/tables/kcurve_paired_$TAG

echo
echo "==================== K-CURVE DONE ===================="
echo "Reading: pair runs/tables/kcurve_exact.md with kcurve_frr_$TAG.md."
echo "  This answers 'why k=32?', NOT AxBench's detection 0.695 (that one is"
echo "  concept->latent lookup, which we never do — see PAPER_OUTLINE section 8)."
echo "  Expected, from M0's k8/k16/k32 trend: exact and FRR both fall as k -> 1,"
echo "  which makes k=32 a measured choice rather than a default, and gives the"
echo "  paper the number of instance-level features editing actually requires."
echo "  If instead k=1 works nearly as well as k=32, then k=32 is unjustified,"
echo "  the conditioning is doing less than we claim, and the paper must say so."
