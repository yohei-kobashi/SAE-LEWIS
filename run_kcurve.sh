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
# One curve that answers every single-latent objection at once, on our own
# model and our own SAE, by measurement instead of argument:
#   AxBench   conditions on 1 latent per CONCEPT      -> detection 0.695 (11/12)
#   LinguaLens conditions on 3 base vectors/phenomenon -> intervention
#   our P-B   conditions on FRC phenomenon features    -> editing collapses ~10x
#   this work conditions on k=32 per INSTANCE          -> exact 0.2237
# If exact and FRR both collapse as k -> 1, then the single-latent regime those
# papers measure is demonstrably not the regime this system operates in — which
# is currently only a structural argument in the Related Work.
#
# It also unifies results we already have: P-A (k selection), P-B (phenomenon
# features), M0 (narrowing hypothesis refuted) all become points on this curve.
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
echo "  The claim to check: exact AND FRR both collapse as k -> 1. If so, the"
echo "  single-latent protocol AxBench uses (1 latent/concept, detection 0.695)"
echo "  and LinguaLens's 3-base-vector intervention sit in a regime where OUR"
echo "  editor also fails — so their negative/positive results about few-feature"
echo "  conditioning do not transfer to the k=32 set this system needs. That"
echo "  turns a structural argument in Related Work into a measurement."
echo "  If instead k=1 works nearly as well as k=32, the k=32 default is"
echo "  unjustified and the paper must say so."
