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

# P-J: WHICH activations to intervene on — LinguaLens's selector vs AxBench's,
# swept over r, through the causal readout.
#
# The spec here is TARGET-FREE (feature-mode pure): the intervened features
# come from phenomenon-level identification computed on held-out pairs, and
# their magnitudes from the SOURCE's own activations. Nothing consults the
# target. This is the deployment-honest configuration, and it directly
# answers the section-8 limitation (specification origin) — for the
# INTERVENTION route, unlike the EF editor where P-B's train/test mismatch
# made the same substitution uninterpretable. The readout trains nothing, so
# that critique cannot apply here: if phenomenon-level features work through
# intervention but not through the trained editor, the features carry the
# phenomenon and the editor's failure was a conditioning-distribution issue.
#
#   selector             r grid                 canonical point
#   FRC   (LinguaLens)   3, 8, 16, 32, 64       r=3  (their protocol)
#   AUROC (AxBench)      1, 3, 8, 16, 32, 64    r=1  (their protocol: THE
#                                                     highest-AUROC latent —
#                                                     top-k>1 is undefined in
#                                                     their paper; the ranking
#                                                     extends to r>1 naturally,
#                                                     and that extension is ours)
#
# Reference points, same readout, from run_b1_improve.sh:
#   instance-level delta spec (k=32, target-peeking) = the upper reference
#   B1 clamp10 0.1743 / B3 steer0.5 0.2337          = the bars
#
# IV/SC default to delta+local (a priori winner); once run_b1_improve.sh has
# a measured winner, re-run with IV=... SC=... — everything is resumable.
V6=./runs/prod_gemma_v6
IV=${IV:-delta}
SC=${SC:-local}
R=$V6/clamp_readout500/select_${IV}_${SC}
EXPL=runs/np_explanations/gemma-2-2b_12-res-16k.json
VALS=$([ "$IV" = clamp ] && echo "10" || echo "0.5,1")

# ---- 1. selections (CPU-light; encodings cached and shared per selector) --
for r in 3 8 16 32 64; do
    if [ ! -f "runs/frc/identified_l12_16k_r$r.json" ]; then
        python scripts/identify_features_frc.py \
            --out "runs/frc/identified_l12_16k_r$r.json" --top-r "$r" \
            --explanations "$EXPL" --device cuda
    fi
done
for r in 1 3 8 16 32 64; do
    if [ ! -f "runs/auroc/identified_l12_16k_r$r.json" ]; then
        python scripts/select_features_auroc.py \
            --out "runs/auroc/identified_l12_16k_r$r.json" --top-r "$r" \
            --explanations "$EXPL" --device cuda
    fi
done

# ---- 2. the readout on each selection (pure = target-free) ---------------
ro () {  # tag json
    if [ ! -f "$R/$1/report.md" ]; then
        echo "-------- $1"
        python scripts/eval_clamp_readout.py \
            --output-dir "$R/$1" \
            --intervention "$IV" --scope "$SC" \
            --feature-sets "$2" --feature-mode pure \
            --clamp-values "$VALS" --delta-thr -1.0 \
            --steps 8 --max-ops-per-step 4 \
            --conditions true,empty,random \
            --sample-size 500 --device cuda
    fi
}
for r in 3 8 16 32 64; do
    ro "frc_r$r"   "runs/frc/identified_l12_16k_r$r.json"
done
for r in 1 3 8 16 32 64; do
    ro "auroc_r$r" "runs/auroc/identified_l12_16k_r$r.json"
done

# ---- 3. one intersect cell: narrowing WITH target-peeking, matched r -----
# separates "phenomenon-level spec costs X" (pure vs instance) from
# "narrowing to the identified set costs Y" (intersect vs instance).
if [ ! -f "$R/frc_r32_intersect/report.md" ]; then
    python scripts/eval_clamp_readout.py \
        --output-dir "$R/frc_r32_intersect" \
        --intervention "$IV" --scope "$SC" \
        --feature-sets "runs/frc/identified_l12_16k_r32.json" \
        --feature-mode intersect \
        --clamp-values "$VALS" --delta-thr -1.0 \
        --steps 8 --max-ops-per-step 4 \
        --conditions true,empty,random \
        --sample-size 500 --device cuda
fi

echo
echo "==================== P-J SELECTION SWEEP DONE ===================="
echo "Collect exact from $R/{frc_r*,auroc_r*}/report.md into two curves"
echo "(exact vs r, one per selector). How to read them:"
echo
echo "  * auroc_r1 is AxBench's protocol verbatim, through our readout. If it"
echo "    is ~0 while r=32 works, the single-latent regime is measured as"
echo "    insufficient ON THE CAUSAL ROUTE — not argued, measured."
echo "  * frc_r3 is LinguaLens's protocol verbatim. Same reading."
echo "  * FRC vs AUROC at matched r isolates the selector (causal vs"
echo "    discriminative) with everything else held fixed."
echo "  * every cell is TARGET-FREE: clearing B3 (0.2337) here means beating"
echo "    the steering family without consulting the target — a stronger"
echo "    result than the instance-level run, which peeks."
echo "  * true vs random within each cell stays the causal control; empty"
echo "    must stay exact~0/copy 1.00 everywhere."
echo "  * frc_r32_intersect vs frc_r32 (pure) vs the instance-level run"
echo "    decomposes: spec-origin cost vs narrowing cost."
echo
echo "FRR afterwards (records are judge-ready; mode key = ${IV}<value>):"
echo "  python scripts/judge_feature_realization.py \\"
echo "      --records $R/<cell>/records.jsonl --mode ${IV}1 --condition true \\"
echo "      --gold-cache runs/frr_final/openai_gpt-4o/gold.jsonl \\"
echo "      --judge openai:gpt-4o --out runs/frr_final/openai_gpt-4o/<cell>.jsonl"
echo "  (gold cache is shared — only the system side costs API calls)"
