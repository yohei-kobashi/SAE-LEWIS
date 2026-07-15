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

# P-B2: does the CONDITIONING SIGNAL's granularity matter, or just its size?
#
# P-B conditioned the editor on LinguaLens's FRC-identified phenomenon features
# (top-16) and exact collapsed ~10x. But that result confounds two things:
#   (i)  the selection is PHENOMENON-LEVEL — aggregated over a corpus
#   (ii) FRC is the selector
# Our claim is (i). A reviewer will press (ii): "FRC is a causal criterion
# built for interpretability; a discriminative selector might do better —
# AxBench got +32% detection from AUROC selection. Did you try it?" AxBench
# answers that for STEERING (detection 0.695->0.917, steering 0.165->0.157).
# Nobody has answered it for editing.
#
# So: run AxBench's own selector (SAE-A: max-pooled activation, AUROC against
# sentence1=1/sentence2=0 labels) on our phenomena, at MATCHED r, and see.
#
# The 2x2 this fills, with P-D covering the instance-level row:
#
#   conditioning          granularity   r=1   r=3   r=16   r=32
#   FRC   (LinguaLens)    phenomenon     .     .    P-B     .
#   AUROC (AxBench SAE-A) phenomenon     .     .     .      .     <- this job
#   tau   (ours)          instance      P-D    -    P-D   0.2237
#
# The decisive cell is AUROC r=32 vs ours at k=32: SAME feature count, only
# the granularity differs. If it still collapses, the culprit is AGGREGATION,
# not count — LinguaLens's EALE averages away the per-instance tau that editing
# needs (their tau_k(s) = a1-a0 IS our z_tgt - z_src, before the mean).
# If AUROC r=32 works, then P-B was about FRC specifically, our story is much
# weaker, and the paper has to say so.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
EXPL=runs/np_explanations/gemma-2-2b_12-res-16k.json
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
RS=${RS:-1 3 16 32}

# ---- selection (encoding cached in runs/auroc/auroc_acts.jsonl, shared) ----
for r in $RS; do
    if [ ! -f "runs/auroc/identified_l12_16k_r$r.json" ]; then
        python scripts/select_features_auroc.py \
            --out "runs/auroc/identified_l12_16k_r$r.json" --top-r "$r" \
            --explanations "$EXPL" --device cuda
    fi
done

# ---- FRC at matched r, so the selector is the ONLY difference --------------
# P-B ran FRC at the default top-r 16 only; without r=1/3/32 we cannot tell a
# selector effect from a count effect.
for r in $RS; do
    if [ ! -f "runs/frc/identified_l12_16k_r$r.json" ]; then
        python scripts/identify_features_frc.py \
            --out "runs/frc/identified_l12_16k_r$r.json" --top-r "$r" \
            --explanations "$EXPL" --device cuda
    fi
done

# ---- condition the editor on each selection -------------------------------
probe () {   # tag feature-sets-json mode
    if [ ! -f "$V6/editflow_s3/$1/probe_report.md" ]; then
        echo "-------- $1"
        python scripts/editflow_probe.py \
            --llm2vec-dir "$LLM2VEC" \
            --editflow-ckpt "$V6/editflow_s3/editflow-final.pt" \
            --output-dir "$V6/editflow_s3/$1" \
            --cond-scope local --blocklist "$BLOCKLIST" \
            --k-amp 64 --k-sup 64 --sample-size 500 \
            --steps 48 --steer-lambda 1 --decode thr0.1 \
            --feature-sets "$2" --feature-mode "$3" \
            --conditions true --device cuda
    fi
}
for r in $RS; do
    probe "probe500_auroc_r${r}_intersect" "runs/auroc/identified_l12_16k_r$r.json" intersect
    probe "probe500_frc_r${r}_intersect"   "runs/frc/identified_l12_16k_r$r.json"   intersect
done

echo
echo "==================== P-B2 DONE ===================="
echo "Collect: for each probe500_{auroc,frc}_r*_intersect, read exact from"
echo "  probe_report.md and put it in the 2x2 against P-D's instance-level row."
echo
echo "Reading:"
echo "  * AUROC r=32 vs ours k=32 is the decisive comparison — same count,"
echo "    only granularity differs. Collapse there => aggregation is the"
echo "    culprit, and the paper's claim ('don't average tau away') holds"
echo "    against a discriminative selector, not just against FRC."
echo "  * AUROC ~= FRC at every r => the selector does not matter, which is"
echo "    the stronger version of P-B: no phenomenon-level selection commands."
echo "  * AUROC >> FRC => P-B was an artifact of FRC and must be rewritten."
echo "  * Check the printed best-latent AUROC first: if it is near AxBench's"
echo "    0.917, the selector demonstrably works and any collapse downstream"
echo "    is not a selection failure. If it is near 0.5, the selection itself"
echo "    failed on our phenomena and the arm proves nothing."
