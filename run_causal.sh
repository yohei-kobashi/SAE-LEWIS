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

# THE CAUSAL AXIS (2026-07-16). The paper's claim moves to: the identified
# SAE activations causally encode the linguistic phenomenon in the LM.
#
# Why EF cannot carry that claim: it never modifies an activation. It takes
# W_dec directions as extra INPUT tokens at layer 0, RMS-renormalized to the
# embedding scale. That is P(edit | Z=z) — conditioning — not P(Y | do(Z=z)).
# Detection alone cannot carry it either: a feature may fire on a confound
# (a word, not the phenomenon), and intervention is what breaks the confound.
#
# So everything here is an intervention, and nothing here is trained:
#
#   P-I  clamp-readout : clamp SAE features at layer 12 of gemma-2-2b (the
#                        model the SAE was TRAINED on — no PT/IT confound),
#                        then read the edit from the LM's OWN head, teacher-
#                        forced. Delta_i = log p_int(x_i) - log p_recon(x_i).
#                        No rate head, no Q head, no training. "Your probe
#                        decodes what you injected" cannot be said of the LM's
#                        own head — the one form the amnesic-probing critique
#                        does not reach. Fills the empty 2x2 cell
#                        (intervention x discrete edits).
#   B1   clamp-rewrite : the same intervention, LM regenerates freely.
#                        Already run: exact 0.1743.
#
# BEST-OF-BOTH, INSIDE THE INTERVENTION FAMILY. The routed system's gain came
# from combining an editor with an intervention, using the editor's own fired-
# edit count as the switch. The same switch exists here without an editor: the
# READOUT's own fire count (how many positions the intervention objected to).
# Small, local objection -> take the readout's surgical edit. Broad objection
# -> take B1's regeneration. Still all intervention, still zero training, so
# the causal claim survives intact.
V6=./runs/prod_gemma_v6
OUT=$V6/clamp_readout500

# ---- P-I: the causal readout, with its controls -------------------------
if [ ! -f "$OUT/report.md" ]; then
    python scripts/eval_clamp_readout.py \
        --output-dir "$OUT" \
        --clamp-values 5,10,20 \
        --delta-thr -1.0 \
        --conditions true,empty,random \
        --sample-size 500 --device cuda
fi

echo
echo "==================== DELTA THRESHOLD SWEEP ===================="
# delta-thr is the only free knob. It is NOT fit on labels — it is the bar
# for "the intervention objects to this token". Sweep it so the operating
# point is reported, not tuned silently.
for THR in -0.5 -2.0 -4.0; do
    D="$OUT/thr${THR}"
    if [ ! -f "$D/report.md" ]; then
        python scripts/eval_clamp_readout.py \
            --output-dir "$D" --clamp-values 10 --delta-thr "$THR" \
            --conditions true,empty,random \
            --sample-size 500 --device cuda
    fi
done

echo
echo "==================== CAUSAL AXIS DONE ===================="
echo "Read $OUT/report.md in this order — the claim lives or dies here:"
echo
echo "1. empty MUST be exact~0 / copy 1.00 / fires 0.00. Delta is identically"
echo "   zero with nothing set, so this is structural. If it is not, the"
echo "   recon baseline is not cancelling and every other number is suspect."
echo "2. true vs random is THE causal test. Same count, same magnitudes, only"
echo "   the feature IDENTITIES differ. true >> random => the identified"
echo "   activations causally carry the phenomenon. true ~= random => they do"
echo "   not, and that is a publishable negative result, not a failure."
echo "3. true vs B1's 0.1743 (clamp + free regeneration). If the readout wins,"
echo "   previous work missed the causal effect because free generation is too"
echo "   blunt a readout, not because the effect is absent."
echo
echo "Then, if the readout has signal, route it against B1 on the readout's"
echo "own fire count (scripts/analyze_router.py, count-rule) — best-of-both"
echo "with no editor and no training anywhere."
