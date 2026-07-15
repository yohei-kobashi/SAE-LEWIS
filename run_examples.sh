#!/bin/bash
# Qualitative examples for the paper (CPU, seconds — no qsub, no judging:
# it only joins records.jsonl outputs with the FRR verdicts already on disk).
#
# Emits, per LinguaLens feature, one success / near-miss / failure case with
# every system's output on the SAME pair, word-diffed against the source.
# The three categories ARE the residual-frontier decomposition of section 6c,
# so each qualitative example backs a specific quantitative claim:
#   success -> the win;  near -> "directionally realizable, not exactly
#   editable" (metaphor/personification);  fail -> genuinely unreachable.
#
# JUDGE picks which judge's realized verdicts label the examples. Default is
# the primary (GPT-4o); the near/fail split is judge-dependent for a few
# features (extraposition splits 2:1 — see section 6c), so re-run with
# JUDGE=hf_google_gemma-2-9b-it to check an example's label before printing it.
set -eo pipefail

V6=./runs/prod_gemma_v6
JUDGE=${JUDGE:-openai_gpt-4o}
FRR=runs/frr_final/$JUDGE
[ -d "$FRR" ] || { echo "no $FRR — run run_frr_rerun.sh first"; exit 1; }
echo "[ex] labelling with judge: $JUDGE"

# --- main table: 4 systems incl. LinguaLens, on the ~499 pairs B1 covers ---
# clamp10 = LinguaLens's own operating point (their paper uses 10).
python scripts/collect_examples.py \
    --sys ours="$V6/routed_system/records.jsonl,routed" \
    --sys ef32="$V6/ksweep500/records.jsonl,k32" \
    --sys lingualens="$V6/clamp_baseline500/records.jsonl,clamp10" \
    --sys steer="$V6/steer_baseline500/records.jsonl,steer0.5" \
    --frr ours="$FRR/routed.jsonl" \
    --frr ef32="$FRR/ef32.jsonl" \
    --frr steer="$FRR/steer.jsonl" \
    --focus ours --per-feature 1 --max-words 40 \
    --out "runs/tables/examples_vs_lingualens_$JUDGE"

echo
# --- wider net: drop LinguaLens, gain the full 997 and every feature ---
python scripts/collect_examples.py \
    --sys ours="$V6/routed_system/records.jsonl,routed" \
    --sys ef32="$V6/ksweep500/records.jsonl,k32" \
    --sys steer="$V6/steer_baseline500/records.jsonl,steer0.5" \
    --frr ours="$FRR/routed.jsonl" \
    --frr ef32="$FRR/ef32.jsonl" \
    --frr steer="$FRR/steer.jsonl" \
    --focus ours --per-feature 2 --max-words 40 \
    --out "runs/tables/examples_997_$JUDGE"

echo
echo "==================== EXAMPLES DONE ===================="
echo "runs/tables/examples_vs_lingualens_$JUDGE.md  <- the paper table"
echo "  (4 systems on the ~499 pairs B1 covers; ours-vs-LinguaLens contrast"
echo "   is ranked first within each feature)"
echo "runs/tables/examples_997_$JUDGE.md            <- wider coverage"
echo "  (no LinguaLens, so all 997 pairs and 2 examples per category)"
echo "Both also emit .jsonl with the full untruncated texts."
echo
echo "Reading: 'near' cases are the paper's most interesting figure —"
echo "the edit moved the feature the right way but missed the target"
echo "string. Check a few 'success' rows where ours picked steer vs ef32"
echo "(the 'via' tag) against section 6c's per-feature split: EF should own"
echo "morphology, steer should own clause structure."
