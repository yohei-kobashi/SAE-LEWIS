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

# FINAL FRR at the FINAL operating points, per-phenomenon (the earlier
# FRR round judged ef64@thr0.1 and steer@alpha1 — champion-mismatched).
# Judges over all 997 pairs (gold cache auto-extends to the fresh 500):
#   ef32 (k=32)        true            — the router head
#   steer0.5           true + random   — the router fallback (random
#                                        exists only on the fresh ~500;
#                                        judged where present)
#   routed             true            — materialized records (count-rule
#                                        T=1); its random control is
#                                        structural (EF path -> copy),
#                                        cite ef64_rnd 0.095 as the floor
# Then aggregates per-phenomenon FRR (+net where random exists).
# JUDGE env: default local gemma-2-9b-it; set JUDGE=openai:gpt-4o (and
# export OPENAI_API_KEY) for the paper's final judging — outputs are
# per-judge directories. ~3000 judge calls (~2h local). Resumes per pair.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
JUDGE=${JUDGE:-hf:google/gemma-2-9b-it}
TAG=$(echo "$JUDGE" | tr ':/' '__')
FRR=runs/frr_final/$TAG
GOLD=$FRR/gold.jsonl
ROUTED=$V6/routed_system/records.jsonl
mkdir -p "$FRR"

# materialize the routed system (CPU, idempotent)
python scripts/score_edit_metrics.py \
    --llm2vec-dir "$LLM2VEC" \
    --cand ef32="$V6/ksweep500/records.jsonl":k32 \
    --cand steer="$V6/steer_baseline500/records.jsonl":steer0.5 \
    --router-head ef32 --router-fallback steer --router-T 1 \
    --emit-routed-records "$ROUTED" \
    --out runs/tables/main_metrics_997

frr () {  # label records mode condition
    if [ ! -f "$FRR/.done.$1" ]; then
        python scripts/judge_feature_realization.py \
            --records "$2" --mode "$3" --condition "$4" \
            --gold-cache "$GOLD" --judge "$JUDGE" \
            --n-ops-ref "$V6/ksweep500/records.jsonl" \
            --out "$FRR/$1.jsonl" --device cuda
        touch "$FRR/.done.$1"
    fi
}
frr routed    "$ROUTED"                                "routed"   true
frr ef32      "$V6/ksweep500/records.jsonl"            "k32"      true
frr steer     "$V6/steer_baseline500/records.jsonl"    "steer0.5" true
frr steer_rnd "$V6/steer_baseline500/records.jsonl"    "steer0.5" random

echo "==================== FRR FINAL DONE ===================="
python scripts/frr_per_feature.py \
    --frr routed="$FRR/routed.jsonl" \
    --frr ef32="$FRR/ef32.jsonl" \
    --frr steer="$FRR/steer.jsonl" \
    --frr steer_rnd="$FRR/steer_rnd.jsonl" \
    --out runs/tables/frr_per_feature_$TAG
echo
echo "Reading: pair this with main_metrics_997_per_feature.csv —"
echo "  features where exact=0 for everyone but FRR is high are"
echo "  'directionally realizable, not exactly editable' (e.g. expect"
echo "  figurative language via steer); features where FRR is ALSO low"
echo "  are genuinely unreachable. routed's random control is structural"
echo "  (empty/random -> EF path -> copy; cite ef64_rnd FRR 0.095 as"
echo "  the EF-side floor from the earlier round)."
