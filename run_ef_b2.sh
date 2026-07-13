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

# B2 baseline (PAPER_OUTLINE §5, claim C2): can the SAE feature deltas be
# replaced by TEXT? The same conditioning the EF model gets (local scope,
# blocklist, diff k64/64, same 500-pair sample) is rendered as natural
# language via Neuronpedia auto-interp labels and handed to
# google/gemma-2-2b-it — the instruction-tuned sibling of the frozen
# backbone EF is built on (matched capacity). n_desc sweep {8,16} on
# true; empty/random controls at 8. ~1-2h; resumes per pair.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
PIPE=$V6/eval_lingualens_final/records.jsonl
EXPL=runs/np_explanations/gemma-2-2b_12-res-16k.json
OUT=$V6/prompt_baseline500

if [ ! -f "$EXPL" ]; then
    python scripts/fetch_sae_explanations.py --out "$EXPL"
fi

if [ ! -f "$OUT/report.md" ]; then
    python scripts/eval_prompt_baseline.py \
        --llm2vec-dir "$LLM2VEC" \
        --explanations "$EXPL" \
        --blocklist "$BLOCKLIST" \
        --output-dir "$OUT" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --n-desc-list 8,16 \
        --device cuda
fi

echo "==================== B2 DONE ===================="
cat "$OUT/report.md"
echo
echo "-------- matched-pair vs pipeline (+ 300-pair holdout) --------"
python scripts/compare_ef_pipeline.py \
    --ef "$OUT/records.jsonl" --pipeline "$PIPE"
python scripts/compare_ef_pipeline.py \
    --ef "$OUT/records.jsonl" --pipeline "$PIPE" \
    --exclude "$V6/editflow_s3/probe/records.jsonl"
echo
echo "Reading (claim C2 — same 500 pairs as S4/M1):"
echo "  EF champion S3 thr0.1 = exact 0.1904 / sim 0.6192; thr0.5 ="
echo "  0.1683/0.6533; pipeline = 0.1102/0.6681; input-copy sim 0.6116."
echo "  - B2 exact << EF exact -> C2 holds strongly: feature-space"
echo "    conditioning is NOT reducible to text rendering at matched"
echo "    backbone capacity."
echo "  - B2 exact ~ EF exact -> C2 weakens to controllability/premise"
echo "    grounds: check empty copy (must be ~1.00) and random copy —"
echo "    prompt models usually leak premise under arbitrary asks."
echo "  - B2 exact > EF exact -> paper framing needs rework (SAE"
echo "    conditioning value unproven at this scale) — escalate before"
echo "    writing further."
echo "  Caveats to record either way: auto-interp label quality is a"
echo "  shared confounder; B2 sees top-8/16 features vs EF's k=64."
