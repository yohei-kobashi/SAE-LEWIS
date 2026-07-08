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

# v6 PILOT (README §13.6 follow-up): does content-diverse training data
# move the editor's OOD fill accuracy?
#   B  new word-class families (FUNCSWAP/MORPH) + family-priority pick
#   C  natural edit pairs (PAWS paraphrases, WikiAtomicEdits ins/del)
# All new records land in the SHARED training cache under shard-v6*-;
# the control run excludes them with the load-time family filter, so the
# two 30k-step editors differ in DATA ONLY. Verdict = gold-template
# probe (fill top-1 / true−empty gap) treatment vs control. Only if the
# probe moves do we pay for the full v6 generation + 100k retrain.
# Every phase has a done-marker — resubmission-safe.
V4=./runs/prod_gemma_v4
V5=./runs/prod_gemma_v5
PILOT=$V5/pilot_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
UNIGRAM=$V4/corruption/unigram.json
BLOCKLIST=runs/blocklist/blocklist.npy
NEW_FAMILIES="FUNCSWAP,MORPH,PAWS,COEDIT"
mkdir -p "$PILOT"

# ---- Phase 0: B-pilot top-up (~30k records, transform-heavy) -------------- #
if [ ! -f "$PILOT/topup.done" ]; then
    SEEN=$(python -c "import json;print(json.load(open('$V4/corruption/meta_v5topup.json'))['sentences_seen'])")
    TOPUP=$V4/corruption_v6pilot
    mkdir -p "$TOPUP"
    cp -n "$UNIGRAM" "$TOPUP/" 2>/dev/null || true

    WORKERS=6 \
    OUT_DIR=$TOPUP \
    LLM2VEC_DIR=$LLM2VEC \
    CORRUPTION_SAMPLES=30000 CORRUPTION_SHARD=2000 \
    BLOCKLIST=$BLOCKLIST \
    TRANSFORM_PROB=0.6 \
    TRANSFORM_COMPOSE_PROB=0.15 \
    FAMILY_PRIORITY_PICK=1 \
    SKIP_SENTENCES=$SEEN \
    bash scripts/corruption_parallel.sh

    for f in "$TOPUP"/shard-w*.jsonl.gz; do
        [ -e "$f" ] || continue
        mv "$f" "$V4/corruption/shard-v6b-$(basename "${f#*shard-}")"
    done
    cp "$TOPUP/meta.json" "$V4/corruption/meta_v6pilot_topup.json"
    touch "$PILOT/topup.done"
fi

# ---- Phase 1: C ingest (PAWS + WikiAtomicEdits) ---------------------------- #
if [ ! -f "$PILOT/ingest.done" ]; then
    for spec in "paws:12000" "coedit:15000"; do
        SRC=${spec%%:*}; NPAIRS=${spec##*:}
        DIR=$V4/natural_edits_$SRC
        if [ ! -f "$DIR/meta.json" ]; then
            python scripts/ingest_edit_pairs.py \
                --source "$SRC" --max-pairs "$NPAIRS" \
                --llm2vec-dir "$LLM2VEC" \
                --unigram-cache "$UNIGRAM" \
                --blocklist "$BLOCKLIST" \
                --out-dir "$DIR" \
                --device cuda
        fi
        for f in "$DIR"/shard-*.jsonl.gz; do
            [ -e "$f" ] || continue
            mv "$f" "$V4/corruption/shard-v6c-$SRC-$(basename "${f#*shard-}")"
        done
        cp "$DIR/meta.json" "$V4/corruption/meta_v6c_$SRC.json"
    done
    touch "$PILOT/ingest.done"
fi

# ---- Phase 2: paired 30k-step editors (control excludes new families) ----- #
train_editor () {  # $1 = output dir, $2.. = extra args
    local OUT=$1; shift
    python train_editor_phaseA.py \
        --corruption-dir "$V5/corruption" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --max-steps 30000 \
        --warmup-steps 500 \
        --proj-a-freeze-steps 1000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --batch-size 8 \
        --num-workers 4 \
        --save-steps 2000 \
        --logging-steps 50 \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --eval-steps 2000 --dev-batches 384 \
        --device cuda --seed 42 \
        "$@"
}

if [ ! -f "$PILOT/editor_control/editor-final.pt" ]; then
    train_editor "$PILOT/editor_control" --exclude-families "$NEW_FAMILIES"
fi
if [ ! -f "$PILOT/editor_treated/editor-final.pt" ]; then
    train_editor "$PILOT/editor_treated"
fi

# ---- Phase 3: gold-template probes (the verdict) --------------------------- #
for arm in control treated; do
    python scripts/lingualens_gold_template_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editor-ckpt "$PILOT/editor_$arm/editor-final.pt" \
        --output-dir "$PILOT/probe_$arm" \
        --k-amp 64 --k-sup 64 --sample-size 200 --device cuda
done

echo "==================== PILOT VERDICT ===================="
for arm in control treated; do
    echo "---- $arm ----"
    sed -n '/## Fill accuracy/,/^$/p;/## Multi-site/,/^$/p' \
        "$PILOT/probe_$arm/probe_report.md" || true
done
