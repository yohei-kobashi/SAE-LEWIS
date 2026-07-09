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

# v6 STAGE 1 (§6.2.11 / §13.6): full content-diverse generation + 100k
# retrains + eval suite. Pilot verdict: treated > control under
# training-parity conditioning; e2e with local extraction + lens λ=1
# already beats the input-copy baseline (sim 0.6447 > 0.6072). This run
# scales the data and retrains both modules.
#   Phase 0  +200k B top-up (FUNCSWAP/MORPH + priority pick), dev supply
#   Phase 1  full C ingest (PAWS ~21k + CoEdIT pairs), dedup vs pilot
#   Phase 2  editor 100k + tagger 100k under runs/prod_gemma_v6
#   Phase 3  eval suite (per-family, ceiling, probe, e2e sample 500)
# Total ≈ 14h — RESUBMIT this job until it prints "ALL DONE"; every
# phase is guarded by a done-marker / output check.
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
mkdir -p "$V6"

# unigram resolver (plain loop — ls|head dies under set -e -o pipefail)
UNIGRAM=""
for cand in "$V4/corruption/unigram.json" \
            "$V4"/corruption_v6pilot/worker-*/unigram.json \
            "$V4"/corruption_v6full/worker-*/unigram.json \
            "$V4"/corruption_v5topup/worker-*/unigram.json; do
    if [ -f "$cand" ]; then UNIGRAM=$cand; break; fi
done
if [ -z "$UNIGRAM" ]; then
    echo "[v6] no unigram.json found under $V4" >&2
    exit 1
fi
echo "[v6] unigram baseline: $UNIGRAM"

# ---- Phase 0: +200k B top-up (same recipe as the pilot) ------------------- #
if [ ! -f "$V6/data_topup.done" ]; then
    SEEN=$(python -c "import json;print(json.load(open('$V4/corruption/meta_v6pilot_topup.json'))['sentences_seen'])")
    TOPUP=$V4/corruption_v6full
    mkdir -p "$TOPUP"
    cp -n "$UNIGRAM" "$TOPUP/unigram.json" 2>/dev/null || true

    WORKERS=6 \
    OUT_DIR=$TOPUP \
    LLM2VEC_DIR=$LLM2VEC \
    CORRUPTION_SAMPLES=200000 CORRUPTION_SHARD=2000 \
    BLOCKLIST=$BLOCKLIST \
    TRANSFORM_PROB=0.6 \
    TRANSFORM_COMPOSE_PROB=0.15 \
    FAMILY_PRIORITY_PICK=1 \
    SKIP_SENTENCES=$SEEN \
    bash scripts/corruption_parallel.sh

    # Dev supply: w0 → seldev, w1+w2 → reporting dev (2k each). shard-v6*
    # sorts after shard-v5-* and before shard-w*, so raise eval
    # --max-samples to reach them (Phase 3 uses 18000).
    for i in 0 1 2; do
        new=""
        for f in "$TOPUP"/shard-w$i-*.jsonl.gz; do
            [ -e "$f" ] && new=$f && break
        done
        if [ -z "$new" ]; then
            echo "[v6] no shard for worker $i — top-up incomplete?" >&2
            exit 1
        fi
        base=$(basename "$new")
        if [ "$i" = 0 ]; then
            mv "$new" "$V4/corruption_seldev/shard-v6f-${base#shard-}"
        else
            mv "$new" "$V4/corruption_dev/shard-v6f-${base#shard-}"
        fi
    done
    for f in "$TOPUP"/shard-w*.jsonl.gz; do
        [ -e "$f" ] || continue
        mv "$f" "$V4/corruption/shard-v6f-$(basename "${f#*shard-}")"
    done
    cp "$TOPUP/meta.json" "$V4/corruption/meta_v6full_topup.json"
    touch "$V6/data_topup.done"
fi

# ---- Phase 1: full C ingest (dedup vs the pilot's merged shards) ---------- #
if [ ! -f "$V6/data_ingest.done" ]; then
    for spec in "paws:25000" "coedit:45000"; do
        SRC=${spec%%:*}; NPAIRS=${spec##*:}
        DIR=$V4/natural_edits_${SRC}_full
        if [ ! -f "$DIR/meta.json" ]; then
            python scripts/ingest_edit_pairs.py \
                --source "$SRC" --max-pairs "$NPAIRS" \
                --skip-source-ids-glob "$V4/corruption/shard-v6c-$SRC-*.jsonl.gz" \
                --llm2vec-dir "$LLM2VEC" \
                --unigram-cache "$UNIGRAM" \
                --blocklist "$BLOCKLIST" \
                --out-dir "$DIR" \
                --device cuda
        fi
        # Dev supply: first shard → reporting dev (+ second coedit shard
        # → seldev so the checkpoint monitor sees natural edits).
        n=0
        for f in "$DIR"/shard-*.jsonl.gz; do
            [ -e "$f" ] || continue
            base=$(basename "$f")
            if [ "$n" = 0 ]; then
                mv "$f" "$V4/corruption_dev/shard-v6d-$SRC-${base#shard-}"
            elif [ "$n" = 1 ] && [ "$SRC" = "coedit" ]; then
                mv "$f" "$V4/corruption_seldev/shard-v6d-$SRC-${base#shard-}"
            else
                mv "$f" "$V4/corruption/shard-v6d-$SRC-${base#shard-}"
            fi
            n=$((n + 1))
        done
        cp "$DIR/meta.json" "$V4/corruption/meta_v6d_$SRC.json"
    done
    touch "$V6/data_ingest.done"
fi

# ---- Phase 2: v6 training (editor 100k + tagger 100k) --------------------- #
[ -e "$V6/corruption" ] || ln -s ../prod_gemma_v4/corruption "$V6/corruption"

RUN_DIR=$V6 \
EDITOR_STEPS=100000 TAGGER_STEPS=100000 \
DEV_CORRUPTION_DIR=$V4/corruption_seldev \
DEV_BATCHES=768 \
K_TOP=32 K_DRAW=log:1-32 \
LLM2VEC_DIR=$LLM2VEC \
SIMCSE_DIR=$LLM2VEC \
bash scripts/run_production.sh

# ---- Phase 3: eval suite --------------------------------------------------- #
# max-samples 18000 spans dev's shard order: v5 (10k) + v6d natural-edit
# (4k) + v6f new-family (4k), so the per-family table covers everything.
if [ ! -f "$V6/eval_dev_k64_fam/eval_metrics.json" ]; then
    python eval_tagger_editor.py \
        --corruption-dir "$V4/corruption_dev" \
        --llm2vec-dir "$LLM2VEC" \
        --tagger-ckpt "$V6/tagger/tagger-final.pt" \
        --editor-ckpt "$V6/editor/editor-final.pt" \
        --output-dir "$V6/eval_dev_k64_fam" \
        --k-top 64 --k-amp 64 --k-sup 64 --ins-threshold 0.9 \
        --per-family --max-samples 18000 \
        --device cuda
fi

if [ ! -f "$V6/ceiling_k64/ceiling_report.md" ]; then
    python scripts/measure_editor_ceiling.py \
        --corruption-dir "$V4/corruption_dev" \
        --llm2vec-dir "$LLM2VEC" \
        --editor-ckpt "$V6/editor/editor-final.pt" \
        --output-dir "$V6/ceiling_k64" \
        --k-top 64 --k-amp 64 --k-sup 64 \
        --max-samples 4000 --device cuda
fi

if [ ! -f "$V6/probe_local/probe_report.md" ]; then
    python scripts/lingualens_gold_template_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editor-ckpt "$V6/editor/editor-final.pt" \
        --output-dir "$V6/probe_local" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --modes parallel --steer-lambdas 0.5,1,2 \
        --device cuda
fi

# Paper numbers: 500 pairs, operating point local + λ=1.
if [ ! -f "$V6/eval_lingualens_final/report.md" ]; then
    python eval_lingualens.py \
        --llm2vec-dir "$LLM2VEC" \
        --tagger-ckpt "$V6/tagger/tagger-final.pt" \
        --editor-ckpt "$V6/editor/editor-final.pt" \
        --output-dir "$V6/eval_lingualens_final" \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --steer-lambda 1 \
        --k-amp 64 --k-sup 64 --ins-threshold 0.9 \
        --sample-size 500 --refine-passes 3 --refine-recompute \
        --fluency-gate 0.5 --dump-details --device cuda
fi

echo "==================== ALL DONE ===================="
echo "results:"
echo "  $V6/eval_dev_k64_fam/eval_report.md"
echo "  $V6/ceiling_k64/ceiling_report.md"
echo "  $V6/probe_local/probe_report.md"
echo "  $V6/eval_lingualens_final/report.md"
