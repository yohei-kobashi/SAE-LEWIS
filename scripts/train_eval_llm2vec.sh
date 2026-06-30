#!/usr/bin/env bash
# scripts/train_eval_llm2vec.sh
#
# LLM2Vec を学習し、続けて学習効果を 5 項目で自動評価するパイプライン。
#
#   stage 1: train_llm2vec.py   (MNTP, bidir patch, [INS]/[DEL] 追加)
#   stage 2: eval_llm2vec.py    (MNTP loss / causal PPL drift / bidir-vs-causal /
#                                special-token sanity / MTEB STS-B)
#
# 同じ RUN_DIR で再実行すると stage 1 は最新 checkpoint-* から再開
# (train_llm2vec.py が --resume をデフォルト ON にしている)。
# stage 2 (評価) は毎回再実行する — 計算は軽く、報告は上書きされる。
#
# Usage:
#   bash scripts/train_eval_llm2vec.sh
#   RUN_DIR=./runs/llm2vec-exp1 bash scripts/train_eval_llm2vec.sh
#   LLM2VEC_STEPS=2000 bash scripts/train_eval_llm2vec.sh        # smoke
#   FORCE_FRESH=1 bash scripts/train_eval_llm2vec.sh             # 全部やり直し
#   SKIP_TRAIN=1 LLM2VEC_DIR=./runs/prod/llm2vec \
#     bash scripts/train_eval_llm2vec.sh                         # 評価のみ
#
# Environment overrides:
#   RUN_DIR              出力ルート (default: ./runs/llm2vec_eval)
#   LLM2VEC_DIR          学習出力先 (default: $RUN_DIR/llm2vec)
#   EVAL_DIR             評価出力先 (default: $RUN_DIR/eval)
#   DEVICE               cuda | cpu  (default: cuda)
#   SEED                 (default: 42)
#
#   --- training (stage 1) ---
#   LLM                  HF base model id (default: google/gemma-2-2b)
#   DOLMA_MAX_FILES      Dolma shards to download (default: 32)
#   LLM2VEC_STEPS        MNTP steps (default: 10000)
#   LLM2VEC_BATCH        per-device batch (default: 8)
#   LLM2VEC_ACCUM        grad-accum-steps (default: 4)
#   LLM2VEC_LR           learning rate (default: 1e-5)
#   LLM2VEC_WARMUP       warmup steps (default: 1000)
#   LLM2VEC_SAVE_STEPS   checkpoint cadence (default: 1000)
#   NUM_WORKERS          data-loader workers (default: 4)
#   MAX_SEQ_LENGTH       (default: 256)
#   MLM_PROBABILITY      (default: 0.15)
#
#   --- evaluation (stage 2) ---
#   N_SENTENCES          held-out sentences for eval 1+2 (default: 500)
#   N_BIDIR_CAUSAL       sentences for eval 3 (default: 100)
#   POOLING              mean | last | weighted_mean (default: mean)
#   MTEB_TASKS           space-separated task list (default: "STSBenchmark")
#                         empty string disables eval 5
#   MTEB_BATCH_SIZE      (default: 8)
#   SKIP_MTEB            1 → skip eval 5 (default: 0)
#   BASELINE_LLM         override; "none" to skip base-LLM causal-PPL load
#
#   --- pipeline control ---
#   FORCE_FRESH          1 = wipe RUN_DIR + pass --no-resume to training
#   SKIP_TRAIN           1 = 評価のみ。LLM2VEC_DIR を別途指定すれば既存
#                         checkpoint への評価が走る
#
# Per-stage log:  $RUN_DIR/logs/<stage>.log
# Timing summary: $RUN_DIR/timing.tsv
# Final report:   $EVAL_DIR/eval_report.md   (末尾で stdout にも cat される)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RUN_DIR=${RUN_DIR:-"./runs/llm2vec_eval"}
LLM2VEC_DIR=${LLM2VEC_DIR:-"$RUN_DIR/llm2vec"}
EVAL_DIR=${EVAL_DIR:-"$RUN_DIR/eval"}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

LLM=${LLM:-"google/gemma-2-2b"}
DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-32}
LLM2VEC_STEPS=${LLM2VEC_STEPS:-10000}
LLM2VEC_BATCH=${LLM2VEC_BATCH:-8}
LLM2VEC_ACCUM=${LLM2VEC_ACCUM:-4}
LLM2VEC_LR=${LLM2VEC_LR:-1e-5}
LLM2VEC_WARMUP=${LLM2VEC_WARMUP:-1000}
LLM2VEC_SAVE_STEPS=${LLM2VEC_SAVE_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
MLM_PROBABILITY=${MLM_PROBABILITY:-0.15}

N_SENTENCES=${N_SENTENCES:-500}
N_BIDIR_CAUSAL=${N_BIDIR_CAUSAL:-100}
POOLING=${POOLING:-mean}
MTEB_TASKS=${MTEB_TASKS:-"STSBenchmark"}
MTEB_BATCH_SIZE=${MTEB_BATCH_SIZE:-8}
SKIP_MTEB=${SKIP_MTEB:-0}

FORCE_FRESH=${FORCE_FRESH:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}

# FORCE_FRESH=1: rm -rf RUN_DIR + pass --no-resume to the trainer
RESUME_FLAG=""
if [[ "$FORCE_FRESH" == "1" ]]; then
    if [[ "$SKIP_TRAIN" == "1" ]]; then
        echo "[train-eval] FORCE_FRESH=1 と SKIP_TRAIN=1 を同時指定しても" \
             "学習は skip されます。LLM2VEC_DIR の中身は wipe しません。" >&2
    else
        if [[ -d "$RUN_DIR" ]]; then
            echo "[train-eval] FORCE_FRESH=1: removing $RUN_DIR"
            rm -rf "$RUN_DIR"
        fi
        RESUME_FLAG="--no-resume"
    fi
fi

# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
DOLMA_CACHE="$RUN_DIR/dolma_cache"

SUMMARY="$RUN_DIR/timing.tsv"
if [[ ! -f "$SUMMARY" ]]; then
    printf "stage\tstart\tend\telapsed_sec\tstatus\n" > "$SUMMARY"
fi

# --------------------------------------------------------------------------- #
# Helpers (same shape as run_production.sh)
# --------------------------------------------------------------------------- #
banner() {
    printf '\n===============================================================\n'
    printf ' %s\n' "$1"
    printf '===============================================================\n'
}

record() {
    local stage=$1 start=${2:-} end=${3:-} elapsed=${4:-0} status=$5
    printf "%s\t%s\t%s\t%s\t%s\n" "$stage" "$start" "$end" "$elapsed" "$status" \
        >> "$SUMMARY"
}

run_stage() {
    local name=$1; shift
    local log="$LOG_DIR/${name}.log"
    banner "[train-eval] stage: $name"
    echo "[train-eval] cmd: $*" | tee "$log"
    local start
    start=$(date +%s)
    set +e
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    local end
    end=$(date +%s)
    local elapsed=$((end - start))
    if [[ "$rc" -ne 0 ]]; then
        record "$name" "$start" "$end" "$elapsed" "fail(rc=$rc)"
        printf '[train-eval] %-22s FAILED rc=%d  see %s\n' "$name" "$rc" "$log" >&2
        echo "[train-eval] 同じコマンドを再実行すれば最新 checkpoint から続行します" >&2
        exit "$rc"
    fi
    record "$name" "$start" "$end" "$elapsed" "ok"
    printf '[train-eval] %-22s elapsed=%dh%02dm%02ds  log=%s\n' \
        "$name" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) \
        $((elapsed % 60)) "$log"
}

skip_stage() {
    local name=$1 reason=$2
    record "$name" "" "" 0 "skip"
    printf '[train-eval] %-22s SKIPPED  %s\n' "$name" "$reason"
}

# --------------------------------------------------------------------------- #
# Print config
# --------------------------------------------------------------------------- #
banner "[train-eval] config"
cat <<EOF
RUN_DIR              = $RUN_DIR     (resume by default; FORCE_FRESH=$FORCE_FRESH)
LLM2VEC_DIR          = $LLM2VEC_DIR
EVAL_DIR             = $EVAL_DIR
DEVICE               = $DEVICE        SEED = $SEED
LLM                  = $LLM
DOLMA_MAX_FILES      = $DOLMA_MAX_FILES
LLM2VEC_STEPS / BS   = $LLM2VEC_STEPS / $LLM2VEC_BATCH (accum $LLM2VEC_ACCUM, effective $((LLM2VEC_BATCH * LLM2VEC_ACCUM)))
LLM2VEC_LR / WARMUP  = $LLM2VEC_LR / $LLM2VEC_WARMUP
SAVE_STEPS           = $LLM2VEC_SAVE_STEPS
NUM_WORKERS          = $NUM_WORKERS
MAX_SEQ_LENGTH       = $MAX_SEQ_LENGTH
MLM_PROBABILITY      = $MLM_PROBABILITY
N_SENTENCES          = $N_SENTENCES
N_BIDIR_CAUSAL       = $N_BIDIR_CAUSAL
POOLING              = $POOLING
MTEB_TASKS           = $MTEB_TASKS
MTEB_BATCH_SIZE      = $MTEB_BATCH_SIZE
SKIP_MTEB            = $SKIP_MTEB
BASELINE_LLM         = ${BASELINE_LLM:-<llm2vec_meta.json から自動>}
SKIP_TRAIN           = $SKIP_TRAIN
EOF

T0=$(date +%s)

# --------------------------------------------------------------------------- #
# Stage 1: train LLM2Vec (MNTP)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_TRAIN" == "1" ]]; then
    skip_stage "01_train_llm2vec" "SKIP_TRAIN=1; evaluating $LLM2VEC_DIR"
    if [[ ! -f "$LLM2VEC_DIR/llm2vec_meta.json" ]]; then
        echo "[train-eval] SKIP_TRAIN=1 だが $LLM2VEC_DIR/llm2vec_meta.json が存在しません" >&2
        echo "[train-eval] 学習済み checkpoint へ LLM2VEC_DIR を向けてください" >&2
        exit 1
    fi
elif [[ -f "$LLM2VEC_DIR/llm2vec_meta.json" ]]; then
    skip_stage "01_train_llm2vec" "exists at $LLM2VEC_DIR (FORCE_FRESH=1 で再学習)"
else
    run_stage "01_train_llm2vec" \
        python train_llm2vec.py \
            --llm "$LLM" \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --output-dir "$LLM2VEC_DIR" \
            --max-steps "$LLM2VEC_STEPS" \
            --warmup-steps "$LLM2VEC_WARMUP" \
            --learning-rate "$LLM2VEC_LR" \
            --per-device-batch-size "$LLM2VEC_BATCH" \
            --grad-accum-steps "$LLM2VEC_ACCUM" \
            --save-steps "$LLM2VEC_SAVE_STEPS" \
            --logging-steps 50 \
            --max-seq-length "$MAX_SEQ_LENGTH" \
            --mlm-probability "$MLM_PROBABILITY" \
            --num-workers "$NUM_WORKERS" \
            --seed "$SEED" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 2: evaluate (always re-runs; eval is cheap and the report is overwritten)
# --------------------------------------------------------------------------- #
EVAL_EXTRA=()
if [[ -n "${BASELINE_LLM:-}" ]]; then
    EVAL_EXTRA+=(--baseline-llm "$BASELINE_LLM")
fi
if [[ "$SKIP_MTEB" == "1" ]]; then
    EVAL_EXTRA+=(--skip-mteb)
fi

mkdir -p "$EVAL_DIR"

# MTEB_TASKS is intentionally word-split for multi-task input.
# shellcheck disable=SC2086
run_stage "02_eval_llm2vec" \
    python eval_llm2vec.py \
        --llm2vec-dir "$LLM2VEC_DIR" \
        --output-dir "$EVAL_DIR" \
        --data-cache-dir "$DOLMA_CACHE" \
        --max-files 1 \
        --n-sentences "$N_SENTENCES" \
        --n-bidir-causal "$N_BIDIR_CAUSAL" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --mlm-probability "$MLM_PROBABILITY" \
        --pooling "$POOLING" \
        --mteb-tasks $MTEB_TASKS \
        --mteb-batch-size "$MTEB_BATCH_SIZE" \
        --device "$DEVICE" \
        --seed "$SEED" \
        "${EVAL_EXTRA[@]}"

# --------------------------------------------------------------------------- #
# Wrap up
# --------------------------------------------------------------------------- #
T1=$(date +%s)
TOTAL=$((T1 - T0))

banner "[train-eval] eval_report.md"
if [[ -f "$EVAL_DIR/eval_report.md" ]]; then
    cat "$EVAL_DIR/eval_report.md"
else
    echo "(eval_report.md not found at $EVAL_DIR — eval failed?)"
fi

banner "[train-eval] DONE"
printf 'Total wall time this invocation: %dh%02dm%02ds\n' \
    $((TOTAL / 3600)) $(((TOTAL % 3600) / 60)) $((TOTAL % 60))
echo
echo "Artifacts:"
echo "  Trained ckpt    : $LLM2VEC_DIR"
echo "  Metrics JSON    : $EVAL_DIR/eval_metrics.json"
echo "  Report MD       : $EVAL_DIR/eval_report.md"
echo "  Per-stage logs  : $LOG_DIR/"
echo "  Timing summary  : $SUMMARY"
echo
echo "学習効果のクイック判定 (詳細は eval_report.md):"
echo "  - eval 1 MNTP loss が <8 なら MNTP は走った (random init ≈ 12)"
echo "  - eval 2 causal-PPL drift が +5〜+30% 程度なら MNTP の causal 影響は健全"
echo "  - eval 3 mean cosine が <0.999 なら bidir patch が効いている"
echo "  - eval 5 STSBenchmark Spearman が >0.65 なら sentence embedding 学習に成功"
