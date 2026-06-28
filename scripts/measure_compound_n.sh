#!/usr/bin/env bash
# scripts/measure_compound_n.sh
#
# Measures the impact of varying the compound op count N on corruption
# yield, PPL ratio, and SAE shift. Runs corruption.py once per N value in
# --calibration-mode (so every attempt's metrics are recorded regardless
# of the gate), and then calls analyze_compound_n.py to produce a
# comparison report.
#
# Outputs:
#   $RUN_DIR/n0/calibration.jsonl       (N=0, identity baseline)
#   $RUN_DIR/n1/calibration.jsonl       ...
#   $RUN_DIR/nK/calibration.jsonl
#   $RUN_DIR/report.tsv                 per-N summary
#   $RUN_DIR/report.md                  human-readable report
#
# This script does NOT regenerate stage 0 / stage 1 — they must already
# exist (typically from a prior smoke or production run). The corruption
# stage is the only one that depends on the per-N comparison.
#
# Usage:
#   LLM2VEC_DIR=./runs/smoke-XXX/llm2vec \
#   SAE_CACHE_DIR=./runs/smoke-XXX/sae_cache \
#   bash scripts/measure_compound_n.sh
#
#   N_VALUES="1 2 3 4 5" ATTEMPTS_PER_N=300 bash scripts/measure_compound_n.sh
#
# Environment variables (override any of these):
#   RUN_DIR             output root (default: ./runs/measure-N-<ts>)
#   DEVICE              cuda | cpu        (default: cuda)
#   SEED                                  (default: 42)
#   LLM2VEC_DIR         (required) prior MNTP'd Gemma checkpoint
#   DOLMA_MAX_FILES     # of Dolma shards (default: 1)
#   N_VALUES            space-separated N values to test (default "0 1 2 3 4 5")
#   ATTEMPTS_PER_N      target attempts per N value (default: 500)
#                        This is the cap on --target-samples; corruption
#                        only writes accepted samples but counts every
#                        attempt's metric in calibration mode.
#   K_BUDGET            first-accept attempts per source sentence (default: 6)
#   SAMPLES_PER_SHARD   (default: 1000)
#   LLM                 (default: google/gemma-2-2b)
#   SAE_REPO            (default: google/gemma-scope-2b-pt-res)
#   SAE_PATH            (default: layer_12/width_16k/average_l0_82/params.npz)
#   SAE_LAYER           (default: 12)
#   MLM_MODEL           (default: modernbert-base)
#   SPACY_MODEL         (default: en_core_web_sm)
#   N_MAX               cap on N (default: 5; align with corruption.py's
#                        --n-max so highest force-n values are not clamped).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RUN_DIR=${RUN_DIR:-"./runs/measure-N-$(date +%Y%m%d-%H%M%S)"}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

LLM2VEC_DIR=${LLM2VEC_DIR:?LLM2VEC_DIR is required (point at an existing MNTP\'d Gemma checkpoint)}
DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-1}

N_VALUES=${N_VALUES:-"0 1 2 3 4 5"}
ATTEMPTS_PER_N=${ATTEMPTS_PER_N:-500}
K_BUDGET=${K_BUDGET:-6}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-1000}
N_MAX=${N_MAX:-5}

LLM=${LLM:-"google/gemma-2-2b"}
SAE_REPO=${SAE_REPO:-"google/gemma-scope-2b-pt-res"}
SAE_PATH=${SAE_PATH:-"layer_12/width_16k/average_l0_82/params.npz"}
SAE_LAYER=${SAE_LAYER:-12}
MLM_MODEL=${MLM_MODEL:-"modernbert-base"}
SPACY_MODEL=${SPACY_MODEL:-"en_core_web_sm"}

# Dolma cache: keep alongside the run so re-runs reuse shards.
DOLMA_CACHE="$RUN_DIR/dolma_cache"

mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

banner() {
    printf '\n===============================================================\n'
    printf ' %s\n' "$1"
    printf '===============================================================\n'
}

banner "[measure-N] config"
cat <<EOF
RUN_DIR            = $RUN_DIR
LLM2VEC_DIR        = $LLM2VEC_DIR
N_VALUES           = $N_VALUES
ATTEMPTS_PER_N     = $ATTEMPTS_PER_N
K_BUDGET           = $K_BUDGET
N_MAX              = $N_MAX
DEVICE             = $DEVICE        SEED = $SEED
DOLMA_MAX_FILES    = $DOLMA_MAX_FILES
LLM / SAE / MLM    = $LLM / $SAE_PATH / $MLM_MODEL
SPACY_MODEL        = $SPACY_MODEL
EOF

# --------------------------------------------------------------------------- #
# Run corruption.py once per N value in --calibration-mode
# --------------------------------------------------------------------------- #
for N in $N_VALUES; do
    sub_dir="$RUN_DIR/n${N}"
    log="$LOG_DIR/n${N}.log"
    if [[ -f "$sub_dir/calibration.jsonl" ]]; then
        echo "[measure-N] N=$N already calibrated at $sub_dir — skipping"
        continue
    fi
    mkdir -p "$sub_dir"
    banner "[measure-N] N=$N  →  $sub_dir"
    echo "[measure-N] log: $log"
    # We seed each run differently so a single source-sentence stream
    # does not bias all N values toward the same sentences. (The seed
    # offset is the N value itself; deterministic across re-runs.)
    SEED_N=$((SEED + N))
    set +e
    python corruption.py \
        --data-cache-dir "$DOLMA_CACHE" \
        --max-files "$DOLMA_MAX_FILES" \
        --out-dir "$sub_dir" \
        --llm2vec-dir "$LLM2VEC_DIR" \
        --llm "$LLM" \
        --sae-repo "$SAE_REPO" \
        --sae-path "$SAE_PATH" \
        --sae-layer "$SAE_LAYER" \
        --mlm-model "$MLM_MODEL" \
        --spacy-model "$SPACY_MODEL" \
        --target-samples "$ATTEMPTS_PER_N" \
        --samples-per-shard "$SAMPLES_PER_SHARD" \
        --k-budget "$K_BUDGET" \
        --n-max "$N_MAX" \
        --force-n "$N" \
        --calibration-mode \
        --calibration-out "$sub_dir/calibration.jsonl" \
        --device "$DEVICE" \
        --seed "$SEED_N" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -ne 0 ]]; then
        printf '[measure-N] N=%s FAILED rc=%d  see %s\n' "$N" "$rc" "$log" >&2
        exit "$rc"
    fi
done

# --------------------------------------------------------------------------- #
# Analyze
# --------------------------------------------------------------------------- #
banner "[measure-N] analyze"
python scripts/analyze_compound_n.py "$RUN_DIR" \
    --report-tsv "$RUN_DIR/report.tsv" \
    --report-md "$RUN_DIR/report.md" \
    --n-values $N_VALUES

cat <<EOF

Artifacts:
  Run dir        : $RUN_DIR
  Per-N logs     : $LOG_DIR/n{N}.log
  Per-N metrics  : $RUN_DIR/n{N}/calibration.jsonl
  Summary TSV    : $RUN_DIR/report.tsv
  Summary MD     : $RUN_DIR/report.md
EOF
