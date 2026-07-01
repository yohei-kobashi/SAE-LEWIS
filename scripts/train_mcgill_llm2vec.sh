#!/usr/bin/env bash
# scripts/train_mcgill_llm2vec.sh
#
# Full pipeline: vendored McGill run_mntp.py → run_simcse.py → merge into
# a drop-in --llm2vec-dir for SAE-LEWIS downstream.
#
# All training runs in the isolated venv set up by setup_llm2vec_repro.sh
# (transformers 4.44.2 / tokenizers 0.19.x / peft < 0.13). The merge step
# also runs in that venv since it uses peft. Downstream stages
# (corruption / tagger / editor / eval) continue to use openr1 as usual —
# they only read the merged HF checkpoint that this pipeline emits.
#
# Usage:
#   bash scripts/train_mcgill_llm2vec.sh                    # default: Gemma-2-2b, full recipe
#   bash scripts/train_mcgill_llm2vec.sh --skip-mntp        # SimCSE only (needs mntp/ exists)
#   bash scripts/train_mcgill_llm2vec.sh --skip-simcse      # MNTP only, no simcse merge
#   RUN_ROOT=./runs/mcgill_v2 bash scripts/train_mcgill_llm2vec.sh   # custom output
#
# Prerequisites:
#   1. isolated venv exists:      bash scripts/setup_llm2vec_repro.sh
#   2. McGill code vendored:      bash scripts/vendor_mcgill_llm2vec.sh
#   3. Wiki1M downloaded:         handled inline below (~200MB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=${VENV_DIR:-"$HOME/venvs/llm2vec_repro"}
VENDOR_DIR=${VENDOR_DIR:-"$REPO_ROOT/vendored/mcgill_llm2vec"}
RUN_ROOT=${RUN_ROOT:-"$REPO_ROOT/runs/mcgill_gemma_repro"}

MNTP_CONFIG=${MNTP_CONFIG:-"$REPO_ROOT/mcgill_configs/mntp/Gemma-2-2b.json"}
SIMCSE_CONFIG=${SIMCSE_CONFIG:-"$REPO_ROOT/mcgill_configs/simcse/Gemma-2-2b.json"}
BASE_MODEL=${BASE_MODEL:-"google/gemma-2-2b"}

# The training script `cd`s into $VENDOR_DIR so it can find its own
# `cache/wiki1m_for_simcse.txt` on relative paths. That breaks any
# relative paths the caller passed in — a `./runs/mcgill_sheared_repro`
# gets reinterpreted as `$VENDOR_DIR/runs/mcgill_sheared_repro`. Convert
# everything user-facing to absolute NOW, before we touch anything else.
mkdir -p "$RUN_ROOT"
RUN_ROOT=$(cd "$RUN_ROOT" && pwd)
MNTP_CONFIG=$(cd "$(dirname "$MNTP_CONFIG")" && pwd)/$(basename "$MNTP_CONFIG")
SIMCSE_CONFIG=$(cd "$(dirname "$SIMCSE_CONFIG")" && pwd)/$(basename "$SIMCSE_CONFIG")

WIKI1M_URL=${WIKI1M_URL:-"https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt"}
WIKI1M_PATH=${WIKI1M_PATH:-"$VENDOR_DIR/cache/wiki1m_for_simcse.txt"}

SKIP_MNTP=0
SKIP_SIMCSE=0
for arg in "$@"; do
    case "$arg" in
        --skip-mntp)   SKIP_MNTP=1 ;;
        --skip-simcse) SKIP_SIMCSE=1 ;;
        *)             echo "[train-mcgill] unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
banner() {
    printf '\n===============================================================\n'
    printf ' %s\n' "$1"
    printf '===============================================================\n'
}

banner "[train-mcgill] preflight"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[train-mcgill] FATAL: isolated venv not found at $VENV_DIR"
    echo "               run: bash scripts/setup_llm2vec_repro.sh"
    exit 1
fi
if [[ ! -d "$VENDOR_DIR" ]]; then
    echo "[train-mcgill] FATAL: McGill code not vendored at $VENDOR_DIR"
    echo "               run: bash scripts/vendor_mcgill_llm2vec.sh"
    exit 1
fi
for py in run_mntp.py run_simcse.py; do
    if [[ ! -f "$VENDOR_DIR/experiments/$py" ]]; then
        echo "[train-mcgill] FATAL: expected $VENDOR_DIR/experiments/$py"
        exit 1
    fi
done
for cfg in "$MNTP_CONFIG" "$SIMCSE_CONFIG"; do
    if [[ ! -f "$cfg" ]]; then
        echo "[train-mcgill] FATAL: config not found: $cfg"
        exit 1
    fi
done

echo "[train-mcgill] venv       : $VENV_DIR"
echo "[train-mcgill] vendored   : $VENDOR_DIR"
echo "[train-mcgill] MNTP cfg   : $MNTP_CONFIG"
echo "[train-mcgill] SimCSE cfg : $SIMCSE_CONFIG"
echo "[train-mcgill] run root   : $RUN_ROOT"

# Activate the isolated venv (all training + merge runs inside it).
# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"
echo "[train-mcgill] active python: $(which python)"
echo "[train-mcgill] transformers : $(python -c 'import transformers; print(transformers.__version__)')"

mkdir -p "$RUN_ROOT"

# --------------------------------------------------------------------------- #
# Prepare Wiki1M (needed for SimCSE unless --skip-simcse)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_SIMCSE" -eq 0 ]]; then
    banner "[train-mcgill] Wiki1M dataset"
    mkdir -p "$(dirname "$WIKI1M_PATH")"
    if [[ -s "$WIKI1M_PATH" ]]; then
        echo "[train-mcgill] Wiki1M already at $WIKI1M_PATH "
        echo "  ($(wc -l < "$WIKI1M_PATH") lines, $(du -h "$WIKI1M_PATH" | cut -f1))"
    else
        echo "[train-mcgill] downloading Wiki1M (~200MB from $WIKI1M_URL)"
        # Try wget first, fall back to curl.
        if command -v wget >/dev/null 2>&1; then
            wget --quiet -O "$WIKI1M_PATH" "$WIKI1M_URL"
        else
            curl -sfSL -o "$WIKI1M_PATH" "$WIKI1M_URL"
        fi
        echo "[train-mcgill] downloaded ($(wc -l < "$WIKI1M_PATH") lines)"
    fi
fi

# --------------------------------------------------------------------------- #
# Stage 1: MNTP (McGill run_mntp.py)
# --------------------------------------------------------------------------- #
MNTP_OUT="$RUN_ROOT/mntp"
if [[ "$SKIP_MNTP" -eq 1 ]]; then
    banner "[train-mcgill] MNTP  (SKIPPED by --skip-mntp)"
elif [[ -f "$MNTP_OUT/adapter_config.json" ]]; then
    banner "[train-mcgill] MNTP  (SKIPPED — adapter already at $MNTP_OUT)"
else
    banner "[train-mcgill] MNTP"
    # Copy the config into RUN_ROOT with output_dir rewritten to match RUN_ROOT.
    # Also auto-enable HF Trainer resume when we detect partial checkpoint
    # dirs left over from a previous interrupted invocation, so re-running
    # the same command after e.g. an HPC walltime kill picks up from the
    # last save (save_steps: 200 in the config = at most 200 steps re-done).
    STAGE_CFG="$RUN_ROOT/_mntp_config.json"
    python - <<PYEOF
import json, pathlib
src = pathlib.Path("$MNTP_CONFIG")
cfg = json.loads(src.read_text())
cfg["output_dir"] = "$MNTP_OUT"
# Ensure model path is what we asked for at the shell level.
cfg["model_name_or_path"] = "$BASE_MODEL"
# HF Trainer accepts True (auto-detect latest checkpoint) or a specific path.
mntp_out = pathlib.Path("$MNTP_OUT")
if mntp_out.exists() and any(mntp_out.glob("checkpoint-*")):
    cfg["resume_from_checkpoint"] = True
    print(f"[stage-cfg] MNTP resume ENABLED (existing checkpoints found in {mntp_out})")
else:
    print("[stage-cfg] MNTP starting fresh (no prior checkpoints)")
pathlib.Path("$STAGE_CFG").write_text(json.dumps(cfg, indent=2))
PYEOF
    # Run McGill's training script from the vendored dir so relative paths
    # (like cache/wiki1m_for_simcse.txt) resolve as they expect.
    cd "$VENDOR_DIR"
    python experiments/run_mntp.py "$STAGE_CFG"
    cd "$REPO_ROOT"
fi

# --------------------------------------------------------------------------- #
# Stage 2: SimCSE (McGill run_simcse.py, using the MNTP LoRA as its base)
# --------------------------------------------------------------------------- #
SIMCSE_OUT="$RUN_ROOT/simcse"
if [[ "$SKIP_SIMCSE" -eq 1 ]]; then
    banner "[train-mcgill] SimCSE  (SKIPPED by --skip-simcse)"
elif [[ -f "$SIMCSE_OUT/adapter_config.json" ]]; then
    banner "[train-mcgill] SimCSE  (SKIPPED — adapter already at $SIMCSE_OUT)"
else
    banner "[train-mcgill] SimCSE"
    STAGE_CFG="$RUN_ROOT/_simcse_config.json"
    python - <<PYEOF
import json, pathlib
src = pathlib.Path("$SIMCSE_CONFIG")
cfg = json.loads(src.read_text())
cfg["output_dir"] = "$SIMCSE_OUT"
cfg["model_name_or_path"] = "$BASE_MODEL"
cfg["peft_model_name_or_path"] = "$MNTP_OUT"
cfg["dataset_file_path"] = "$WIKI1M_PATH"
# Same resume plumbing as MNTP stage.
simcse_out = pathlib.Path("$SIMCSE_OUT")
if simcse_out.exists() and any(simcse_out.glob("checkpoint-*")):
    cfg["resume_from_checkpoint"] = True
    print(f"[stage-cfg] SimCSE resume ENABLED (existing checkpoints found in {simcse_out})")
else:
    print("[stage-cfg] SimCSE starting fresh (no prior checkpoints)")
pathlib.Path("$STAGE_CFG").write_text(json.dumps(cfg, indent=2))
PYEOF
    cd "$VENDOR_DIR"
    python experiments/run_simcse.py "$STAGE_CFG"
    cd "$REPO_ROOT"
fi

# --------------------------------------------------------------------------- #
# Stage 3: merge into a drop-in --llm2vec-dir for downstream
# --------------------------------------------------------------------------- #
FINAL_DIR="$RUN_ROOT/final"
banner "[train-mcgill] merging LoRA + adding special tokens → $FINAL_DIR"

# McGill's run_mntp.py / run_simcse.py write two kinds of adapter dirs:
#
#   $OUT/checkpoint-N/         ← trainer's save_steps + save-on-stop path.
#                                Guaranteed to contain the trained LoRA
#                                (adapter_config.json + adapter_model.safetensors)
#                                after each save_steps trigger.
#   $OUT/                      ← trainer.save_model(output_dir) at the end.
#                                Empirically unreliable when StopTrainingCallback
#                                interrupts the loop: MNTP writes a stale/zero
#                                adapter here (merge delta = 0.0e+00); SimCSE
#                                skips this save entirely so nothing lands here.
#
# So: prefer the highest-N checkpoint dir. Fall back to $OUT only if no
# checkpoint dir exists (should never happen after a full run, but keeps
# the flow correct if someone hand-crafts an adapter dir).
select_adapter_dir() {
    local base="$1"
    [[ -d "$base" ]] || { echo ""; return; }
    local best_ck="" best_n=-1
    for ck in "$base"/checkpoint-*; do
        [[ -d "$ck" ]] || continue
        local n="${ck##*checkpoint-}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        if (( n > best_n )) && [[ -f "$ck/adapter_config.json" ]]; then
            best_ck="$ck"; best_n=$n
        fi
    done
    if [[ -n "$best_ck" ]]; then
        echo "$best_ck"
    elif [[ -f "$base/adapter_config.json" ]]; then
        echo "$base"
    fi
}

MNTP_ADAPTER=$(select_adapter_dir "$MNTP_OUT")
SIMCSE_ADAPTER=$(select_adapter_dir "$SIMCSE_OUT")

if [[ -z "$MNTP_ADAPTER" ]]; then
    echo "[train-mcgill] FATAL: no MNTP adapter found under $MNTP_OUT" >&2
    exit 1
fi
echo "[train-mcgill] MNTP adapter    : $MNTP_ADAPTER"
echo "[train-mcgill] SimCSE adapter  : ${SIMCSE_ADAPTER:-<none>}"

MERGE_ARGS=(
    --base-model "$BASE_MODEL"
    --mntp-adapter "$MNTP_ADAPTER"
    --output-dir "$FINAL_DIR"
    --dtype bfloat16
)
if [[ "$SKIP_SIMCSE" -eq 0 && -n "$SIMCSE_ADAPTER" ]]; then
    MERGE_ARGS+=(--simcse-adapter "$SIMCSE_ADAPTER")
else
    echo "[train-mcgill] (SimCSE adapter absent; saving Bi+MNTP only)"
fi

python scripts/mcgill_merge_and_expand.py "${MERGE_ARGS[@]}"

# --------------------------------------------------------------------------- #
# Stage 4: STS-B eval on the merged checkpoint (paper number comparison)
# --------------------------------------------------------------------------- #
# Runs in the SAME isolated venv (still active). Uses the official llm2vec
# library's LLM2Vec.from_pretrained on the merged HF dir, computes cosine
# Spearman on the mteb/stsbenchmark-sts test split, and prints Δ vs the
# paper number for whatever base LLM this checkpoint was trained on.
#
# Set SKIP_EVAL=1 to disable (e.g. if you want to do a longer eval sweep
# separately).
if [[ "${SKIP_EVAL:-0}" -eq 1 ]]; then
    banner "[train-mcgill] eval SKIPPED (SKIP_EVAL=1)"
elif [[ ! -f "$FINAL_DIR/config.json" ]]; then
    banner "[train-mcgill] eval SKIPPED (no merged ckpt at $FINAL_DIR)"
else
    banner "[train-mcgill] eval — STS-B on merged ckpt"
    EVAL_JSON="$RUN_ROOT/eval_stsb.json"
    python scripts/llm2vec_repro_eval.py \
        --from-dir "$FINAL_DIR" \
        --label "$(basename "$RUN_ROOT")" \
        --tasks STSBenchmark \
        --output-json "$EVAL_JSON" \
        --no-mteb
fi

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
banner "[train-mcgill] DONE"
echo "  MNTP adapter    : $MNTP_OUT"
echo "  SimCSE adapter  : $SIMCSE_OUT"
echo "  Merged ckpt     : $FINAL_DIR"
if [[ -f "$RUN_ROOT/eval_stsb.json" ]]; then
    echo "  STS-B eval json : $RUN_ROOT/eval_stsb.json"
fi
echo
echo "Next: point SAE-LEWIS downstream at $FINAL_DIR"
echo
echo "  LLM2VEC_DIR=$FINAL_DIR \\"
echo "  SIMCSE_DIR=$FINAL_DIR \\"
echo "  RUN_DIR=./runs/prod_mcgill_gemma \\"
echo "    bash scripts/run_production.sh"
