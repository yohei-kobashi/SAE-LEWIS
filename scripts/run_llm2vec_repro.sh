#!/usr/bin/env bash
# scripts/run_llm2vec_repro.sh
#
# One-command driver for the LLM2Vec paper reproduction with the dedicated
# venv. Lazily sets up the venv on first invocation, activates it, runs
# the eval, then prints next-step pointers.
#
# Usage:
#   bash scripts/run_llm2vec_repro.sh
#   bash scripts/run_llm2vec_repro.sh --variants unsup-simcse supervised
#   bash scripts/run_llm2vec_repro.sh --tasks STSBenchmark STS17 BIOSSES
#
# Override the venv location:
#   VENV_DIR=/scratch/$USER/llm2vec_repro bash scripts/run_llm2vec_repro.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=${VENV_DIR:-"$HOME/venvs/llm2vec_repro"}

# Step 1: lazy setup. The setup script is idempotent (exits early if the
# venv already exists), so it's safe to call every time.
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[run-repro] no venv at $VENV_DIR — running setup..."
    VENV_DIR="$VENV_DIR" bash scripts/setup_llm2vec_repro.sh
fi

# Step 2: activate
# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

echo
echo "[run-repro] active python : $(which python)"
echo "[run-repro] transformers  : $(python -c 'import transformers; print(transformers.__version__)')"
echo

# Step 3: run the eval, forwarding any user-supplied flags.
python scripts/llm2vec_repro_eval.py "$@"

echo
echo "[run-repro] done. Raw results in runs/llm2vec_repro/results.json"
echo "[run-repro] to evaluate the supervised variant too:"
echo "  bash scripts/run_llm2vec_repro.sh --variants unsup-simcse supervised"
