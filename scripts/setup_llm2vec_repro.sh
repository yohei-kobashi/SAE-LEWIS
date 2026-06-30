#!/usr/bin/env bash
# scripts/setup_llm2vec_repro.sh
#
# Set up a DEDICATED venv to reproduce LLM2Vec paper numbers without
# touching the main `openr1` environment.
#
# Why a separate venv?
#   - llm2vec 0.2.3 pins `transformers <= 4.44.2` and `tokenizers < 0.20`,
#     which forces a downgrade. Installing it in openr1 broke our
#     existing checkpoints' tokenizer.json (newer format).
#   - This script creates a clean venv with --system-site-packages so we
#     INHERIT system torch (cu126 + aarch64 wheel), then layer
#     llm2vec + its older transformers/peft locally inside the venv.
#   - The venv-local transformers takes precedence inside the venv; the
#     system openr1 is unaffected.
#
# Usage:
#   bash scripts/setup_llm2vec_repro.sh
#   # then activate it:
#   source ~/venvs/llm2vec_repro/bin/activate
#
# Override the venv location:
#   VENV_DIR=/scratch/$USER/llm2vec_repro bash scripts/setup_llm2vec_repro.sh

set -euo pipefail

VENV_DIR=${VENV_DIR:-"$HOME/venvs/llm2vec_repro"}
PY=${PY:-python}

echo "[setup] target venv: $VENV_DIR"

if [[ -d "$VENV_DIR" ]]; then
    echo "[setup] venv already exists at $VENV_DIR."
    echo "[setup] activate with: source $VENV_DIR/bin/activate"
    echo "[setup] to redo, delete the dir first: rm -rf $VENV_DIR"
    exit 0
fi

# Create venv that inherits the system / openr1 site-packages so we keep
# the working aarch64 torch wheel. Inside the venv we'll layer llm2vec's
# strict transformers/peft on top — pip prefers venv-local packages over
# system ones at import time.
echo "[setup] creating venv with --system-site-packages..."
mkdir -p "$(dirname "$VENV_DIR")"
$PY -m venv --system-site-packages "$VENV_DIR"

# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

# Sanity check: torch must come from the inherited environment.
echo "[setup] verifying inherited torch..."
python - <<'PYEOF'
import sys
import torch
print(f"  python  = {sys.version.split()[0]}")
print(f"  torch   = {torch.__version__}")
print(f"  cuda    = {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("[setup] FATAL: inherited torch can't see CUDA. Run the script "
             "from a node/login shell that already has CUDA available "
             "(activate openr1 first?).")
PYEOF

echo "[setup] upgrading pip inside venv..."
pip install --quiet --upgrade pip

# Install llm2vec with its strict deps. This pulls transformers 4.44.2 and
# tokenizers 0.19.x locally — these will shadow the system versions only
# when this venv is activated.
echo "[setup] installing llm2vec (this pins transformers <= 4.44.2)..."
pip install --quiet llm2vec

# mteb < 2.0 has the older Encoder API that llm2vec's docs/examples use.
echo "[setup] installing mteb < 2 (compatible with llm2vec's wrapper)..."
pip install --quiet "mteb<2"

# Useful extras for the eval:
pip install --quiet seqeval datasets scipy

echo
echo "[setup] verifying installation..."
python - <<'PYEOF'
import importlib
mods = ["llm2vec", "transformers", "peft", "mteb", "datasets", "scipy", "torch"]
import torch
print("  module          version")
print("  -----------     ------------------")
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "(no __version__)")
        print(f"  {m:<14s}  {v}")
    except ImportError as e:
        print(f"  {m:<14s}  FAILED: {e}")
print()
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  torch.cuda.device_count() = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

# Quick smoke: can we actually instantiate LLM2Vec.from_pretrained signature?
from llm2vec import LLM2Vec
print(f"  LLM2Vec.from_pretrained importable: yes")
PYEOF

cat <<EOF

===============================================================================
 Setup OK.
===============================================================================

 To use this venv:

     source $VENV_DIR/bin/activate

 To deactivate and return to openr1:

     deactivate
     # then re-activate openr1 if needed

 The system openr1 env is untouched. Verify by running this in a fresh shell
 WITHOUT activating the venv:

     python -c "import transformers; print(transformers.__version__)"
     # should still show 4.57.6 (openr1's version), not 4.44.2

 Next: run the reproduction eval (see scripts/run_llm2vec_repro.sh).
EOF
