#!/usr/bin/env bash
# scripts/setup_llm2vec_repro.sh
#
# Set up a DEDICATED venv to reproduce LLM2Vec paper numbers without
# touching the main `openr1` environment.
#
# Why a separate venv?
#   - llm2vec 0.2.3 pins `transformers <= 4.44.2` and `tokenizers < 0.20`,
#     which forces a downgrade. Installing it into openr1 broke our
#     existing checkpoints' tokenizer.json (newer format).
#   - This script creates a clean venv with NO --system-site-packages
#     (that flag only inherits from openr1's BASE python — /usr —
#     which doesn't have torch). Instead we drop a `.pth` file pointing
#     at openr1's actual site-packages so the new venv can see openr1's
#     aarch64 torch wheel. Venv-local packages still take precedence,
#     so the venv-local transformers 4.44.2 shadows openr1's 4.57.6.
#
# Usage:
#   source ~/openr1/...  # (or wherever openr1 lives — must be ACTIVE)
#   bash scripts/setup_llm2vec_repro.sh
#   source ~/venvs/llm2vec_repro/bin/activate
#
# Override the venv location:
#   VENV_DIR=/scratch/$USER/llm2vec_repro bash scripts/setup_llm2vec_repro.sh

set -euo pipefail

VENV_DIR=${VENV_DIR:-"$HOME/venvs/llm2vec_repro"}

echo "[setup] target venv: $VENV_DIR"

if [[ -d "$VENV_DIR" ]]; then
    echo "[setup] venv already exists at $VENV_DIR."
    echo "[setup] activate with: source $VENV_DIR/bin/activate"
    echo "[setup] to redo, delete the dir first: rm -rf $VENV_DIR"
    exit 0
fi

# --------------------------------------------------------------------------- #
# 1. Verify openr1 is active and grab its site-packages directory.
#    The HPC's aarch64 torch wheel lives there; we'll inherit via .pth.
# --------------------------------------------------------------------------- #
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[setup] FATAL: no venv active. Activate openr1 first, e.g.:"
    echo "  source /work/go25/b20048/open-r1/openr1/bin/activate"
    exit 1
fi
OPENR1_ROOT="$VIRTUAL_ENV"
OPENR1_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PY_TAG="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[setup] inheriting from openr1:"
echo "  VIRTUAL_ENV    = $OPENR1_ROOT"
echo "  site-packages  = $OPENR1_SITE"
echo "  python tag     = $PY_TAG"
if [[ ! -d "$OPENR1_SITE" ]]; then
    echo "[setup] FATAL: openr1 site-packages does not exist: $OPENR1_SITE"
    exit 1
fi
if [[ ! -f "$OPENR1_SITE/torch/__init__.py" && ! -f "$OPENR1_SITE/torch/__init__.pyi" ]]; then
    echo "[setup] FATAL: torch not found under $OPENR1_SITE/torch."
    echo "[setup]        openr1 may be in a different state than expected."
    exit 1
fi

# --------------------------------------------------------------------------- #
# 2. Create plain venv (no --system-site-packages — see header).
# --------------------------------------------------------------------------- #
echo "[setup] creating clean venv..."
mkdir -p "$(dirname "$VENV_DIR")"
python -m venv "$VENV_DIR"

# --------------------------------------------------------------------------- #
# 3. Drop a .pth file pointing at openr1's site-packages so the new venv
#    inherits torch + CUDA + numpy + datasets while still being able to
#    layer its own transformers / peft on top.
# --------------------------------------------------------------------------- #
NEW_SITE="$VENV_DIR/lib/$PY_TAG/site-packages"
if [[ ! -d "$NEW_SITE" ]]; then
    # Some systems (e.g. when running aarch64 with debian-style layout) put
    # site-packages under lib64/ as well. Pick whichever exists.
    NEW_SITE_ALT="$VENV_DIR/lib64/$PY_TAG/site-packages"
    if [[ -d "$NEW_SITE_ALT" ]]; then
        NEW_SITE="$NEW_SITE_ALT"
    else
        echo "[setup] FATAL: cannot locate the new venv's site-packages."
        echo "[setup]        tried: $NEW_SITE  AND  $NEW_SITE_ALT"
        exit 1
    fi
fi
echo "[setup] writing inherit .pth in $NEW_SITE/"
echo "$OPENR1_SITE" > "$NEW_SITE/00_openr1_inherit.pth"

# --------------------------------------------------------------------------- #
# 4. Activate the new venv and verify torch resolves to openr1's wheel.
# --------------------------------------------------------------------------- #
# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] verifying inherited torch from $NEW_SITE/00_openr1_inherit.pth..."
python - <<'PYEOF'
import sys, torch
print(f"  python  = {sys.version.split()[0]}")
print(f"  torch   = {torch.__version__}")
print(f"  cuda    = {torch.cuda.is_available()}")
print(f"  torch path = {torch.__file__}")
if not torch.cuda.is_available():
    sys.exit("[setup] FATAL: torch loaded but CUDA is unavailable. "
             "Check that openr1's torch is built for CUDA.")
PYEOF

echo "[setup] upgrading pip inside venv..."
pip install --quiet --upgrade pip

# --------------------------------------------------------------------------- #
# 5. Install llm2vec + companions LOCALLY in the new venv. These will
#    shadow openr1's newer transformers / tokenizers when this venv is
#    active.
# --------------------------------------------------------------------------- #
echo "[setup] installing llm2vec (pins transformers <= 4.44.2)..."
pip install --quiet llm2vec

echo "[setup] installing mteb < 2 (older API expected by llm2vec docs)..."
pip install --quiet "mteb<2"

# llm2vec only requires peft >= 0.7, so pip's resolver accepted openr1's
# inherited peft 0.19.1 (via our .pth) without installing a venv-local
# copy. But peft 0.19 references torch.distributed.tensor.DTensor at
# import-time of certain LoRA layer code, and openr1's torch 2.11 build
# doesn't expose that submodule — every LLM2Vec.from_pretrained() call
# then dies with `AttributeError: module 'torch.distributed' has no
# attribute 'tensor'`. The check was added in peft 0.13; older versions
# don't reach that code path. Force-install peft<0.13 in venv-local so
# it shadows the openr1 one.
echo "[setup] force-installing older peft (< 0.13) compatible with torch 2.11..."
pip install --quiet --force-reinstall --no-deps "peft>=0.10,<0.13"

echo "[setup] installing eval extras..."
pip install --quiet seqeval scipy

# --------------------------------------------------------------------------- #
# 6. Final smoke test.
# --------------------------------------------------------------------------- #
echo
echo "[setup] verifying installation..."
python - <<'PYEOF'
import importlib, sys
mods = ["llm2vec", "transformers", "peft", "mteb", "datasets", "scipy", "torch"]
print("  module         version              path")
print("  -----------    -------------------  ------------------------------------")
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "(no __version__)")
        p = getattr(mod, "__file__", "(no __file__)")
        if p and p.startswith("/"):
            # Shorten for readability
            short = p
            if len(short) > 38:
                short = "..." + short[-35:]
        else:
            short = str(p)
        print(f"  {m:<13s}  {v:<19s}  {short}")
    except ImportError as e:
        print(f"  {m:<13s}  FAILED               {e}")
import torch
print()
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  torch.cuda.device_count() = "
      f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}")
from llm2vec import LLM2Vec
print(f"  LLM2Vec.from_pretrained: importable")
PYEOF

cat <<EOF

===============================================================================
 Setup OK.
===============================================================================

 To use this venv:

     source $VENV_DIR/bin/activate

 Confirm it's working (transformers 4.44.2 / tokenizers 0.19.x):

     python -c "import transformers; print(transformers.__version__)"

 To deactivate and return to openr1:

     deactivate
     # then re-source openr1 if needed

 Next: run the reproduction eval

     bash scripts/run_llm2vec_repro.sh
EOF
