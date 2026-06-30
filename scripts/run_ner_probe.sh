#!/usr/bin/env bash
# scripts/run_ner_probe.sh
#
# Runs NER probing on a fixed comparison set:
#   1. Base Gemma-2B, causal (the native mode)
#   2. Base Gemma-2B + bidir patch (free attention flip, no training)
#   3. Our Bi+MNTP+SimCSE LoRA Gemma
# Optionally also:
#   4. Base Mistral-7B + bidir patch
#   5. Our merged McGill Mistral (Bi+MNTP+SimCSE)
#
# Prints a delta table at the end so the contribution of (bidir patch) vs
# (Bi+MNTP+SimCSE training) is visible side-by-side.
#
# Per-run takes ~5-10 min on H200 (Gemma-2B) / ~15-20 min (Mistral-7B):
#   - 5 epochs over CoNLL-2003 train (14K sentences) with a frozen
#     encoder forward per batch
#   - eval on 3.5K test sentences
#
# Usage:
#   bash scripts/run_ner_probe.sh                          # default 3-way Gemma
#   INCLUDE_MISTRAL=1 bash scripts/run_ner_probe.sh        # add Mistral pair
#   EPOCHS=3 bash scripts/run_ner_probe.sh                 # faster smoke
#   OUR_GEMMA_DIR=./runs/llm2vec_lora_v2/llm2vec_simcse bash scripts/run_ner_probe.sh
#
# Outputs: $RUN_DIR/ner_{run}.json plus a summary table to stdout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR=${RUN_DIR:-"./runs/ner_probe"}
mkdir -p "$RUN_DIR"

BASE_GEMMA=${BASE_GEMMA:-"google/gemma-2-2b"}
OUR_GEMMA_DIR=${OUR_GEMMA_DIR:-"./runs/llm2vec_lora/llm2vec_simcse"}
BASE_MISTRAL=${BASE_MISTRAL:-"mistralai/Mistral-7B-Instruct-v0.2"}
OUR_MCGILL_DIR=${OUR_MCGILL_DIR:-"./runs/mcgill_ref/llm2vec_simcse"}

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-3}
SEED=${SEED:-42}

INCLUDE_MISTRAL=${INCLUDE_MISTRAL:-0}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
run_one() {
    local label=$1 encoder=$2 bidir_arg=$3
    local out="$RUN_DIR/ner_${label}.json"
    if [[ -f "$out" ]]; then
        echo "[ner-probe] SKIP $label — $out exists. Delete to redo."
        return
    fi
    echo
    echo "==============================================================="
    echo " NER probe: $label"
    echo "==============================================================="
    python scripts/eval_ner_probe.py \
        --encoder "$encoder" \
        $bidir_arg \
        --output-json "$out" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        --dtype bfloat16 \
        --device cuda
}

# --------------------------------------------------------------------------- #
# Run set
# --------------------------------------------------------------------------- #
echo "[ner-probe] writing results to $RUN_DIR/"
echo "[ner-probe] Gemma comparisons: base-causal, base-bidir, our LoRA"

run_one "gemma_base_causal"   "$BASE_GEMMA"     "--no-bidir"
run_one "gemma_base_bidir"    "$BASE_GEMMA"     ""
run_one "gemma_lora_bidir"    "$OUR_GEMMA_DIR"  ""

if [[ "$INCLUDE_MISTRAL" == "1" ]]; then
    echo
    echo "[ner-probe] Mistral comparisons: base-bidir, McGill merged"
    run_one "mistral_base_bidir"  "$BASE_MISTRAL"   ""
    run_one "mistral_mcgill_bidir" "$OUR_MCGILL_DIR" ""
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo
echo "==============================================================="
echo " NER probe summary"
echo "==============================================================="
python - <<EOF
import json
from pathlib import Path

rows = []
for path in sorted(Path("$RUN_DIR").glob("ner_*.json")):
    with open(path) as f:
        r = json.load(f)
    rows.append({
        "label": path.stem.replace("ner_", ""),
        "encoder": r["encoder"].split("/")[-1] if "/" in r["encoder"] else r["encoder"],
        "bidir": "yes" if r["bidirectional_patch"] else "no",
        "entity_f1": r.get("entity_f1"),
        "token_macro_f1": r.get("token_macro_f1_non_o"),
    })

if not rows:
    print("(no runs found)")
else:
    print(f"{'label':<26s} {'encoder':<28s} {'bidir':>6s} {'entity_F1':>10s} {'token_F1':>10s}")
    print("-" * 84)
    for r in rows:
        ef = r["entity_f1"]
        tf = r["token_macro_f1"]
        ef_s = f"{ef:.4f}" if ef is not None else "    -"
        tf_s = f"{tf:.4f}" if tf is not None else "    -"
        print(f"{r['label']:<26s} {r['encoder']:<28s} {r['bidir']:>6s} {ef_s:>10s} {tf_s:>10s}")

    # Side-by-side deltas
    by = {r["label"]: r for r in rows}
    print()
    print("Δ entity_F1 (= recipe contribution):")
    if "gemma_base_causal" in by and "gemma_base_bidir" in by:
        a = by["gemma_base_causal"]["entity_f1"]
        b = by["gemma_base_bidir"]["entity_f1"]
        if a is not None and b is not None:
            print(f"  Gemma: bidir patch alone   ({b:.4f}) - causal ({a:.4f}) = {b-a:+.4f}")
    if "gemma_base_bidir" in by and "gemma_lora_bidir" in by:
        a = by["gemma_base_bidir"]["entity_f1"]
        b = by["gemma_lora_bidir"]["entity_f1"]
        if a is not None and b is not None:
            print(f"  Gemma: Bi+MNTP+SimCSE LoRA ({b:.4f}) - base+bidir ({a:.4f}) = {b-a:+.4f}")
    if "mistral_base_bidir" in by and "mistral_mcgill_bidir" in by:
        a = by["mistral_base_bidir"]["entity_f1"]
        b = by["mistral_mcgill_bidir"]["entity_f1"]
        if a is not None and b is not None:
            print(f"  Mistral: McGill Bi+MNTP+SimCSE ({b:.4f}) - base+bidir ({a:.4f}) = {b-a:+.4f}")
    print()
    print("Interpretation guide (from LLM2Vec paper §4 Table 2):")
    print("  Δ ≥ 0.10  → recipe is doing real work (paper-typical for Bi+MNTP+SimCSE)")
    print("  0.05 ≤ Δ < 0.10 → recipe modestly helps")
    print("  Δ < 0.05  → training barely moved per-token representations")
EOF

echo
echo "Per-run JSONs: $RUN_DIR/ner_*.json"
