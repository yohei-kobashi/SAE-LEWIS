#!/bin/bash
# C1-0 adoption run — e2e paper numbers with the cmlm8+lens1 decode
# (README §13.7): same 500 pairs / operating point as
# eval_lingualens_final, only --fill-iterative 8 added. Expect ~2-3x the
# editor forwards of the parallel run (fixed-point early exit keeps
# single-site templates at ~1 forward). ~1.5-3h on one GPU.
# Run on miyabi with a GPU session:  bash scripts/run_e2e_cmlm_v6.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
OUT=$V6/eval_lingualens_cmlm

python eval_lingualens.py \
    --llm2vec-dir "$LLM2VEC" \
    --tagger-ckpt "$V6/tagger/tagger-final.pt" \
    --editor-ckpt "$V6/editor/editor-final.pt" \
    --output-dir "$OUT" \
    --sae-path layer_12/width_16k/average_l0_82/params.npz \
    --cond-scope local --blocklist "$BLOCKLIST" \
    --steer-lambda 1 --fill-iterative 8 \
    --k-amp 64 --k-sup 64 --ins-threshold 0.9 \
    --sample-size 500 --refine-passes 3 --refine-recompute \
    --fluency-gate 0.5 --dump-details --device cuda

echo "==================== CMLM vs PARALLEL (same 500 pairs) ===================="
python - "$V6" <<'EOF'
import json, sys
v6 = sys.argv[1]
rows = {}
for name, d in (("parallel (final)", "eval_lingualens_final"),
                ("cmlm8 (this run)", "eval_lingualens_cmlm")):
    try:
        s = json.load(open(f"{v6}/{d}/summary.json"))
        rows[name] = s
    except FileNotFoundError:
        print(f"[compare] missing {d}/summary.json")
keys = ("sim_target", "exact_match", "copy_rate", "edit_loc_iou",
        "sae_shift")
print(f"{'run':22s}" + "".join(f"{k:>14s}" for k in keys))
for name, s in rows.items():
    for cond in ("true", "empty"):
        t = s["conditions"].get(cond)
        if t is None:
            continue
        label = f"{name} [{cond}]"
        print(f"{label:22s}" + "".join(
            f"{t.get(k, float('nan')):14.4f}" for k in keys))
    print(f"{'  input-copy baseline':22s}"
          f"{s['input_copy_baseline']['sim_target']:14.4f}")
EOF
