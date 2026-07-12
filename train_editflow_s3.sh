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

# S3 (EDIT_FLOWS_ZERO §5): Localized Edit Flows (paper appendix C.1) on
# the S2 champion. Training now samples localized propagation paths —
# every alignment slot self-fires at t* = u^{1/3} and propagates to
# Pois(λ_prop·Δt) neighbors, so x_t contains locally-consistent edit
# clusters — with per-op loss weights λ_eff = w(t) + λ_prop·(#adjacent
# sources) (eqs 43-44). The hazard base extends to w(t) + λ_prop·adj
# (adj = observable already-edited-neighbor count, fed via a zero-init
# embedding), so p stays a probability and warm-start from S2 is exact.
# At decode, thr's bar for a site next to an applied edit drops to
# p ≥ F·w/(w + λ_prop·adj) — the locality prior the 2-3/4-8 buckets
# have been missing. Paper reports +48% Pass@1 from this on MBPP.
# Target: the multi-site tail. ~5h; resubmit until "S3 DONE".
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
OUT=$V6/editflow_s3

if [ ! -f "$OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --init-from-editflow "$V6/editflow_s2/editflow-final.pt" \
        --rate-param hazard \
        --cond-mode feature-tokens \
        --true-align \
        --lora-r 32 \
        --lam-prop 4.0 \
        --max-steps 50000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 4000 \
        --batch-size 8 --num-workers 2 \
        --resume --device cuda
fi

if [ ! -f "$OUT/probe/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$OUT/editflow-final.pt" \
        --output-dir "$OUT/probe" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --steps 48 --steer-lambda 1 \
        --decode det,thr0.05,thr0.1,thr0.25,thr0.5,bo4@temp0.7 \
        --device cuda
fi

echo "==================== S3 DONE ===================="
sed -n '/## Gate (a)/,$p' "$OUT/probe/probe_report.md" | head -80
echo "Gates vs S2 champion (thr0.5 = 0.1859/0.6766, lambda-IoU 0.7609,"
echo "  empty 1.00, random 0.889):"
echo "  (i) THE POINT — multi-site buckets: 2-3 exact >= 0.29 (the"
echo "      pilot's number S2 never recovered) and/or 4-8 exact > 0.052;"
echo "  (ii) headline thr0.5 >= S2; (iii) empty no_edit 1.00, random"
echo "      >= 0.88 at the operating point; (iv) lambda-IoU >= 0.76."
echo "  Pass -> S4 (500-pair judgment vs v6 pipeline). Fail on (i) with"
echo "  (ii)-(iv) held -> locality doesn't transfer to LinguaLens spans;"
echo "  S4 runs with the S2 champion instead."
