#!/bin/bash
# Ablation queue for interact-g (user instruction 2026-07-19: stack ALL
# Tier-1 + Tier-2 ablations on interact-g). Every stage is guarded by its
# output, so a 2h walltime kill just means "run me again" — the driver
# relaunches this script until it prints ALL-ABLATIONS-DONE.
# Order: FIC gen resume -> Tier2 probes (cheap) -> Tier1 trainings.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

L2V=runs/mcgill_gemma_repro_3k/final
SAE=layer_12/width_16k/average_l0_82/params.npz
BLK=runs/blocklist/blocklist.npy
CKPT=runs/prod_gemma_v6/eflm_l12_v5f/eflm-final.pt
BASE=(--frame repeat --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12
      --k-amp 64 --k-sup 64 --device cuda)

# ---- S0: finish FIC generation (resumes; fast no-op when complete) ----
python scripts/eval_fic_gen.py --llm2vec-dir "$L2V" --sae-path "$SAE" \
    --sae-layer 12 --blocklist "$BLK" --arms steer,clamp,prompting \
    --output-dir runs/prod_gemma_v6/fic_l12 --device cuda

# ---- S1: Tier2-6 iterative inference (rounds=3, probe only) -----------
if [ ! -f runs/prod_gemma_v6/abl_rounds3/report.md ]; then
    python scripts/eval_ef_bare.py "${BASE[@]}" --blocklist "$BLK" \
        --ef-ckpt "$CKPT" --arms ef --rounds 3 --sample-size 499 \
        --output-dir runs/prod_gemma_v6/abl_rounds3
fi

# ---- S2: Tier2-5 inference-time top-k sweep (200 pairs each) ----------
for K in 1 4 8 16 32; do
    if [ ! -f runs/prod_gemma_v6/abl_k$K/report.md ]; then
        python scripts/eval_ef_bare.py --frame repeat \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp $K --k-sup $K \
            --ef-ckpt "$CKPT" --arms ef --sample-size 499 \
            --conditions true,random \
            --output-dir runs/prod_gemma_v6/abl_k$K --device cuda
    fi
done

# ---- S2b: intervention-magnitude sweep at best k (user addition) ------
for S in 0.5 0.75 1.25 1.5 2.0; do
    if [ ! -f runs/prod_gemma_v6/abl_scale$S/report.md ]; then
        python scripts/eval_ef_bare.py "${BASE[@]}" --blocklist "$BLK" \
            --ef-ckpt "$CKPT" --arms ef --ef-scale $S --sample-size 200 \
            --conditions true,random \
            --output-dir runs/prod_gemma_v6/abl_scale$S
    fi
done

# ---- S3: Tier2-7 spec scope global (probe only) -----------------------
if [ ! -f runs/prod_gemma_v6/abl_scope_global/report.md ]; then
    python scripts/eval_ef_bare.py "${BASE[@]}" --blocklist "$BLK" \
        --ef-ckpt "$CKPT" --arms ef --spec-scope global \
        --sample-size 499 --conditions true,random \
        --output-dir runs/prod_gemma_v6/abl_scope_global
fi

# ---- S4: Tier2-8 no blocklist (probe only) ----------------------------
if [ ! -f runs/prod_gemma_v6/abl_noblock/report.md ]; then
    python scripts/eval_ef_bare.py "${BASE[@]}" \
        --ef-ckpt "$CKPT" --arms ef --sample-size 499 \
        --conditions true,random \
        --output-dir runs/prod_gemma_v6/abl_noblock
fi

# ---- L20 pipeline MOVED to short-g batch (user 2026-07-20: second
# ---- short-g slot approved; interact-g unreliable). Do NOT train L20
# ---- here — a short-g job owns editflow_s2_l20/s3_l20/eflm_l20_v5f2
# ---- (double-writer hazard). ------------------------------------------

# ---- Tier1 trainings (each guarded by its own probe500 report) --------
# S5: Tier1-2 no S3 warm start
if [ ! -f runs/prod_gemma_v6/eflm_l12_v5f_noS3/probe500/report.md ]; then
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        OUT_SUFFIX=_v5f_noS3 MAX_STEPS=40000 bash run_ef_editor.sh
fi
# S6: Tier1-3 no contrast teachers (mismatch 0, empty 0)
if [ ! -f runs/prod_gemma_v6/eflm_l12_v5f_noctr/probe500/report.md ]; then
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        FLOW_INIT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
        MM_PROB=0.0 EMPTY_PROB=0.0 \
        OUT_SUFFIX=_v5f_noctr MAX_STEPS=40000 bash run_ef_editor.sh
fi
# S7: Tier1-4 no norm budget
if [ ! -f runs/prod_gemma_v6/eflm_l12_v5f_nobudget/probe500/report.md ]; then
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        FLOW_INIT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
        NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        OUT_SUFFIX=_v5f_nobudget MAX_STEPS=40000 bash run_ef_editor.sh
fi

# ---- S8: per-feature aggregation (stdlib, cheap) ----------------------
python3 scripts/aggregate_per_feature.py \
    --records runs/prod_gemma_v6/eflm_l12_v5f/probe500/records.jsonl \
    --out runs/tables/perfeature_v5f_l12.md
python3 scripts/aggregate_per_feature.py --out runs/tables/perfeature_ksweep.md \
    --sweep "k1=runs/prod_gemma_v6/abl_k1/records.jsonl:k4=runs/prod_gemma_v6/abl_k4/records.jsonl:k8=runs/prod_gemma_v6/abl_k8/records.jsonl:k16=runs/prod_gemma_v6/abl_k16/records.jsonl:k32=runs/prod_gemma_v6/abl_k32/records.jsonl:k64=runs/prod_gemma_v6/eflm_l12_v5f/probe500/records.jsonl"

echo "==================== ALL-ABLATIONS-DONE ===================="
