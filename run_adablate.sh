#!/bin/bash -l

# Ours-AD ablation pair (user 2026-07-25), both at ~3.2 epochs matched
# to the frozen Ours-AD point (p100 s8000 = 3.24 ep of 9,876 rows):
#   A) T4-v1 naive adaptation extended: same first-approach recipe
#      (cache t4_ctx_l12 WITHOUT scrambled-null rows, default LR 3e-4,
#      init=T2) resumed 4000 -> 6320 steps (= 3.20 ep of 7,902 rows).
#   B) no-synthetic-pretraining: the exact Ours-AD defended recipe
#      (t4_ctx_l12_v2 + scramble rows, LR 1e-4) but trained DIRECTLY on
#      LLM2Vec — no --init-ckpt, i.e. no Dolma corruption-pair stage.
#      8000 steps to match Ours-AD's compute point.
# Both eval500 at the matched point only (ablation rows; no dev
# selection — the champion keeps its E'-selected protocol).
# Run inside interact-g: bash run_adablate.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random
     --sample-size 500 --device cuda)

# ---- A) v1 naive, resumed to 3.2 epochs ---------------------------------
V1=$P/eflm_l12_t4_v6t2
if [ ! -f "$V1/eflm-step6320.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $V4/t4_ctx_l12 \
        --dev-corruption-dir $V4/corruption_seldev_zctx_l12 \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$V1" \
        --inject-layer 12 \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat \
        --init-ckpt $P/eflm_l12_v6t2/eflm-final.pt \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 6320 --resume --device cuda
fi
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/fs_t4v1e32_l12$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${EVC[@]}" --ef-ckpt "$V1/eflm-step6320.pt" \
        --output-dir "$O" $X
done

# ---- B) defended recipe, no synthetic pretraining (scratch on LLM2Vec) --
SCR=$P/eflm_l12_adscratch
if [ ! -f "$SCR/eflm-step8000.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $V4/t4_ctx_l12_v2 \
        --dev-corruption-dir $V4/corruption_seldev_zctx_l12 \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$SCR" \
        --inject-layer 12 \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat \
        --learning-rate 1e-4 --backbone-lr 1e-4 \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 8000 --resume --device cuda
fi
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/fs_adscratch_l12$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${EVC[@]}" --ef-ckpt "$SCR/eflm-step8000.pt" \
        --output-dir "$O" $X
done

echo "==================== ADABLATE-DONE ===================="
for d in $P/fs_t4v1e32_l12* $P/fs_adscratch_l12*; do
    echo "--- $d"
    grep -E "^\| (true|random) \| ef" $d/report.md || true
done
