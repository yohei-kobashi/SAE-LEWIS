#!/bin/bash -l

# Ratio ablation at FIXED step (user 2026-07-24: compare p25/p75/p100 at
# s2000, the operating point chosen for Ours-AD = p50 s2000). eval500
# both directions per ratio -> fs_mix{p25,p75,p100}_l12{,_amp}.
# (p50 s2000 = fs_mixft_l12{,_amp}, p0 = Ours-ZS, already measured.)
# Run inside interact-g: bash run_mixratio_eval.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for PMIX in p25 p75 p100; do
    CK=$P/eflm_l12_mixft_$PMIX/eflm-step2000.pt
    [ -f "$CK" ] || { echo "missing $CK — train first"; exit 1; }
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_mix${PMIX}_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --sample-size 500 --output-dir "$O" $X
    done
done

echo "==================== MIXRATIO-DONE ===================="
for d in $P/fs_mixp25_l12* $P/fs_mixp75_l12* $P/fs_mixp100_l12*; do
    echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
done
