#!/bin/bash -l

# v3a stage 2 (2026-07-24): the dev-200 absolute threshold (0.02) was
# miscalibrated — zero-shot T2 itself shows rmax 0.045 on that sample.
# Verdict moves to eval500 directly for the top blends:
#   v1final a=0.4  (dev net 0.2175, rmax 0.050 ~ T2's own 0.045)
#   v2s1500 a=0.4  (dev net 0.1450, rmax 0.040 < T2's 0.045)
# Accept iff eval500 random <= 0.02 per direction AND net > T2 (0.142/0.140).
# Run inside interact-g: bash run_v3a2.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
BL=$P/v3a_blends2

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for TAG in v1final_a0.4 v2s1500_a0.4; do
    CK=$BL/$TAG.pt
    [ -f "$CK" ] || { echo "missing blend $CK"; exit 1; }
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_v3a_${TAG//./}_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --sample-size 500 --output-dir "$O" $X
    done
done

echo "==================== V3A2-DONE ===================="
for d in $P/fs_v3a_v1final_a04_l12* $P/fs_v3a_v2s1500_a04_l12*; do
    echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
done
