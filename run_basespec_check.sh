#!/bin/bash -l

# User question 2026-07-23: could T2/T3 be better WITHOUT the in-context
# spec (7)? Pure eval check (no training): T3 and T2 evaluated with the
# BASE bare-sentence spec (l12_spec.json, dev scale 3.5 = same as ctx),
# both directions. ~4 x 5min on interact-g.
#   bash run_basespec_check.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
for CK in v6t3 v6t2; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
        O=$P/fs_${CK}base_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --fspec-scale 3.5 --conditions true,random --arms ef \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_$CK/eflm-final.pt \
            --sample-size 500 --device cuda \
            --output-dir $O $EXTRA
    done
done

echo "==================== BASESPEC-DONE ===================="
for d in $P/fs_v6t3base_l12* $P/fs_v6t2base_l12*; do
    echo "--- $d"; grep -E "^\| (true|random) \|" $d/report.md || true
done
