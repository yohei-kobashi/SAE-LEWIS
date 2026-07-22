#!/bin/bash
# retrieval-spec sup eval (improvement A, direction check). interact-g.
cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail
git pull || true
P=runs/prod_gemma_v6
FS=runs/feature_specs
if [ ! -f $P/fs_retr_l12/report.md ]; then
    python scripts/eval_ef_bare.py \
        --frame repeat --feature-spec $FS/l12_spec.json \
        --fspec-scale 3.5 --fspec-retrieve $FS/l12_retrieve.json \
        --retrieve-m 5 --conditions true,random --arms ef \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
        --k-amp 64 --k-sup 64 \
        --ef-ckpt $P/eflm_l12_v5f_nobudget/eflm-final.pt \
        --sample-size 500 --device cuda --output-dir $P/fs_retr_l12
fi
echo "==================== RETR-SUP-DONE ===================="
