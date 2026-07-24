#!/bin/bash -l

# Minimal ablations of the ADOPTED method under the FINAL protocol
# (user 2026-07-24). L12, (7) ctx spec scale 3.5, eval500, both dirs.
#   A. -EF backbone   (nb_noS3 ckpt)      B. -contrastive teachers (nb_noctr)
#   C. mean->top-3 spec truncation (LL-style selection width)
#   D. k=64 -> 8 at eval
# The T2/(7) 2x2 factorial + steer + scale-1.0 cells already exist.
# Run inside interact-g: bash run_ablation_final.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs

# C: top-3 truncated ctx spec (magnitude kept, rescale target kept)
[ -f $FS/l12_specctx_top3.json ] || python - <<PY
import json
spec = json.loads(open("$FS/l12_specctx.json").read())
out = {}
for ph, fs in spec.items():
    items = sorted(fs["spec"].items(), key=lambda kv: -abs(kv[1]))[:3]
    out[ph] = dict(fs, spec={k: v for k, v in items})
open("$FS/l12_specctx_top3.json", "w").write(json.dumps(out))
print(f"[top3] {len(out)} features")
PY

COMMON=(--frame repeat --fspec-scale 3.5 --conditions true,random --arms ef
    --llm2vec-dir runs/mcgill_gemma_repro_3k/final
    --sae-path layer_12/width_16k/average_l0_82/params.npz
    --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
    --sample-size 500 --device cuda)

run2 () {  # $1 outbase  $2.. extra args
    local OB=$1; shift
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/$OB$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${COMMON[@]}" "$@" --output-dir "$O" $X
    done
}

run2 abl_noS3_l12  --feature-spec $FS/l12_specctx.json --k-amp 64 --k-sup 64 \
    --ef-ckpt $P/eflm_l12_nb_noS3/eflm-final.pt
run2 abl_noctr_l12 --feature-spec $FS/l12_specctx.json --k-amp 64 --k-sup 64 \
    --ef-ckpt $P/eflm_l12_nb_noctr/eflm-final.pt
run2 abl_top3_l12  --feature-spec $FS/l12_specctx_top3.json --k-amp 64 --k-sup 64 \
    --ef-ckpt $P/eflm_l12_v6t2/eflm-final.pt
run2 abl_k8_l12    --feature-spec $FS/l12_specctx.json --k-amp 8 --k-sup 8 \
    --ef-ckpt $P/eflm_l12_v6t2/eflm-final.pt

echo "==================== ABL-FINAL-DONE ===================="
for d in $P/abl_noS3_l12* $P/abl_noctr_l12* $P/abl_top3_l12* $P/abl_k8_l12*; do
    echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
done
