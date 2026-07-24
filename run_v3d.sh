#!/bin/bash -l

# v3d-cond (user 2026-07-24): ADAPT WITHOUT OVERWRITING — the CTX-train
# LoRA + heads are hard-frozen; only the spec-conditioning interface
# (proj_a_corr low-rank correction, type_emb, cond_scale, mag_proj,
# <1M params ~3%) trains on the T4v2 mixed cache (replay + scramble
# nulls, mismatch 0.30). Constrained ckpt selection as in E'
# (relative dev rule, +1SE margin), single eval500 verification.
# Run inside interact-g: bash run_v3d.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json
OUT=$P/eflm_l12_v3d

[ -f $V4/t4v2_mix_l12/build.done ] || { echo "mix cache missing"; exit 1; }

if [ ! -f "$OUT/eflm-step2000.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $V4/t4v2_mix_l12 \
        --dev-corruption-dir $V4/corruption_seldev_zctx_l12 \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$OUT" \
        --inject-layer 12 \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.30 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat \
        --init-ckpt $P/eflm_l12_v6t2/eflm-final.pt \
        --train-only-cond --learning-rate 1e-4 \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 2000 --resume --device cuda
fi

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for ST in 500 1000 1500 2000; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/v3ddev_s$ST$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$ST.pt" \
            --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
    done
done

SEL=$(python - "$P" <<'PY'
import re, sys
P = sys.argv[1]
def cell(base):
    nets, rands = [], []
    for suf in ("", "_amp"):
        t = open(f"{P}/{base}{suf}/report.md").read()
        tr = float(re.search(r"\| true \| ef \| ([0-9.]+)", t).group(1))
        rd = float(re.search(r"\| random \| ef \| ([0-9.]+)", t).group(1))
        nets.append(tr - rd); rands.append(rd)
    return sum(nets) / 2, max(rands)
ref, refr = cell("v3adev_t2ref")
best = None
for st in (500, 1000, 1500, 2000):
    try:
        net, rmax = cell(f"v3ddev_s{st}")
    except Exception:
        continue
    feas = rmax <= refr + 0.015 + 1e-9
    print(f"#  s{st}: net={net:.4f} rmax={rmax:.4f} feas={feas} "
          f"(ZS ref {ref:.4f}/{refr:.4f})", file=sys.stderr)
    if feas and net > ref and (best is None or net > best[1]):
        best = (st, net)
print(best[0] if best else "NONE")
PY
)
echo "[v3d] selected step: $SEL"
if [ "$SEL" != "NONE" ]; then
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_v3d_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$SEL.pt" \
            --sample-size 500 --output-dir "$O" $X
    done
    for d in $P/fs_v3d_l12 $P/fs_v3d_l12_amp; do
        echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
    done
fi
echo "==================== V3D-DONE ===================="
