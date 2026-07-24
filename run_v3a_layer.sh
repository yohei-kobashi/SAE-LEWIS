#!/bin/bash -l

# v3a blend rollout to L4/L20 (user 2026-07-24; main row = zero-shot T2,
# blend = adaptation row). Per layer:
#   1. naive T4 adaptation model if missing (L4 exists; L20 trains 4k)
#   2. blends alpha in {0.2,0.3,0.4}
#   3. dev-200 evals + T2 dev reference; RELATIVE selection
#      (rmax <= T2's own dev rmax AND net > T2 dev net)
#   4. eval500 of the selected point -> fs_v3a_final_l{L}{,_amp}
# Run inside interact-g: LAYER=4 bash run_v3a_layer.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

LAYER=${LAYER:?set LAYER=4|20}
case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy; SC=2.5 ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy; SC=2.5 ;;
esac
P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json
T2=$P/eflm_l${LAYER}_v6t2/eflm-final.pt
T4=$P/eflm_l${LAYER}_t4_v6t2
BL=$P/v3a_blends2

# ---- 1. naive T4 model (v1 recipe, mirrors the old stage 5) -------------
T4C=$V4/t4_ctx_l$LAYER
if ! python -c "import json,sys;m=json.load(open('$T4C/meta.json'));sys.exit(0 if 'd_sae' in m else 1)" 2>/dev/null; then
    python scripts/make_t4_cache.py \
        --spec $FS/l${LAYER}_specctx.json --split $SPLIT \
        --out $T4C --scale $SC --meta-from $V4/corruption_zctx_l$LAYER/meta.json
fi
if [ ! -f "$T4/eflm-final.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $T4C \
        --dev-corruption-dir $V4/corruption_seldev_zctx_l$LAYER \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$T4" \
        --inject-layer $LAYER --sae-path "$SAE" \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat --init-ckpt "$T2" \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 4000 --resume --device cuda
fi

EVC=(--frame repeat --feature-spec $FS/l${LAYER}_specctx.json
     --fspec-scale $SC --arms ef
     --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path "$SAE" --sae-layer "$LAYER" --blocklist "$BLK"
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

dev_eval () {
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/v3adev_l${LAYER}_$2$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$1" \
            --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
    done
}
dev_eval "$T2" t2ref
for A in 0.2 0.3 0.4; do
    CK=$BL/l${LAYER}_v1final_a$A.pt
    [ -f "$CK" ] || python scripts/blend_ckpt.py \
        --a "$T2" --b "$T4/eflm-final.pt" --alpha $A --out "$CK"
    dev_eval "$CK" a$A
done

SEL=$(python - "$P" "$LAYER" <<'PY'
import re, sys
P, L = sys.argv[1], sys.argv[2]
def cell(tag):
    nets, rands = [], []
    for suf in ("", "_amp"):
        t = open(f"{P}/v3adev_l{L}_{tag}{suf}/report.md").read()
        tr = float(re.search(r"\| true \| ef \| ([0-9.]+)", t).group(1))
        rd = float(re.search(r"\| random \| ef \| ([0-9.]+)", t).group(1))
        nets.append(tr - rd); rands.append(rd)
    return sum(nets) / 2, max(rands)
ref, refr = cell("t2ref")
best = None
for a in ("0.2", "0.3", "0.4"):
    try:
        net, rmax = cell(f"a{a}")
    except Exception:
        continue
    print(f"#  L{L} a={a}: net={net:.4f} rmax={rmax:.4f} "
          f"(T2 ref {ref:.4f}/{refr:.4f})", file=sys.stderr)
    if rmax <= refr + 1e-9 and net > ref and (best is None or net > best[1]):
        best = (a, net)
print(best[0] if best else "NONE")
PY
)
echo "[v3aL] L$LAYER selected alpha: $SEL"
if [ "$SEL" != "NONE" ]; then
    CK=$BL/l${LAYER}_v1final_a$SEL.pt
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_v3a_final_l$LAYER$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --sample-size 500 --output-dir "$O" $X
    done
    for d in $P/fs_v3a_final_l$LAYER $P/fs_v3a_final_l${LAYER}_amp; do
        echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
    done
fi
echo "==================== V3AL-L$LAYER-DONE ===================="
