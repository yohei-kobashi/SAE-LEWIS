#!/bin/bash -l

# (user 2026-07-24/25) Ratio table at s4000 + p75 extension probe:
#   1. eval500 at s4000 for p25 and p50 (training already done to 4k)
#   2. continue p75 to 6000; dev-score s4500..6000
#   3. rule: if best new dev net <= s4000's dev net (0.2975) -> stop at
#      s4000 (no eval). Else eval500 the best new step.
# Run inside interact-g: bash run_mixft5.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

# ---- 1. s4000 ratio evals ------------------------------------------------
for PMIX in p25 p50 p100; do
    CK=$P/eflm_l12_mixft_$PMIX/eflm-step4000.pt
    [ -f "$CK" ] || { echo "missing $CK"; exit 1; }
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_mix${PMIX}s4k_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --sample-size 500 --output-dir "$O" $X
    done
done

# ---- 2. p75 -> 6000 ------------------------------------------------------
OUT=$P/eflm_l12_mixft_p75
if [ ! -f "$OUT/eflm-step6000.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $V4/mixft_p75_l12 \
        --dev-corruption-dir $V4/corruption_seldev_zctx_l12 \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$OUT" \
        --inject-layer 12 \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat \
        --init-ckpt $P/eflm_l12_v6t2/eflm-final.pt \
        --learning-rate 1e-4 --backbone-lr 1e-4 \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 6000 --resume --device cuda
fi
for ST in 4500 5000 5500 6000; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/mixftdev_p75_s$ST$DIRX
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
inc, _ = cell("mixftdev_p75_s4000")
best = None
for st in (4500, 5000, 5500, 6000):
    try:
        net, rmax = cell(f"mixftdev_p75_s{st}")
    except Exception:
        continue
    feas = rmax <= refr + 0.015 + 1e-9
    print(f"#  p75 s{st}: net={net:.4f} rmax={rmax:.4f} feas={feas} "
          f"(s4000 incumbent {inc:.4f})", file=sys.stderr)
    if feas and net > inc and (best is None or net > best[1]):
        best = (st, net)
print(best[0] if best else "STOP4000")
PY
)
echo "[mixft5] p75 extension verdict: $SEL"
if [ "$SEL" != "STOP4000" ]; then
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_mixft5_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$SEL.pt" \
            --sample-size 500 --output-dir "$O" $X
    done
fi
echo "==================== MIXFT5-DONE ===================="
for d in $P/fs_mixp25s4k_l12* $P/fs_mixp50s4k_l12* $P/fs_mixp100s4k_l12* $P/fs_mixft5_l12*; do
    [ -f "$d/report.md" ] && { echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md; }
done
