#!/bin/bash -l

# p100 second extension to s10000 (user 2026-07-25). Same rule:
# dev-score s8500..10000; if best <= s8000 incumbent (dev 0.4250) ->
# stop at s8000; else eval500 the best new step.
# Run inside interact-g: bash run_adapt10k.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json
OUT=$P/eflm_l12_mixft_p100

if [ ! -f "$OUT/eflm-step10000.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $V4/t4_ctx_l12_v2 \
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
        --max-steps 10000 --resume --device cuda
fi

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for ST in 8500 9000 9500 10000; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/mixftdev_p100_s$ST$DIRX
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
inc, _ = cell("mixftdev_p100_s8000")
best = None
for st in (8500, 9000, 9500, 10000):
    try:
        net, rmax = cell(f"mixftdev_p100_s{st}")
    except Exception:
        continue
    feas = rmax <= refr + 0.015 + 1e-9
    print(f"#  p100 s{st}: net={net:.4f} rmax={rmax:.4f} feas={feas} "
          f"(s8000 inc {inc:.4f})", file=sys.stderr)
    if feas and net > inc and (best is None or net > best[1]):
        best = (st, net)
print(best[0] if best else "STOP8000")
PY
)
echo "[adapt10k] verdict: $SEL"
if [ "$SEL" != "STOP8000" ]; then
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_p100s10kbest_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$SEL.pt" \
            --sample-size 500 --output-dir "$O" $X
    done
    for d in $P/fs_p100s10kbest_l12 $P/fs_p100s10kbest_l12_amp; do
        echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
    done
fi
echo "==================== ADAPT10K-DONE ===================="
