#!/bin/bash -l

# Final adaptation runs (user 2026-07-25: drop the mix-ratio framing —
# the adaptation row is the DEFENDED FT on the train section alone
# (p100: scramble negatives + LR 1e-4 + E' selection)).
#   1. eval500 of p100 s4000 (dev-best so far, 0.320)
#   2. extend p75 and p100 to s6000; dev-score s4500..6000
#   3. eval500 any step that beats its s4000 incumbent on dev
# Run inside interact-g: bash run_adapt_final.sh

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

for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/fs_adapt_s4k_l12$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${EVC[@]}" --ef-ckpt $P/eflm_l12_mixft_p100/eflm-step4000.pt \
        --sample-size 500 --output-dir "$O" $X
done

ext_to_6k () {
    local OUT=$P/eflm_l12_mixft_$1
    [ -f "$OUT/eflm-step6000.pt" ] && return 0
    python train_ef_editor.py \
        --corruption-dir $2 \
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
}
ext_to_6k p100 $V4/t4_ctx_l12_v2
ext_to_6k p75  $V4/mixft_p75_l12

for PMIX in p100 p75; do
    for ST in 4500 5000 5500 6000; do
        for DIRX in "" "_amp"; do
            if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
            O=$P/mixftdev_${PMIX}_s$ST$DIRX
            [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
                "${EVC[@]}" --ef-ckpt \
                "$P/eflm_l12_mixft_$PMIX/eflm-step$ST.pt" \
                --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
        done
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
picks = []
for pm in ("p100", "p75"):
    inc, _ = cell(f"mixftdev_{pm}_s4000")
    best = None
    for st in (4500, 5000, 5500, 6000):
        try:
            net, rmax = cell(f"mixftdev_{pm}_s{st}")
        except Exception:
            continue
        feas = rmax <= refr + 0.015 + 1e-9
        print(f"#  {pm} s{st}: net={net:.4f} rmax={rmax:.4f} feas={feas} "
              f"(s4000 inc {inc:.4f})", file=sys.stderr)
        if feas and net > inc and (best is None or net > best[1]):
            best = (st, net)
    picks.append(f"{pm}:{best[0] if best else 'STOP4000'}")
print(" ".join(picks))
PY
)
echo "[adapt] extension verdicts: $SEL"
for PICK in $SEL; do
    PM=${PICK%%:*}; ST=${PICK##*:}
    if [ "$ST" != "STOP4000" ]; then
        for DIRX in "" "_amp"; do
            if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
            O=$P/fs_${PM}s6kbest_l12$DIRX
            [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
                "${EVC[@]}" --ef-ckpt \
                "$P/eflm_l12_mixft_$PM/eflm-step$ST.pt" \
                --sample-size 500 --output-dir "$O" $X
        done
    fi
done
echo "==================== ADAPT-FINAL-DONE ===================="
for d in $P/fs_adapt_s4k_l12* $P/fs_p100s6kbest_l12* $P/fs_p75s6kbest_l12*; do
    [ -f "$d/report.md" ] && { echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md; }
done
