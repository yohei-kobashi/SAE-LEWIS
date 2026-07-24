#!/bin/bash -l

# Mix-ratio completion (user 2026-07-24): add p75 and extend p25 to the
# same 4k budget so the ratio ablation compares equal-budget best-ckpts.
# Joint E' selection across all (ratio, step) cells; eval500 only if the
# global best beats the current incumbent's dev net.
# Run inside interact-g: bash run_mixft3.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json
T4C=$V4/t4_ctx_l12_v2

if [ ! -f $V4/mixft_p75_l12/build.done ]; then
    MIX=$V4/mixft_p75_l12
    mkdir -p $MIX
    cp $T4C/meta.json $MIX/meta.json
    for f in $V4/corruption_zctx_l12/shard-*.jsonl.gz; do
        ln -sf "$(readlink -f $f)" "$MIX/$(basename $f)"
    done
    for f in $T4C/shard-*.jsonl.gz; do
        for i in $(seq 1 91); do
            ln -sf "$(readlink -f $f)" \
                "$MIX/$(basename ${f%.jsonl.gz})-dup$i.jsonl.gz"
        done
    done
    touch $MIX/build.done
fi

train_to_4k () {  # $1 = pmix tag
    local OUT=$P/eflm_l12_mixft_$1
    [ -f "$OUT/eflm-step4000.pt" ] && return 0
    python train_ef_editor.py \
        --corruption-dir $V4/mixft_$1_l12 \
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
        --max-steps 4000 --resume --device cuda
}
train_to_4k p75
train_to_4k p25

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for PMIX in p75 p25; do
    for ST in 500 1000 1500 2000 2500 3000 3500 4000; do
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
rows, best, inc = [], None, 0.0
for pm in ("p25", "p50", "p75"):
    for st in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
        try:
            net, rmax = cell(f"mixftdev_{pm}_s{st}")
        except Exception:
            continue
        feas = rmax <= refr + 0.015 + 1e-9
        rows.append((pm, st, net, rmax, feas))
        if pm == "p50" and st == 2000:
            inc = net
        if feas and (best is None or net > best[2]):
            best = (pm, st, net)
print("# ratio x step dev table (net/rmax/feas):", file=sys.stderr)
for pm, st, net, rmax, feas in rows:
    print(f"#  {pm} s{st}: {net:.4f}/{rmax:.4f}/{feas}", file=sys.stderr)
if best and best[2] > inc and not (best[0] == "p50" and best[1] == 2000):
    print(f"{best[0]} {best[1]}")
else:
    print("KEEP none")
PY
)
echo "[mixft3] global selection: $SEL"
PM=$(echo "$SEL" | awk '{print $1}')
ST=$(echo "$SEL" | awk '{print $2}')
if [ "$PM" != "KEEP" ]; then
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_mixft3_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$P/eflm_l12_mixft_$PM/eflm-step$ST.pt" \
            --sample-size 500 --output-dir "$O" $X
    done
    for d in $P/fs_mixft3_l12 $P/fs_mixft3_l12_amp; do
        echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
    done
fi
echo "==================== MIXFT3-DONE ===================="
