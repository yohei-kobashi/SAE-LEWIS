#!/bin/bash -l

# Mixed-data fine-tune from Ours-ZS (user 2026-07-24: T2からの追加学習、
# 破損データ混合は忘却防止). Minimal grid: ONLY the T4 mix ratio moves
# (p in {25%, 50%}); LR 1e-4 (midpoint of the v1/v2 failure bracket),
# mismatch 0.12 (adopted recipe), scramble nulls on, 2k steps with
# E'-style constrained ckpt selection; winner only -> eval500.
# Run inside interact-g: bash run_mixft.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json
T4C=$V4/t4_ctx_l12_v2      # both-direction rows + scramble nulls (built)

build_mix () {  # $1 = dup count, $2 = out dir
    local MIX=$2
    [ -f $MIX/build.done ] && return 0
    mkdir -p $MIX
    cp $T4C/meta.json $MIX/meta.json
    for f in $V4/corruption_zctx_l12/shard-*.jsonl.gz; do
        ln -sf "$(readlink -f $f)" "$MIX/$(basename $f)"
    done
    for f in $T4C/shard-*.jsonl.gz; do
        for i in $(seq 1 $1); do
            ln -sf "$(readlink -f $f)" \
                "$MIX/$(basename ${f%.jsonl.gz})-dup$i.jsonl.gz"
        done
    done
    touch $MIX/build.done
}
build_mix 10 $V4/mixft_p25_l12     # ~99k/399k = 25%
build_mix 30 $V4/mixft_p50_l12     # ~297k/597k = 50%

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for PMIX in p25 p50; do
    OUT=$P/eflm_l12_mixft_$PMIX
    if [ ! -f "$OUT/eflm-step2000.pt" ]; then
        python train_ef_editor.py \
            --corruption-dir $V4/mixft_${PMIX}_l12 \
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
            --max-steps 2000 --resume --device cuda
    fi
    for ST in 500 1000 1500 2000; do
        for DIRX in "" "_amp"; do
            if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
            O=$P/mixftdev_${PMIX}_s$ST$DIRX
            [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
                "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$ST.pt" \
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
best = None
for pm in ("p25", "p50"):
    for st in (500, 1000, 1500, 2000):
        try:
            net, rmax = cell(f"mixftdev_{pm}_s{st}")
        except Exception:
            continue
        feas = rmax <= refr + 0.015 + 1e-9
        print(f"#  {pm} s{st}: net={net:.4f} rmax={rmax:.4f} feas={feas} "
              f"(ZS {ref:.4f}/{refr:.4f})", file=sys.stderr)
        if feas and net > ref and (best is None or net > best[2]):
            best = (pm, st, net)
print(f"{best[0]} {best[1]}" if best else "NONE none")
PY
)
echo "[mixft] selected: $SEL"
PM=$(echo "$SEL" | awk '{print $1}')
ST=$(echo "$SEL" | awk '{print $2}')
if [ "$PM" != "NONE" ]; then
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_mixft_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$P/eflm_l12_mixft_$PM/eflm-step$ST.pt" \
            --sample-size 500 --output-dir "$O" $X
    done
    for d in $P/fs_mixft_l12 $P/fs_mixft_l12_amp; do
        echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
    done
fi
echo "==================== MIXFT-DONE ===================="
