#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# 9n (pulled forward, user 2026-07-23): the LinguaLens-faithful and
# AxBench-faithful baseline arms measured IN THE REPEAT FRAME, L4/L12/L20,
# both directions, with strength dev-selection on the v2 dev section
# (eval500 untouched). One short-g job:
#   per layer:  A. dev grids (dev-200, true only)
#                  clampset enhancement SET value in {2,5,10,20}
#                  axbsteer factor in {0.2,0.6,1,2,4} x both directions
#               B. eval500 (true,random) at the selected strengths
#                  bl_clamp_l$L{,_amp} / bl_axb_l$L{,_amp}
# FIC judging runs on prepost afterwards (separate script).
# Batch: qsub -N blrep run_bl_repeat.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

FS=runs/feature_specs
P=runs/prod_gemma_v6
SPLIT=runs/tables/eval_split.json
LLM2VEC=runs/mcgill_gemma_repro_3k/final

best_of () {  # $1 = arm, $2.. = report dirs; prints dir with max true exact
    python - "$@" <<'PY'
import re, sys
arm = sys.argv[1]
best, bv = None, -1.0
for d in sys.argv[2:]:
    try:
        t = open(f"{d}/report.md").read()
    except OSError:
        continue
    m = re.search(rf"\| true \| {arm} \| ([0-9.]+)", t)
    if m and float(m.group(1)) > bv:
        bv, best = float(m.group(1)), d
print(best or "", bv)
PY
}

for LAYER in 4 12 20; do
    case "$LAYER" in
        4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
            BLK=runs/blocklist_l4/blocklist.npy ;;
        12) SAE="layer_12/width_16k/average_l0_82/params.npz"
            BLK=runs/blocklist/blocklist.npy ;;
        20) SAE="layer_20/width_16k/average_l0_71/params.npz"
            BLK=runs/blocklist_l20/blocklist.npy ;;
    esac
    FRC=$FS/l${LAYER}_frc_r3.json
    AUR=$FS/l${LAYER}_auroc_r1.json
    MX=$FS/l${LAYER}_auroc_r1_maxact.json
    [ -f "$MX" ] || python scripts/extract_pool_maxact.py \
        --pairs $FS/l${LAYER}_pairs.jsonl --sets "$AUR" \
        --split "$SPLIT" --out "$MX"

    COMMON=(--frame repeat --llm2vec-dir "$LLM2VEC" --sae-path "$SAE"
            --sae-layer "$LAYER" --blocklist "$BLK" --device cuda)

    # ---- A. dev grids (dev section, 200 pairs, true only) --------------
    for V in 2 5 10 20; do
        O=$P/bl_dev/clamp_enh_l${LAYER}_v$V
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${COMMON[@]}" --arms clampset --conditions true \
            --fsets "$FRC" --clamp-value "$V" --reverse-pairs \
            --pool-dev "$SPLIT" --sample-size 200 --output-dir "$O"
    done
    for F in 0.2 0.6 1 2 4; do
        O=$P/bl_dev/axb_abl_l${LAYER}_f$F
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${COMMON[@]}" --arms axbsteer --conditions true \
            --fsets "$AUR" --axb-factor "$F" \
            --pool-dev "$SPLIT" --sample-size 200 --output-dir "$O"
        O=$P/bl_dev/axb_enh_l${LAYER}_f$F
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${COMMON[@]}" --arms axbsteer --conditions true \
            --fsets "$AUR" --fsets-maxact "$MX" --axb-factor "$F" \
            --reverse-pairs \
            --pool-dev "$SPLIT" --sample-size 200 --output-dir "$O"
    done

    read VB _ < <(best_of clampset \
        $P/bl_dev/clamp_enh_l${LAYER}_v{2,5,10,20})
    VSEL=${VB##*_v}; VSEL=${VSEL:-10}
    read FB _ < <(best_of axbsteer \
        $P/bl_dev/axb_abl_l${LAYER}_f{0.2,0.6,1,2,4})
    FSELA=${FB##*_f}; FSELA=${FSELA:-1}
    read FB2 _ < <(best_of axbsteer \
        $P/bl_dev/axb_enh_l${LAYER}_f{0.2,0.6,1,2,4})
    FSELE=${FB2##*_f}; FSELE=${FSELE:-1}
    echo "[blrep] L$LAYER selected: clamp_enh V=$VSEL " \
         "axb_abl f=$FSELA axb_enh f=$FSELE"

    # ---- B. eval500, true+random, selected strengths -------------------
    [ -f $P/bl_clamp_l$LAYER/report.md ] || python scripts/eval_ef_bare.py \
        "${COMMON[@]}" --arms clampset --conditions true,random \
        --fsets "$FRC" --sample-size 500 \
        --output-dir $P/bl_clamp_l$LAYER
    [ -f $P/bl_clamp_l${LAYER}_amp/report.md ] || \
        python scripts/eval_ef_bare.py \
        "${COMMON[@]}" --arms clampset --conditions true,random \
        --fsets "$FRC" --clamp-value "$VSEL" --reverse-pairs \
        --sample-size 500 --output-dir $P/bl_clamp_l${LAYER}_amp
    [ -f $P/bl_axb_l$LAYER/report.md ] || python scripts/eval_ef_bare.py \
        "${COMMON[@]}" --arms axbsteer --conditions true,random \
        --fsets "$AUR" --axb-factor "$FSELA" --sample-size 500 \
        --output-dir $P/bl_axb_l$LAYER
    [ -f $P/bl_axb_l${LAYER}_amp/report.md ] || \
        python scripts/eval_ef_bare.py \
        "${COMMON[@]}" --arms axbsteer --conditions true,random \
        --fsets "$AUR" --fsets-maxact "$MX" --axb-factor "$FSELE" \
        --reverse-pairs --sample-size 500 \
        --output-dir $P/bl_axb_l${LAYER}_amp
done

echo "==================== BL-REPEAT-DONE ===================="
for d in $P/bl_clamp_l* $P/bl_axb_l*; do
    echo "--- $d"; grep -E "^\| (true|random) \|" $d/report.md || true
done
