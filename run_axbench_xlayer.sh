#!/bin/bash
# AxBenchプロトコルの層拡張(L4/L12、ユーザー指示 2026-07-18)。
# 彼らのconcept500資産(概念+ラベル付きlatent_eval_data)はL10/L20のみ
# なので、L4/L12は「彼らのSAE-A方式(全latentのAUROC argmax選択)を
# 当該層のSAEに適用」する = sae_a腕のみ(vanilla/ll_set10は概念↔latent
# 対応が層固有のため定義不能)。概念・指示・factor格子・judgeはL20
# configのものを流用。論文表記: L10/L20=厳密再現、L4/L12=プロトコル拡張。
# interact-gで: LAYER=4 bash run_axbench_xlayer.sh(2h、resume対応)

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

LAYER=${LAYER:?set LAYER=4 or 12}
case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz" ;;
    12) SAE="layer_12/width_16k/average_l0_82/params.npz" ;;
    *)  echo "L10/L20 use run_axbench_repro.sh (strict repro)"; exit 1 ;;
esac

python scripts/eval_axbench_repro_gen.py \
    --output-dir "runs/axbench_repro_l$LAYER" \
    --axbench-dir third_party/axbench \
    --config prod_2b_l20_v1 --layer "$LAYER" --sae-path "$SAE" \
    --arms sae_a --num-concepts 100 --device cuda

echo "==================== AXBENCH XLAYER L$LAYER GEN DONE ===================="
echo "judge: RUN_DIR=runs/axbench_repro_l$LAYER CONFIG=prod_2b_l20_v1 ..."
