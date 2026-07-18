#!/bin/bash
# T5 — LinguaLens再現の見直し(EF_LM_LOSS_PLAN §4b)。
# interact-g で実行(2hセッション、全段resumable — 切れたら再実行):
#   printf 'cd SAE-LEWIS && git pull && bash run_t5_ll_repro.sh\nexit\n' \
#     | ssh -tt miyabi 'qsub -I -l select=1 -W group_list=go25 -q interact-g'
# 段: (1) FRC同定 L4/L20(L12は既存) (2) D.1再現gen(修正版decode、
#     プロンプト込み全文)を L12/L4/L20 で。judgeは別途 prepost で
#     scripts/judge_ll_repro.py --run-dir runs/ll_repro_v2_l{L}。

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

SAE_L4="layer_4/width_16k/average_l0_60/params.npz"
SAE_L12="layer_12/width_16k/average_l0_82/params.npz"
SAE_L20="layer_20/width_16k/average_l0_71/params.npz"

# (1) FRC identification for the new layers (leakage-fixed recipe).
if [ ! -f runs/frc/identified_l4_16k_r3.json ]; then
    python scripts/identify_features_frc.py \
        --out runs/frc/identified_l4_16k_r3.json \
        --sae-path "$SAE_L4" --sae-layer 4 --top-r 16 --device cuda
fi
if [ ! -f runs/frc/identified_l20_16k_r3.json ]; then
    python scripts/identify_features_frc.py \
        --out runs/frc/identified_l20_16k_r3.json \
        --sae-path "$SAE_L20" --sae-layer 20 --top-r 16 --device cuda
fi

# (2) D.1 anchor reproduction, fixed decode (prompt-inclusive), original
#     settings (n=50, set 10/0, temp 1.0, 100tok). Per-layer dirs; the
#     gen script resumes from its own records.
python scripts/eval_ll_repro_gen.py --output-dir runs/ll_repro_v2_l12 \
    --sae-path "$SAE_L12" --sae-layer 12 \
    --frc-sets runs/frc/identified_l12_16k_r3.json --device cuda
python scripts/eval_ll_repro_gen.py --output-dir runs/ll_repro_v2_l4 \
    --sae-path "$SAE_L4" --sae-layer 4 \
    --frc-sets runs/frc/identified_l4_16k_r3.json --device cuda
python scripts/eval_ll_repro_gen.py --output-dir runs/ll_repro_v2_l20 \
    --sae-path "$SAE_L20" --sae-layer 20 \
    --frc-sets runs/frc/identified_l20_16k_r3.json --device cuda

echo "==================== T5 GEN DONE ===================="
echo "next (prepost): for L in 12 4 20; do python scripts/judge_ll_repro.py \\"
echo "  --run-dir runs/ll_repro_v2_l\$L; done"
