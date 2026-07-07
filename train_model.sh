#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

RUN=./runs/prod_gemma_v4
LLM2VEC=runs/mcgill_gemma_repro_3k/final

# 0. corruption 生成 + 選択用/報告用 dev の分離
#    corruption_dev/meta.json を分割完了マーカーにする: 分割後に
#    corruption_parallel.sh を再実行すると移動済み shard を worker が
#    作り直して重複するため、分割済みなら丸ごとスキップ。
if [ ! -f "$RUN/corruption_dev/meta.json" ]; then
    WORKERS=6 \
    OUT_DIR=$RUN/corruption \
    LLM2VEC_DIR=$LLM2VEC \
    CORRUPTION_SAMPLES=500000 CORRUPTION_SHARD=2000 \
    BLOCKLIST=runs/blocklist/blocklist.npy \
    bash scripts/corruption_parallel.sh

    # 選択用(seldev: 学習中の best-checkpoint 選択)/ 報告用(dev)を分離
    mkdir -p "$RUN/corruption_seldev" "$RUN/corruption_dev"
    mv "$RUN/corruption/shard-w0-00001.jsonl.gz" "$RUN/corruption_seldev/"
    for i in 1 2 3 4 5; do
        mv "$RUN/corruption/shard-w$i-00001.jsonl.gz" "$RUN/corruption_dev/"
    done
    cp "$RUN/corruption/meta.json" "$RUN/corruption_seldev/"
    cp "$RUN/corruption/meta.json" "$RUN/corruption_dev/"
fi

# 1. dev 監視付き学習(*-final.pt = best dev 状態)
#    run_production.sh 自体が段ごとに resume/skip するので再投入安全。
RUN_DIR=$RUN \
EDITOR_STEPS=100000 TAGGER_STEPS=30000 \
DEV_CORRUPTION_DIR=$RUN/corruption_seldev \
LLM2VEC_DIR=$LLM2VEC \
SIMCSE_DIR=$LLM2VEC \
bash scripts/run_production.sh

# 2. 評価3本(報告用 dev + LinguaLens)
#    eval_tagger_editor / measure_editor_ceiling は --sae-path を取らない
#    (チェックポイント内の Proj_A を使う)。eval_lingualens のみ
#    intervention 用に SAE をロードするので l0_82 を明示。
python eval_tagger_editor.py \
    --corruption-dir "$RUN/corruption_dev" \
    --llm2vec-dir "$LLM2VEC" \
    --tagger-ckpt "$RUN/tagger/tagger-final.pt" \
    --editor-ckpt "$RUN/editor/editor-final.pt" \
    --output-dir "$RUN/eval_dev" \
    --max-samples 2000 \
    --device cuda

python scripts/measure_editor_ceiling.py \
    --corruption-dir "$RUN/corruption_dev" \
    --llm2vec-dir "$LLM2VEC" \
    --editor-ckpt "$RUN/editor/editor-final.pt" \
    --output-dir "$RUN/ceiling" \
    --max-samples 2000 \
    --device cuda

python eval_lingualens.py \
    --llm2vec-dir "$LLM2VEC" \
    --tagger-ckpt "$RUN/tagger/tagger-final.pt" \
    --editor-ckpt "$RUN/editor/editor-final.pt" \
    --output-dir "$RUN/eval_lingualens" \
    --sae-path layer_12/width_16k/average_l0_82/params.npz \
    --sample-size 100 \
    --dump-details \
    --device cuda
