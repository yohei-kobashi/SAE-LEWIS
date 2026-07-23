#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

# AxBench SAE/SAE-A steering reproduction — GENERATION (GPU).
# Batch:      cd ~/SAE-LEWIS && qsub run_axbench_repro.sh   (login node)
# Interactive: qsub -I -l select=1 -W group_list=go25 -q interact-g
#              then: source start_gpu_nodes.sh && cd SAE-LEWIS && git pull
#                    && bash run_axbench_repro.sh
# Env knobs: CONFIG=prod_2b_l10_v1 LAYER=10 N_CONCEPTS=100
# Repo-faithful: AlpacaEval 10 instr/concept (their sampler), 14-factor
# grid, addition steering h + f*max_act*W_dec, temp 1.0, 128 tok. Arms:
# sae (vanilla latent), sae_a (AUROC argmax on their labeled eval data),
# ll_set10 (LinguaLens mechanics, cross-protocol).
# Resumable — rerun this script to extend past a walltime kill.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

CONFIG=${CONFIG:-prod_2b_l20_v1}
LAYER=${LAYER:-20}
N_CONCEPTS=${N_CONCEPTS:-100}
AX=third_party/axbench

# 2026-07-17: the v1 prompt path (chat template decoded to a string and
# re-tokenized) silently produced EMPTY prompts on the GPU env — every
# generation was unconditioned babble and the judge scored 0.000 across
# all arms. The records and judge cache from that run are POISON for the
# resume logic (same keys), so quarantine them once.
REC=runs/axbench_repro/records_${CONFIG}.jsonl
if [ -f "$REC" ] && [ ! -f "$REC.broken-v1" ]; then
    mv "$REC" "$REC.broken-v1"
    for JC in runs/axbench_repro/judge_cache_${CONFIG}_*.jsonl; do
        [ -f "$JC" ] && mv "$JC" "$JC.broken-v1"
    done
    echo "[axrepro] quarantined broken v1 records + judge cache"
fi

# vendored test data: clone once, fetch alpaca_eval.json once
if [ ! -d "$AX" ]; then
    mkdir -p third_party
    git clone --depth 1 https://github.com/stanfordnlp/axbench "$AX"
fi
[ -f "$AX/alpaca_eval.json" ] || \
    wget -q -O "$AX/alpaca_eval.json" \
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"

python scripts/eval_axbench_repro_gen.py \
    --output-dir runs/axbench_repro \
    --axbench-dir "$AX" \
    --config "$CONFIG" --layer "$LAYER" \
    --num-concepts "$N_CONCEPTS" \
    --device cuda

echo "==================== AXBENCH-REPRO GENERATION DONE ===================="
echo "next (prepost): CONFIG=$CONFIG bash run_axbench_repro_judge.sh"
