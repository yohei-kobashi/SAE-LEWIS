#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=0:30:00
#PBS -W group_list=go25
#PBS -j oe

# One-shot debug for the AxBench L20 all-zero judge result: no-hook
# baseline vs factor 0.2 / 1.0 steering on concept 0. Read the job's
# .o file for the verdict:
#   * NOHOOK off-topic too -> generation frame broken (not steering).
#   * NOHOOK on-topic, f0.2 off-topic -> steering scale destroys the
#     prompt; recheck AxBench's magnitude formula against their repo.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true
python scripts/debug_axbench_gen.py
echo "==================== AXBENCH DEBUG DONE ===================="
