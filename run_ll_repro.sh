#!/bin/bash
# LinguaLens intervention-evaluation reproduction — GENERATION (GPU).
# Run inside interact-g:
#   qsub -I -l select=1 -W group_list=go25 -q interact-g
#   source start_gpu_nodes.sh && cd SAE-LEWIS && git pull && bash run_ll_repro.sh
#
# Repo-faithful (Intervener.run_intervention_experiment): set 0/10 + full
# recon replacement, prompt_only=False, raw prompt (no chat template),
# temperature 1.0, max_new_tokens 100. 4 anchor features x (control 50 +
# targeted 3x2x50 + random 2x50) = 1800 sampled generations.
# Judging happens separately on prepost: run_ll_repro_judge.sh
set -eo pipefail
cd "$(dirname "$0")"

FRC=runs/frc/identified_l12_16k_r3.json
[ -f "$FRC" ] || { echo "missing $FRC (produced by identify_features_frc.py"\
  " — same file run_protocol_e2e.sh used)"; exit 1; }

python scripts/eval_ll_repro_gen.py \
    --output-dir runs/ll_repro \
    --frc-sets "$FRC" \
    --features past_tense,linking_verb,politeness,metaphor \
    --num-experiments 50 \
    --device cuda

echo "==================== LL-REPRO GENERATION DONE ===================="
echo "next (prepost): bash run_ll_repro_judge.sh"
