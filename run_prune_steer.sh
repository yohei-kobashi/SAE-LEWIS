#!/bin/bash
# B-2: P-O intervention version + S_min-stable-core vs FRC top-3 comparison.
#   qsub -I -l select=1 -W group_list=go25 -q interact-g
#   source start_gpu_nodes.sh && cd SAE-LEWIS && git pull && bash run_prune_steer.sh
#
# Stage 1 (GPU): prune the steer-exact pairs' specs to minimal intervention
#   sets S_min (prune_spec.py --effector steer; spec rebuilt with the same
#   k=64/64 the steer baseline ran with — step-0 guard rejects drift).
#   Resumable per pair; rerun to extend past a 2h session.
# Stage 2 (CPU): per-phenomenon stable core (features surviving in >=50% of
#   the phenomenon's S_min sets) vs LinguaLens FRC top-3 — the licensed
#   form (ii) of the "feature-corresponding activations" claim.
set -eo pipefail
cd "$(dirname "$0")"

EXPL=runs/np_explanations/gemma-2-2b_12-res-16k.json
EXPL_ARG=""
[ -f "$EXPL" ] && EXPL_ARG="--explanations $EXPL"

python scripts/prune_spec.py --effector steer \
    --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
    --exact-from runs/prod_gemma_v6/steer_baseline500/records.jsonl:steer0.5 \
    --k-amp 64 --k-sup 64 \
    $EXPL_ARG \
    --output-dir runs/prod_gemma_v6/prune_spec_steer \
    --max-pairs 120 \
    --device cuda

python scripts/compare_smin_frc.py \
    --smin-records runs/prod_gemma_v6/prune_spec_steer/records.jsonl \
    --out runs/tables/smin_vs_frc

echo "==================== PRUNE-STEER + SMIN-VS-FRC DONE ===================="
echo "commit runs/tables/smin_vs_frc.md (tables are git-included)"
