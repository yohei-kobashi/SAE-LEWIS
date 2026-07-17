#!/bin/bash
# C: editable/uneditable decision tree (CPU, prepost — records live on miyabi)
#   qsub -I -l select=1 -W group_list=go25 -q prepost
#   cd SAE-LEWIS && git pull && source env-c/bin/activate && bash run_feature_tree.sh
# Two variants: paper (⚫ EF-free — C class keyed on P-I WHERE) and internal
# (EF columns kept as a diagnostic). Both land in runs/tables/.
set -eo pipefail
cd "$(dirname "$0")"

python scripts/build_feature_tree.py --paper-mode \
    --out runs/tables/feature_tree_paper
python scripts/build_feature_tree.py \
    --out runs/tables/feature_tree

echo "==================== FEATURE TREE DONE ===================="
echo "commit runs/tables/feature_tree*.{md,csv} (tables are git-included)"
