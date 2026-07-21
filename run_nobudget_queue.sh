#!/bin/bash
# Nobudget-champion queue for interact-g (user decision 2026-07-21:
# nobudget = new champion; exact優先).
# Stages (all guarded / resumable; the chain driver reruns me until
# NB-QUEUE-DONE):
#   S1  L12 nobudget 40k -> 80k extension (seeded resume, two-phase,
#       same procedure as the v5f 80k extensions)
#   S2  pick winner steps by true-ef exact (40k=0.3166 vs 80k)
#   S3  noS3 ablation under the nobudget recipe, SAME two-phase procedure
#   S4  noctr ablation under the nobudget recipe, SAME two-phase procedure

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
S3CKPT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt

seed80 () {  # seed80 <40k-dir> <80k-dir>
    mkdir -p "$2"
    cp -n "$1/eflm-step40000.pt" "$1/eflm-step40000.state.pt" \
          "$1/best.json" "$2/" 2>/dev/null || true
}

exact_of () {  # exact_of <probe500-report>
    python3 - "$1" <<'PY'
import sys
for l in open(sys.argv[1]):
    c = [x.strip() for x in l.strip().strip('|').split('|')]
    if len(c) >= 3 and c[0] == 'true' and c[1] == 'ef':
        print(c[2]); break
PY
}

# ---- S1: L12 nobudget 80k extension -----------------------------------
if [ ! -f $P/eflm_l12_v5f_nobudget_80k/probe500/report.md ]; then
    seed80 $P/eflm_l12_v5f_nobudget $P/eflm_l12_v5f_nobudget_80k
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        FLOW_INIT=$S3CKPT NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        OUT_SUFFIX=_v5f_nobudget_80k MAX_STEPS=80000 bash run_ef_editor.sh
fi

# ---- S2: winner steps --------------------------------------------------
E40=$(exact_of $P/eflm_l12_v5f_nobudget/probe500/report.md)
E80=$(exact_of $P/eflm_l12_v5f_nobudget_80k/probe500/report.md)
W=$(python3 -c "print(80000 if float('$E80') > float('$E40') else 40000)")
echo "NB winner: steps=$W (40k=$E40, 80k=$E80)"

# ---- S2b: amp-direction probes, all axes at L12 (user 2026-07-21:
# ---- 全評価軸でamp/sup両方; cheap ~2.5h, runs before the long ablations)
bash run_amp_probes.sh

# ---- S3: noS3 under nobudget (two-phase mirrors the champion) ----------
if [ ! -f $P/eflm_l12_nb_noS3/probe500/report.md ]; then
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        OUT_SUFFIX=_nb_noS3 MAX_STEPS=40000 bash run_ef_editor.sh
fi
if [ "$W" = "80000" ] && [ ! -f $P/eflm_l12_nb_noS3_80k/probe500/report.md ]; then
    seed80 $P/eflm_l12_nb_noS3 $P/eflm_l12_nb_noS3_80k
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        OUT_SUFFIX=_nb_noS3_80k MAX_STEPS=80000 bash run_ef_editor.sh
fi

# ---- S4: noctr under nobudget (two-phase mirrors the champion) ---------
if [ ! -f $P/eflm_l12_nb_noctr/probe500/report.md ]; then
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        FLOW_INIT=$S3CKPT MM_PROB=0.0 EMPTY_PROB=0.0 \
        NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        OUT_SUFFIX=_nb_noctr MAX_STEPS=40000 bash run_ef_editor.sh
fi
if [ "$W" = "80000" ] && [ ! -f $P/eflm_l12_nb_noctr_80k/probe500/report.md ]; then
    seed80 $P/eflm_l12_nb_noctr $P/eflm_l12_nb_noctr_80k
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        FLOW_INIT=$S3CKPT MM_PROB=0.0 EMPTY_PROB=0.0 \
        NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        OUT_SUFFIX=_nb_noctr_80k MAX_STEPS=80000 bash run_ef_editor.sh
fi

echo "==================== NB-QUEUE-DONE ===================="
