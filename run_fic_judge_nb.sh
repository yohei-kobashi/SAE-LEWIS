#!/bin/bash
# FIC re-judge for the nobudget champion family (user 2026-07-21 (3):
# 学習が終わったモデルから順次FIC再評価 — prepost).
# Polls inside the session: judges whatever probe500 records exist,
# sleeps and retries until all six are judged (walltime kills are fine —
# the chain driver relaunches me; judge caches make reruns cheap).

cd ~/SAE-LEWIS
source env-c/bin/activate

set -o pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

P=runs/prod_gemma_v6
PAIRS=(
  "eflm_l12_v5f_nobudget      fic_judge_nb_l12"
  "eflm_l12_v5f_nobudget_80k  fic_judge_nb_l12_80k"
  "eflm_l4_v5f2_nobudget      fic_judge_nb_l4"
  "eflm_l4_v5f2_nobudget_80k  fic_judge_nb_l4_80k"
  "eflm_l20_v5f2_nobudget     fic_judge_nb_l20"
  "eflm_l20_v5f2_nobudget_80k fic_judge_nb_l20_80k"
)

while true; do
    git pull -q || true
    done_n=0
    for CFG in "${PAIRS[@]}"; do
        set -- $CFG
        REC=$P/$1/probe500/records.jsonl
        [ -f "$REC" ] || continue
        echo "==== judging $1 -> $2 ===="
        if python scripts/eval_fic_judge.py \
              --repeat-probe500 "$REC" \
              --dir-map runs/tables/lingualens_dirmap_en.json \
              --output-dir "$P/$2" | tee /tmp/ficnb_last.log \
           && grep -q "FIC-JUDGE-DONE" /tmp/ficnb_last.log; then
            done_n=$((done_n+1))
        fi
    done
    echo "[fic-nb] complete runs: $done_n / ${#PAIRS[@]}"
    if [ "$done_n" -eq "${#PAIRS[@]}" ]; then
        echo "==================== FIC-NB-ALL-DONE ===================="
        break
    fi
    sleep 900
done
