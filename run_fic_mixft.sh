#!/bin/bash
# FIC judge for the mixft winner -> fic_mixft_l12{,_amp}
cd ~/SAE-LEWIS
source env-c/bin/activate
set -o pipefail
git pull || true
[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
P=runs/prod_gemma_v6
for SUF in "" "_amp"; do
    for TAG in mixft mixft2 mixft3 adapt_s4k p100s6kbest p100s8kbest; do
        SRC=$P/fs_${TAG}_l12$SUF/records.jsonl
        OUT=$P/fic_${TAG}_l12$SUF
        [ -f "$SRC" ] || continue
        mkdir -p "$OUT"
        [ -f "$OUT/records_merged.jsonl" ] || python scripts/merge_ef_records.py \
            --base $P/fs_probe_l12$SUF/records.jsonl --ef "$SRC" \
            --out "$OUT/records_merged.jsonl"
        [ -f "$OUT/report.md" ] || python scripts/eval_fic_judge.py \
            --repeat-probe500 "$OUT/records_merged.jsonl" --output-dir "$OUT"
    done
done
echo "==================== FIC-MIXFT-DONE ===================="
grep -E "^\| ef \|" $P/fic_mixft*_l12*/report.md 2>/dev/null
