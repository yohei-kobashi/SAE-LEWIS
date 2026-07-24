#!/bin/bash
# FIC judge for the A3'-luna prompting row (user 2026-07-25):
# luna-authored edit instructions executed by frozen gemma
# (pe_luna_l12{,_amp}). Judge scores ONLY the prompting arm — the base
# probe records are stripped to their empty/raw references.
# Run on prepost: bash run_fic_peluna.sh
cd ~/SAE-LEWIS
source env-c/bin/activate
set -o pipefail
git pull || true
[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
P=runs/prod_gemma_v6
for SUF in "" "_amp"; do
    SRC=$P/pe_luna_l12$SUF/records.jsonl
    OUT=$P/fic_peluna_l12$SUF
    [ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
    mkdir -p "$OUT"
    [ -f "$OUT/records_refonly.jsonl" ] || python - \
        "$P/fs_probe_l12$SUF/records.jsonl" "$OUT/records_refonly.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w") as w:
    for line in open(src):
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r["outputs"].get("empty", {}).get("raw")
        if raw is None:
            continue
        r["outputs"] = {"empty": {"raw": raw}}
        w.write(json.dumps(r) + "\n")
print(f"[refonly] {dst}")
PY
    [ -f "$OUT/report.md" ] || python scripts/eval_fic_judge.py \
        --repeat-probe500 "$OUT/records_refonly.jsonl" \
        --repeat-a3 "$SRC" --output-dir "$OUT"
done
echo "==================== FIC-PELUNA-DONE ===================="
grep -E "^\| prompting \|" $P/fic_peluna_l12*/report.md 2>/dev/null
