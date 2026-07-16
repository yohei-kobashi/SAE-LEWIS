#!/bin/bash
# v7 corruption top-up (P4 + lexical variety) — run INSIDE interact-g:
#   qsub -I -l select=1 -W group_list=go25 -q interact-g
#   source start_gpu_nodes.sh && cd SAE-LEWIS && git pull && bash run_corruption_v7.sh
#
# WHAT (ledger: M1 NO-GO verdict, EDIT_FLOWS_ZERO section 5):
#   P4  matched move geometry — new SPLITINF family ("to ADV V" <-> "to V
#       ... ADV"): LinguaLens's split-infinitive geometry, which the cache's
#       ADVPLACE (pre-verbal <-> clause-final, finite verbs) never produced;
#       the M1 pointer learned cache geometry and missed it. Plus a heavier
#       share for the starved reorder families (family-priority-pick).
#   variety — wider MLM payload sampling for the word-corruption share
#       (repl/del top-k 8 -> 24, del-top1 0.5 -> 0.25) and a FRESH seed so
#       the top-up draws different dolma sentences than v5's seed-42 pass.
#   (P5 mismatched-z nulls are TRAIN-TIME — --mismatch-null-prob in
#    train_editflow.py — and need no cache change.)
#
# MECHANICS:
#   * separate out-dir (corruption.py writes meta.json at the END; running
#     into the main dir would clobber the v5 meta). Merge for training by
#     symlinking shards into the main dir as shard-v7-*.jsonl.gz
#     (CorruptionDataset globs shard-*.jsonl.gz) — done by this script once
#     generation has produced shards.
#   * interact-g is capped at 2h: generation runs under `timeout` so the
#     session exits cleanly; shard numbering auto-continues, so RERUNNING
#     THIS SCRIPT EXTENDS the top-up by another --target-samples.
set -eo pipefail
cd "$(dirname "$0")"

OUT=runs/prod_gemma_v4/corruption_v7topup
MAIN=runs/prod_gemma_v4/corruption
TARGET=${TARGET:-6000}          # per-session target; observe the rate, retune
BUDGET=${BUDGET:-5700}          # seconds; leaves ~5min of the 2h for merge

# ---- gate: SPLITINF must behave on real spaCy before any GPU is spent ----
python - <<'PYEOF'
import random, spacy, sys
sys.path.insert(0, ".")
from transforms import propose_splitinf
nlp = spacy.load("en_core_web_sm")
cases = {
 "She wants to boldly go home.":            "SPLIT->POST",
 "He decided to run the race quickly.":     "POST->SPLIT",
 "They plan to carefully review the data.": "SPLIT->POST",
 "She quickly ran home.":                   None,
}
bad = 0
for text, want in cases.items():
    props = propose_splitinf(nlp(text), text, random.Random(0))
    kinds = [p.t_type.split(":")[1] for p in props]
    ok = (want in kinds) if want else (not props)
    print(("  OK   " if ok else "  FAIL ") + f"{text!r} -> "
          + (str([(p.t_type, p.out_text) for p in props]) or "(none)"))
    bad += not ok
if bad:
    sys.exit(f"{bad} SPLITINF case(s) failed — fix before generating")
print("SPLITINF gate passed")
PYEOF

# ---- generation (resumable; timeout keeps us inside the 2h window) -------
set +e
timeout "$BUDGET" python corruption.py \
    --out-dir "$OUT" \
    --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
    --blocklist runs/blocklist/blocklist.npy \
    --target-samples "$TARGET" --samples-per-shard 1000 \
    --transform-prob 0.9 \
    --transform-families "SPLITINF,ADVPLACE,PPFRONT,PARTICLE,DATIVE,QUOTINV,INVERSION,CLEFT" \
    --family-priority-pick \
    --repl-mlm-topk 24 --del-mlm-topk 24 --del-top1-prob 0.25 \
    --seed 777 \
    --device cuda
RC=$?
set -e
[ $RC -eq 124 ] && echo "[v7] BUDGET hit — shards kept; rerun to extend"

# ---- merge: symlink new shards into the training dir ----------------------
echo "==================== v7 TOP-UP MERGE ===================="
n=0
for s in "$OUT"/shard-*.jsonl.gz; do
    [ -e "$s" ] || continue
    base=$(basename "$s" | sed 's/^shard-/shard-v7-/')
    ln -sf "$(readlink -f "$s")" "$MAIN/$base"
    n=$((n+1))
done
[ -f "$OUT/meta.json" ] && cp "$OUT/meta.json" "$MAIN/meta_v7topup.json"
echo "[v7] linked $n shard(s) into $MAIN as shard-v7-*"
echo "[v7] records so far:"
for s in "$OUT"/shard-*.jsonl.gz; do [ -e "$s" ] && zcat "$s" | wc -l; done | paste -sd+ | bc || true
echo
echo "SPLITINF share check (last shard):"
last=$(ls -t "$OUT"/shard-*.jsonl.gz 2>/dev/null | head -1)
[ -n "$last" ] && zcat "$last" | grep -o '"t_family": "[A-Z]*"' | sort | uniq -c | sort -rn | head -8
echo "==================== v7 SESSION DONE ===================="
