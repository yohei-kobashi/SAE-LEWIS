#!/bin/bash
# FIC judging (prepost) for the REPEAT-FRAME baseline arms (9n pulled
# forward): clampset (LinguaLens-faithful) + axbsteer (AxBench-faithful),
# 3 layers x both directions, from run_bl_repeat.sh records.
# Fresh judge dirs fic_bl_l{L}[_amp] avoid cache-key collisions with the
# rewrite-era clamp/axbench judgments; ef/steer/prompting judgments are
# SEEDED from the old caches (same keys, unchanged outputs) so only the
# two new arms cost API calls.
# Run inside prepost: bash run_bl_judge.sh

cd ~/SAE-LEWIS
source env-c/bin/activate

set -o pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

P=runs/prod_gemma_v6
ALL_OK=1
for L in 4 12 20; do
    for SUF in "" "_amp"; do
        PROBE=$P/fs_probe_l$L$SUF/records.jsonl
        CLAMP=$P/bl_clamp_l$L$SUF/records.jsonl
        AXB=$P/bl_axb_l$L$SUF/records.jsonl
        OUT=$P/fic_bl_l$L$SUF
        if [ ! -f "$CLAMP" ] || [ ! -f "$AXB" ]; then
            echo "[bljudge] records missing for L$L$SUF — skip"
            ALL_OK=0; continue
        fi
        mkdir -p "$OUT"
        # seed: copy old judgments EXCEPT the rewrite-era clamp/axbench
        # rows (same keys would collide with the new repeat outputs)
        if [ ! -f "$OUT/.seeded" ]; then
            for SRC in $P/fic_fs_l$L$SUF $P/fic_fs_axb_l$L$SUF; do
                [ -f "$SRC/judge_cache_gpt-4o.jsonl" ] && \
                python - "$SRC/judge_cache_gpt-4o.jsonl" \
                         "$OUT/judge_cache_gpt-4o.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
seen = set()
try:
    for l in open(dst):
        try:
            seen.add(json.loads(l)["key"])
        except Exception:
            pass
except OSError:
    pass
with open(dst, "a") as f:
    n = 0
    for l in open(src):
        try:
            c = json.loads(l)
        except Exception:
            continue
        arm = c["key"].split("|")[4]
        if arm in ("clamp", "axbench"):
            continue
        if c["key"] in seen:
            continue
        seen.add(c["key"])
        f.write(json.dumps(c) + "\n")
        n += 1
    print(f"[seed] {src} -> +{n}")
PY
            done
            touch "$OUT/.seeded"
        fi
        if [ ! -f "$OUT/report.md" ]; then
            python scripts/eval_fic_judge.py \
                --repeat-probe500 "$PROBE" \
                --repeat-clamp "$CLAMP" --clamp-key clampset \
                --repeat-axb "$AXB" --axb-key axbsteer \
                --output-dir "$OUT" || ALL_OK=0
        fi
    done
done

if [ "$ALL_OK" = "1" ]; then
    echo "==================== BL-JUDGE-DONE ===================="
    for d in $P/fic_bl_l*; do
        echo "--- $d"; sed -n '1,25p' $d/report.md 2>/dev/null
    done
else
    echo "[bljudge] incomplete (records or judge failures) — rerun me"
fi
