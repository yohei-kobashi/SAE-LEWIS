#!/bin/bash
# FIC judge pass 2 (prepost polling): (a) calibrated-steer records
# (fs_steer_l{L}[_amp] -> fic_fs_steer_*) so the FIC steer rows use each
# arm's own scale; (b) re-judge fic_fs_l12 / _amp whose clamp rows came
# out empty (0/0/0) in pass 1. Judge caches make reruns cheap.

cd ~/SAE-LEWIS
source env-c/bin/activate

set -o pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

P=runs/prod_gemma_v6
CELLS=(
  "fs_steer_l4       -                 -                fic_fs_steer_l4"
  "fs_steer_l4_amp   -                 -                fic_fs_steer_l4_amp"
  "fs_steer_l12      -                 -                fic_fs_steer_l12"
  "fs_steer_l12_amp  -                 -                fic_fs_steer_l12_amp"
  "fs_steer_l20      -                 -                fic_fs_steer_l20"
  "fs_steer_l20_amp  -                 -                fic_fs_steer_l20_amp"
  "fs_probe_l12      fs_clamp_l12      a3prime_edit     fic_fs_l12"
  "fs_probe_l12_amp  fs_clampE_l12_amp amp_a3prime_l12  fic_fs_l12_amp"
)

while true; do
    git pull -q || true
    done_n=0
    for CFG in "${CELLS[@]}"; do
        set -- $CFG
        PROBE=$P/$1/records.jsonl
        CLAMP=$([ "$2" = "-" ] || echo $P/$2/records.jsonl)
        A3=$([ "$3" = "-" ] || echo $P/$3/records.jsonl)
        OUT=$P/$4
        [ -f "$PROBE" ] || continue
        echo "==== judging $1 -> $4 ===="
        if python scripts/eval_fic_judge.py \
              --repeat-probe500 "$PROBE" \
              $([ -n "$CLAMP" ] && [ -f "$CLAMP" ] && echo --repeat-clamp "$CLAMP") \
              $([ -n "$A3" ] && [ -f "$A3" ] && echo --repeat-a3 "$A3") \
              --dir-map runs/tables/lingualens_dirmap_en.json \
              --output-dir "$OUT" | tee /tmp/ficfs2_last.log \
           && grep -q "FIC-JUDGE-DONE" /tmp/ficfs2_last.log; then
            if [ -z "$CLAMP" ] || grep -qE "^\| clamp \| .*[0-9]" "$OUT/report.md"; then
                done_n=$((done_n+1))
            fi
        fi
    done
    echo "[fic-fs2] complete cells: $done_n / ${#CELLS[@]}"
    if [ "$done_n" -eq "${#CELLS[@]}" ]; then
        echo "==================== FIC-FS2-DONE ===================="
        break
    fi
    sleep 900
done
