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

# fs_steer records lack the empty/raw frame-control (steer-only runs);
# raw is arm-independent (no-intervention greedy) -> graft it from the
# matching fs_probe records by pair idx.
python3 - <<'PYMERGE'
import json, os
P = "runs/prod_gemma_v6"
for L in (4, 12, 20):
    for sfx in ("", "_amp"):
        src = f"{P}/fs_steer_l{L}{sfx}/records.jsonl"
        ref = f"{P}/fs_probe_l{L}{sfx}/records.jsonl"
        out = f"{P}/fs_steer_l{L}{sfx}/records_merged.jsonl"
        if not (os.path.exists(src) and os.path.exists(ref)):
            continue
        if os.path.exists(out):
            continue
        raws = {}
        for l in open(ref):
            if l.strip():
                r = json.loads(l)
                t = r["outputs"].get("empty", {}).get("raw")
                if t:
                    raws[int(r["idx"])] = t
        n = 0
        with open(out, "w") as f:
            for l in open(src):
                if not l.strip():
                    continue
                r = json.loads(l)
                rw = raws.get(int(r["idx"]))
                if rw:
                    r["outputs"].setdefault("empty", {})["raw"] = rw
                    n += 1
                f.write(json.dumps(r) + "\n")
        print(f"[merge] {out}: {n} raw refs grafted")
PYMERGE

CELLS=(
  "fs_steer_l4/records_merged       -  -  fic_fs_steer_l4"
  "fs_steer_l4_amp/records_merged   -  -  fic_fs_steer_l4_amp"
  "fs_steer_l12/records_merged      -  -  fic_fs_steer_l12"
  "fs_steer_l12_amp/records_merged  -  -  fic_fs_steer_l12_amp"
  "fs_steer_l20/records_merged      -  -  fic_fs_steer_l20"
  "fs_steer_l20_amp/records_merged  -  -  fic_fs_steer_l20_amp"
  "fs_probe_l12      fs_clamp_l12      a3prime_edit     fic_fs_l12"
  "fs_probe_l12_amp  fs_clampE_l12_amp amp_a3prime_l12  fic_fs_l12_amp"
)

while true; do
    git pull -q || true
    done_n=0
    for CFG in "${CELLS[@]}"; do
        set -- $CFG
        case "$1" in
          */records_merged) PROBE=$P/${1%/records_merged}/records_merged.jsonl ;;
          *) PROBE=$P/$1/records.jsonl ;;
        esac
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
            ok=1
            [ -n "$CLAMP" ] && ! grep -qE "^\| clamp \| .*[0-9]" "$OUT/report.md" && ok=0
            case "$4" in fic_fs_steer_*)
                grep -qE "^\| steer \| .*[0-9]" "$OUT/report.md" || ok=0 ;;
            esac
            [ "$ok" = 1 ] && done_n=$((done_n+1))
        fi
    done
    echo "[fic-fs2] complete cells: $done_n / ${#CELLS[@]}"
    if [ "$done_n" -eq "${#CELLS[@]}" ]; then
        echo "==================== FIC-FS2-DONE ===================="
        break
    fi
    sleep 900
done
