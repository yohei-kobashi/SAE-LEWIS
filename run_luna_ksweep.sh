#!/bin/bash -l

# (user 2026-07-25) Two GPU stages, both guarded:
#   1. A3'-luna: gemma executes luna-AUTHORED per-feature edit
#      instructions (runs/a3_prompts/edit_prompts_luna.json, scp'd from
#      local) — the proper "newest prompt author" upgrade of the
#      prompting row. eval500 both directions.
#   2. Ours-AD k-sweep: p100 s8000 ckpt, k in {1..128}, eval500 both
#      directions + overall/4-category aggregation (same protocol as
#      run_ksweep_final.sh for Ours-ZS).
# Run inside interact-g: bash run_luna_ksweep.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs

# ---- 1. A3'-luna prompting (executor = frozen gemma-2-2b-it) ------------
[ -f runs/a3_prompts/edit_prompts_luna.json ] || {
    echo "missing edit_prompts_luna.json"; exit 1; }
PE=(--frame repeat --llm2vec-dir runs/mcgill_gemma_repro_3k/final
    --sae-path layer_12/width_16k/average_l0_82/params.npz
    --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
    --k-amp 64 --k-sup 64 --sample-size 500 --device cuda
    --arms prompting_edit
    --a3-prompts runs/a3_prompts/steering_prompts.json
    --a3-edit-prompts runs/a3_prompts/edit_prompts_luna.json)
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/pe_luna_l12$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${PE[@]}" --output-dir "$O" $X
done
echo "---- pe_luna results ----"
grep -E "^\| (true|random) \| prompting_edit" $P/pe_luna_l12*/report.md || true

# ---- 2. Ours-AD (p100 s8000) k-sweep ------------------------------------
COMMON=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
    --conditions true,random --arms ef
    --llm2vec-dir runs/mcgill_gemma_repro_3k/final
    --sae-path layer_12/width_16k/average_l0_82/params.npz
    --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
    --ef-ckpt $P/eflm_l12_mixft_p100/eflm-step8000.pt
    --sample-size 500 --device cuda)

for K in 1 2 4 8 16 32 64 128; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/kswad_l12_k$K$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${COMMON[@]}" --k-amp $K --k-sup $K --output-dir "$O" $X
    done
done

python - <<'PY'
import json, re
from collections import defaultdict
CATMAP = json.loads(open("runs/tables/feature_categories_en.json").read())
rows, crows = [], []
for K in (1, 2, 4, 8, 16, 32, 64, 128):
    row = {"k": K}
    crow = {"k": K}
    for suf, d in (("", "abl"), ("_amp", "enh")):
        t = open(f"runs/prod_gemma_v6/kswad_l12_k{K}{suf}/report.md").read()
        tr = float(re.search(r"\| true \| ef \| ([0-9.]+)", t).group(1))
        rd = float(re.search(r"\| random \| ef \| ([0-9.]+)", t).group(1))
        row[f"{d}_true"], row[f"{d}_rand"], row[f"{d}_net"] = tr, rd, tr - rd
        agg = defaultdict(lambda: [0, 0, 0])
        for line in open(f"runs/prod_gemma_v6/kswad_l12_k{K}{suf}/records.jsonl"):
            r = json.loads(line)
            c = CATMAP.get(r.get("feature") or "?", "?")
            tt = r["outputs"].get("true", {}).get("ef", {}).get("text")
            rr = r["outputs"].get("random", {}).get("ef", {}).get("text")
            if tt is None or rr is None: continue
            agg[c][0] += 1
            agg[c][1] += tt.strip() == r["tgt"].strip()
            agg[c][2] += rr.strip() == r["tgt"].strip()
        for c, (n, a, b2) in agg.items():
            crow[f"{d}_{c}_net"] = (a - b2) / n
            crow[f"{d}_{c}_n"] = n
    rows.append(row); crows.append(crow)
open("runs/tables/ksweep_ad.json", "w").write(json.dumps(rows))
open("runs/tables/ksweep_ad_cat.json", "w").write(json.dumps(crows))
print("| k | abl true | abl rand | abl net | enh true | enh rand | enh net |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['k']} | {r['abl_true']:.4f} | {r['abl_rand']:.4f} | "
          f"{r['abl_net']:.4f} | {r['enh_true']:.4f} | {r['enh_rand']:.4f} | "
          f"{r['enh_net']:.4f} |")
PY

echo "==================== LUNA-KSWEEP-DONE ===================="
