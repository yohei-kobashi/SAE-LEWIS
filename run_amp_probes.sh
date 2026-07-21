#!/bin/bash
# amp-direction (reversed-pair: the edit ADDS the phenomenon) evaluation
# at L12 for ALL axes (user 2026-07-21: 全ての評価軸でamp/sup両方).
# GPU stages, all guarded — safe to rerun after walltime kills.
# ef ckpt = nobudget champion. sup-direction counterparts are the existing
# probe500 / clamp_baseline500 / a3prime_edit / klc_* runs.

set -eo pipefail

P=runs/prod_gemma_v6
L2V=runs/mcgill_gemma_repro_3k/final
SAE=layer_12/width_16k/average_l0_82/params.npz
BLK=runs/blocklist/blocklist.npy
CKPT=$P/eflm_l12_v5f_nobudget/eflm-final.pt

BASE=(--frame repeat --reverse-pairs --llm2vec-dir "$L2V" --sae-path "$SAE"
      --sae-layer 12 --blocklist "$BLK" --k-amp 64 --k-sup 64
      --steer-alpha 0.5 --sample-size 500 --device cuda)

# A4 ef + A2 steer + raw
if [ ! -f $P/amp_probe_l12_nb/report.md ]; then
    python scripts/eval_ef_bare.py "${BASE[@]}" \
        --ef-ckpt "$CKPT" --arms ef,steer,raw \
        --output-dir $P/amp_probe_l12_nb
fi
# A3' prompting_edit (instruction verb flips to "add")
if [ ! -f $P/amp_a3prime_l12/report.md ]; then
    python scripts/eval_ef_bare.py "${BASE[@]}" \
        --arms prompting_edit \
        --a3-prompts runs/a3_prompts/steering_prompts.json \
        --output-dir $P/amp_a3prime_l12
fi
# A1 clamp (value sweep, same as clamp_baseline500)
if [ ! -f $P/amp_clamp_l12/report.md ]; then
    python scripts/eval_clamp_baseline.py --reverse-pairs \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --sample-size 500 \
        --output-dir $P/amp_clamp_l12 --device cuda
fi
# KL/NLL, amp direction, three arms
if [ ! -f $P/amp_klc_ef_l12/report.md ]; then
    python scripts/eval_kl_consistency.py --reverse-pairs \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --sample-size 500 --arms ef \
        --ef-ckpt "$CKPT" --output-dir $P/amp_klc_ef_l12 --device cuda
fi
if [ ! -f $P/amp_klc_steer_l12/report.md ]; then
    python scripts/eval_kl_consistency.py --reverse-pairs \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --sample-size 500 --arms steer \
        --output-dir $P/amp_klc_steer_l12 --device cuda
fi
if [ ! -f $P/amp_klc_clamp_l12/report.md ]; then
    python scripts/eval_kl_consistency.py --reverse-pairs \
        --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
        --blocklist "$BLK" --sample-size 500 --arms clamp \
        --output-dir $P/amp_klc_clamp_l12 --device cuda
fi

echo "==================== AMP-PROBES-DONE ===================="
