# LinguaLens end-to-end evaluation

dataset: `THU-KEG/LinguaLens-Data` (language=English), sample=500 (seed=42), scored=500, enumeration-skipped=0

intervention: diff-based, scope=local blocklist=yes steer_lambda=1 k_amp=64 k_sup=64 pool_topk=64; l_max=3 ins_threshold=0.9; refine_passes=3 (recompute)

| condition | exact_match | copy_rate | sim_target | sim_source | edit_loc_iou | sae_shift | bleu | chrf |
|---|---|---|---|---|---|---|---|---|
| true | 0.1120 | 0.6280 | 0.6687 | 0.8853 | 0.2961 | 0.0001 | — | — |
| empty | 0.0020 | 1.0000 | 0.6124 | 1.0000 | 0.0020 | 0.0000 | — | — |
| input-copy baseline | 0.0020 | 1.0000 | 0.6124 | 1.0000 | 0.0020 | 0.0000 | — | — |

Reading guide:
- `sim_target` must beat the input-copy baseline for the system to be editing toward the reference at all.
- `Δ(true − empty)` on sim_target / sae_shift is the end-to-end conditioning-causality signal (the ranker's sae_align term participates here, unlike §13.5's probes).
- high `copy_rate` = the tagger proposed nothing or the ranker preferred the unedited candidate.
