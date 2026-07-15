# Judge self-consistency — hf_google_gemma-2-9b-it

On an exact-match pair the system judgment judge(src, out) IS the gold judgment judge(src, tgt) (same feature, same strings), so a self-consistent judge scores realized=True by construction. FRR on that subset therefore measures the judge against itself, ranking judges as instruments with no human labels. It is NOT a threshold for aggregate FRR gaps: noise independent across ~980 pairs averages out of the mean and attenuates real gaps toward chance rather than inventing them (use scripts/frr_paired_test.py to test a gap).

| system | exact pairs | self-consistency [95% CI] | flips | non-exact pairs | FRR (non-exact) | gold-indecisive |
|---|---|---|---|---|---|---|
| ef32 | 213 | 0.9718 [0.940, 0.987] | 6 | 770 | 0.7390 | 14 |
| routed | 269 | 0.9740 [0.947, 0.987] | 7 | 714 | 0.6919 | 14 |
| steer | 225 | 0.9689 [0.937, 0.985] | 7 | 758 | 0.6728 | 14 |

**Pooled self-consistency: 0.9717 [0.957, 0.982] (n=707, flips=20)** — per-judgment disagreement-with-itself 0.028.

Pooled FRR on the non-exact subset: 0.7016 (n=2242) — where the judge does real work.

Reading: self-consistency well below 1.0 means the judge contradicts itself on identical comparisons under presentation-order randomization, and cannot be trusted to resolve the harder non-exact pairs. Compare ACROSS judges to pick the primary — the most self-consistent judge is the least attenuating instrument, which is why it should also show the LARGEST between-system separation. Do not compare this rate against a between-system gap to decide whether the gap is real; that is what scripts/frr_paired_test.py is for.
