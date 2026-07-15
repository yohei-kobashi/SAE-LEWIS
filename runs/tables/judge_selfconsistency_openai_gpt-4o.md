# Judge self-consistency — openai_gpt-4o

On an exact-match pair the system judgment judge(src, out) IS the gold judgment judge(src, tgt) (same feature, same strings), so a self-consistent judge scores realized=True by construction. FRR on that subset therefore measures the judge against itself, ranking judges as instruments with no human labels. It is NOT a threshold for aggregate FRR gaps: noise independent across ~980 pairs averages out of the mean and attenuates real gaps toward chance rather than inventing them (use scripts/frr_paired_test.py to test a gap).

| system | exact pairs | self-consistency [95% CI] | flips | non-exact pairs | FRR (non-exact) | gold-indecisive |
|---|---|---|---|---|---|---|
| ef32 | 215 | 0.9860 [0.960, 0.995] | 3 | 765 | 0.8418 | 17 |
| routed | 271 | 0.9926 [0.973, 0.998] | 2 | 709 | 0.7687 | 17 |
| steer | 228 | 0.9781 [0.950, 0.991] | 5 | 752 | 0.6582 | 17 |

**Pooled self-consistency: 0.9860 [0.974, 0.992] (n=714, flips=10)** — per-judgment disagreement-with-itself 0.014.

Pooled FRR on the non-exact subset: 0.7565 (n=2226) — where the judge does real work.

Reading: self-consistency well below 1.0 means the judge contradicts itself on identical comparisons under presentation-order randomization, and cannot be trusted to resolve the harder non-exact pairs. Compare ACROSS judges to pick the primary — the most self-consistent judge is the least attenuating instrument, which is why it should also show the LARGEST between-system separation. Do not compare this rate against a between-system gap to decide whether the gap is real; that is what scripts/frr_paired_test.py is for.
