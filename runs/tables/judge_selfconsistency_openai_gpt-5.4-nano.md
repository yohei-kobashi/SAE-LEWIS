# Judge self-consistency — openai_gpt-5.4-nano

On an exact-match pair the system judgment judge(src, out) IS the gold judgment judge(src, tgt) (same feature, same strings), so a self-consistent judge scores realized=True by construction. FRR on that subset therefore measures the judge against itself, ranking judges as instruments with no human labels. It is NOT a threshold for aggregate FRR gaps: noise independent across ~980 pairs averages out of the mean and attenuates real gaps toward chance rather than inventing them (use scripts/frr_paired_test.py to test a gap).

| system | exact pairs | self-consistency [95% CI] | flips | non-exact pairs | FRR (non-exact) | gold-indecisive |
|---|---|---|---|---|---|---|
| ef32 | 216 | 0.8843 [0.835, 0.920] | 25 | 762 | 0.7349 | 19 |
| routed | 271 | 0.8893 [0.846, 0.921] | 30 | 707 | 0.6733 | 19 |
| steer | 223 | 0.8610 [0.809, 0.900] | 31 | 755 | 0.6424 | 19 |

**Pooled self-consistency: 0.8789 [0.853, 0.901] (n=710, flips=86)** — per-judgment disagreement-with-itself 0.121.

Pooled FRR on the non-exact subset: 0.6839 (n=2224) — where the judge does real work.

Reading: self-consistency well below 1.0 means the judge contradicts itself on identical comparisons under presentation-order randomization, and cannot be trusted to resolve the harder non-exact pairs. Compare ACROSS judges to pick the primary — the most self-consistent judge is the least attenuating instrument, which is why it should also show the LARGEST between-system separation. Do not compare this rate against a between-system gap to decide whether the gap is real; that is what scripts/frr_paired_test.py is for.
