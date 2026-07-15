# Main edit-metrics table (literature-standard)

pairs: 499; condition true; SARI = set-based n1-4 single-ref (keep/add F1 + del P); BLEU/chrF vs reference, self-BLEU vs source (LEWIS-style); 'oracle' = per-pair best exact across systems (Pass@K analog); 'routed' = count-rule T=1 (ef32 if own hunks<=T else steer)

| system | exact | SARI | sim | BLEU | chrF | selfBLEU |
|---|---|---|---|---|---|---|
| ef32 | 0.2104 | 63.09 | 0.5803 | nan | nan | nan |
| ef64 | 0.1904 | 60.92 | 0.6192 | nan | nan | nan |
| steer | 0.2405 | 61.10 | 0.6120 | nan | nan | nan |
| clamp | 0.1743 | 53.52 | 0.5276 | nan | nan | nan |
| prompt | 0.1242 | 53.61 | 0.6058 | nan | nan | nan |
| pipeline | 0.1102 | 40.99 | 0.6681 | nan | nan | nan |
| routed | 0.2786 | 66.36 | 0.6517 | nan | nan | nan |
| oracle | 0.5210 | 83.24 | 0.8313 | nan | nan | nan |

## Per-phenomenon exact (features with n >= 8; full table in the CSV)

| feature | n | routed | ef32 | steer | pipeline |
|---|---|---|---|---|---|
| adjectival_suffix | 11 | 0.545 | 0.545 | 0.000 | 0.182 |
| split_infinitives | 11 | 0.273 | 0.000 | 0.364 | 0.000 |
| degree_prefix | 10 | 0.300 | 0.300 | 0.100 | 0.000 |
| emphatic_structure | 10 | 0.700 | 0.700 | 0.400 | 0.500 |
| transitional | 10 | 0.300 | 0.000 | 0.500 | 0.000 |
| active_verbs | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| commisive | 9 | 0.111 | 0.111 | 0.111 | 0.000 |
| future_progressive | 9 | 0.222 | 0.222 | 0.222 | 0.000 |
| appositives | 8 | 0.000 | 0.000 | 0.000 | 0.000 |
| cleft_sentences | 8 | 0.500 | 0.000 | 0.625 | 0.000 |
| deixis | 8 | 0.125 | 0.125 | 0.000 | 0.000 |
| deontic | 8 | 0.750 | 0.625 | 0.375 | 0.250 |
| expressive | 8 | 0.625 | 0.375 | 0.375 | 0.000 |
| future | 8 | 0.250 | 0.250 | 0.500 | 0.250 |
| static_dynamic | 8 | 0.375 | 0.250 | 0.500 | 0.000 |
| third_person_singular | 8 | 0.500 | 0.500 | 0.000 | 0.000 |
| turn_taking | 8 | 0.125 | 0.125 | 0.250 | 0.125 |
