# Main edit-metrics table (literature-standard)

pairs: 997; condition true; SARI = set-based n1-4 single-ref (keep/add F1 + del P); BLEU/chrF vs reference, self-BLEU vs source (LEWIS-style); 'oracle' = per-pair best exact across systems (Pass@K analog); 'routed' = count-rule T=1 (ef32 if own hunks<=T else steer)

| system | exact | SARI | sim | BLEU | chrF | selfBLEU |
|---|---|---|---|---|---|---|
| ef32 | 0.2237 | 63.20 | 0.5809 | nan | nan | nan |
| steer | 0.2337 | 58.94 | 0.6016 | nan | nan | nan |
| routed | 0.2839 | 65.88 | 0.6554 | nan | nan | nan |
| oracle | 0.3872 | 74.37 | 0.7262 | nan | nan | nan |

## Per-phenomenon exact (features with n >= 8; full table in the CSV)

| feature | n | routed | ef32 | steer |
|---|---|---|---|---|
| adjectival_suffix | 18 | 0.500 | 0.500 | 0.000 |
| emphatic_structure | 18 | 0.833 | 0.833 | 0.333 |
| active_verbs | 17 | 0.000 | 0.000 | 0.000 |
| commisive | 15 | 0.200 | 0.067 | 0.200 |
| future_progressive | 15 | 0.400 | 0.200 | 0.400 |
| cleft_sentences | 14 | 0.429 | 0.071 | 0.643 |
| elliptical_sentences | 14 | 0.000 | 0.000 | 0.000 |
| past_perfect | 14 | 0.571 | 0.357 | 0.571 |
| synecdoche | 14 | 0.214 | 0.214 | 0.286 |
| third_person_singular | 14 | 0.429 | 0.429 | 0.000 |
| transitional | 14 | 0.357 | 0.000 | 0.571 |
| turn_taking | 14 | 0.214 | 0.214 | 0.143 |
| degree_prefix | 13 | 0.308 | 0.308 | 0.154 |
| directive | 13 | 0.231 | 0.231 | 0.077 |
| mass_noun | 13 | 0.231 | 0.154 | 0.077 |
| spatial_or_directional_prefix | 13 | 0.077 | 0.077 | 0.308 |
| split_infinitives | 13 | 0.308 | 0.077 | 0.385 |
| echo_questions | 12 | 0.000 | 0.000 | 0.000 |
| epistemic | 12 | 0.417 | 0.417 | 0.417 |
| extraposition | 12 | 0.000 | 0.000 | 0.000 |
| future_perfect | 12 | 0.583 | 0.583 | 0.917 |
| nominal_suffix | 12 | 0.250 | 0.250 | 0.000 |
| of_genitive | 12 | 0.250 | 0.167 | 0.667 |
| passive_voice | 12 | 0.583 | 0.000 | 0.583 |
| past_participle_irregular | 12 | 0.000 | 0.000 | 0.000 |
| past_tense_irregular | 12 | 0.417 | 0.417 | 0.083 |
| punctual_durative | 12 | 0.250 | 0.250 | 0.083 |
| referring | 12 | 0.417 | 0.417 | 0.417 |
| relative_clauses | 12 | 0.083 | 0.000 | 0.167 |
| telic_atelic | 12 | 0.083 | 0.083 | 0.167 |
| universal_quantifiers | 12 | 0.000 | 0.000 | 0.000 |
| adverbial_suffix | 11 | 0.091 | 0.091 | 0.000 |
| appositives | 11 | 0.000 | 0.000 | 0.000 |
| deixis | 11 | 0.273 | 0.273 | 0.182 |
| deontic | 11 | 0.818 | 0.727 | 0.455 |
| existential_quantifiers | 11 | 0.545 | 0.545 | 0.091 |
| future | 11 | 0.273 | 0.273 | 0.636 |
| nominal_adverbials | 11 | 0.091 | 0.091 | 0.000 |
| noun_plural | 11 | 0.727 | 0.727 | 0.000 |
| personification | 11 | 0.000 | 0.000 | 0.000 |
| present_progressive | 11 | 0.364 | 0.273 | 0.636 |
| quantifier | 11 | 0.182 | 0.182 | 0.182 |
| static_dynamic | 11 | 0.273 | 0.182 | 0.364 |
| comparative | 10 | 0.400 | 0.400 | 0.000 |
| declaration | 10 | 0.300 | 0.300 | 0.100 |
| expletive | 10 | 0.200 | 0.200 | 0.500 |
| expressive | 10 | 0.500 | 0.300 | 0.300 |
| first_conditional | 10 | 0.000 | 0.000 | 0.000 |
| interrogative | 10 | 0.500 | 0.000 | 0.700 |
| metaphor | 10 | 0.000 | 0.000 | 0.000 |
| middle_verb | 10 | 0.100 | 0.000 | 0.300 |
| object_expletives | 10 | 0.000 | 0.000 | 0.100 |
| optative | 10 | 0.400 | 0.100 | 0.500 |
| past | 10 | 0.600 | 0.600 | 0.500 |
| past_tense | 10 | 0.700 | 0.700 | 0.000 |
| politeness | 10 | 0.300 | 0.100 | 0.500 |
| present_participle | 10 | 0.200 | 0.200 | 0.500 |
| resultative | 10 | 0.000 | 0.000 | 0.000 |
| subject_verb_inversion | 10 | 0.200 | 0.000 | 0.200 |
| agentive_suffix | 9 | 0.111 | 0.111 | 0.111 |
| existential | 9 | 0.222 | 0.000 | 0.333 |
| hyperbole | 9 | 0.000 | 0.000 | 0.111 |
| indirect_speech | 9 | 0.000 | 0.000 | 0.000 |
| intransitive_verb | 9 | 0.000 | 0.000 | 0.000 |
| noun_clauses | 9 | 0.111 | 0.111 | 0.111 |
| past_progressive | 9 | 0.222 | 0.222 | 0.556 |
| present_perfect | 9 | 0.444 | 0.444 | 0.556 |
| quantitative_prefix | 9 | 0.000 | 0.000 | 0.000 |
| superlative | 9 | 0.667 | 0.667 | 0.111 |
| anaphor | 8 | 0.875 | 0.875 | 0.125 |
| clausal_subjects | 8 | 0.375 | 0.000 | 0.625 |
| copular_be | 8 | 0.250 | 0.250 | 0.125 |
| count_nouns | 8 | 0.000 | 0.000 | 0.000 |
| direct_object | 8 | 0.000 | 0.000 | 0.000 |
| discourse_markers | 8 | 0.125 | 0.125 | 0.125 |
| euphemism | 8 | 0.000 | 0.000 | 0.375 |
| factives | 8 | 0.000 | 0.000 | 0.000 |
| futurates | 8 | 0.875 | 0.875 | 1.000 |
| negation_prefix | 8 | 0.500 | 0.500 | 0.125 |
| subject_auxiliary_inversion | 8 | 0.250 | 0.000 | 0.500 |
| subjunctive_mood | 8 | 0.375 | 0.250 | 0.375 |
| verbal_suffix | 8 | 0.125 | 0.125 | 0.000 |
