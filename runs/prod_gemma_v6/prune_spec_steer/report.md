# P-O: per-instance minimal conditioning sets

pairs pruned: 134 (of 135 attempted); effector = steer (steer alpha 0.5, greedy rewrite); total decodes 3685

**|S_min|: median 5, mean 7.9, min 1, max 43 — vs the k=32 default (mean n_full 98.7).**

| phenomenon | pairs | mean \|S_min\| | union | mean pairwise Jaccard | top recurring feature |
|---|---|---|---|---|---|
| futurates | 6 | 1.3 | 5 | 0.40 | 15596 (4/6) forms of the verb "to be" in various tenses |
| cleft_sentences | 5 | 5.4 | 21 | 0.11 | 10721 (3/5) phrases questioning desires, intentions, or meanin |
| emphatic_structure | 5 | 1.2 | 4 | 0.30 | 15191 (3/5) phrases indicating possession or existence of cert |
| of_genitive | 5 | 4.0 | 16 | 0.33 | 5511 (4/5) frequently occurring articles in a text |
| passive_voice | 5 | 1.2 | 4 | 0.30 | 11438 (3/5) phrases that indicate authorship or attribution re |
| clausal_subjects | 5 | 8.6 | 40 | 0.04 | 384 (3/5) instances of the word "what," indicating a focus o |
| future_perfect | 5 | 10.4 | 32 | 0.27 | 11781 (5/5) sentences with future tense and conditional phrase |
| interrogative | 5 | 6.8 | 14 | 0.41 | 3797 (5/5) questions and queries within the text |
| past | 4 | 4.0 | 8 | 0.30 | 11118 (3/4) references to actions related to work and contribu |
| synecdoche | 4 | 5.8 | 23 | 0.00 | 3039 (1/4) references to variable names and types in programm |
| past_perfect | 4 | 19.5 | 56 | 0.14 | 7876 (4/4) instances of the verb "had" in various grammatical |
| subject_auxiliary_inversion | 4 | 8.2 | 32 | 0.01 | 8800 (2/4) terms associated with economic conditions, particu |
| future_progressive | 3 | 5.0 | 9 | 0.39 | 11781 (3/3) sentences with future tense and conditional phrase |
| middle_verb | 3 | 10.7 | 27 | 0.11 | 3391 (3/3) terms related to scientific research and methodolo |
| present_progressive | 3 | 2.3 | 5 | 0.32 | 14768 (3/3) action verbs related to ongoing processes or activ |
| transitional | 3 | 16.0 | 42 | 0.05 | 7961 (2/3) phrases and terms related to problem-solving and m |
| commisive | 3 | 18.0 | 47 | 0.09 | 11781 (3/3) sentences with future tense and conditional phrase |
| deontic | 3 | 8.3 | 23 | 0.04 | 11118 (2/3) references to actions related to work and contribu |
| epistemic | 3 | 7.3 | 21 | 0.05 | 3343 (2/3) conditional language indicating possibility or pot |
| future | 3 | 7.3 | 18 | 0.17 | 11781 (3/3) sentences with future tense and conditional phrase |
| non_synecdoche_metonymy | 3 | 3.3 | 10 | 0.00 | 8965 (1/3) details about furniture characteristics and arrang |
| split_infinitives | 3 | 9.3 | 26 | 0.07 | 8073 (3/3) adverbs indicating degree or extent |
| euphemism | 2 | 16.0 | 31 | 0.03 | 2616 (2/2) words and phrases that consistently appear multipl |
| expletive | 2 | 3.5 | 7 | 0.00 | 13704 (1/2) the word "there" |
| subjunctive_mood | 2 | 6.0 | 12 | 0.00 | 8385 (1/2) expressions of hope or well-wishing |
| expressive | 2 | 14.0 | 27 | 0.04 | 1708 (2/2) the pronoun "I" and its variations, indicating per |
| past_progressive | 2 | 6.0 | 9 | 0.33 | 11118 (2/2) references to actions related to work and contribu |
| politeness | 2 | 10.5 | 15 | 0.40 | 13828 (2/2) instances of the pronoun 'you' |
| present_participle | 2 | 7.5 | 15 | 0.00 | 14560 (1/2) greetings and expressions of friendliness |
| present_perfect | 2 | 5.0 | 9 | 0.11 | 12220 (2/2) words indicating possession or existence |
| referring | 2 | 19.0 | 31 | 0.23 | 4608 (2/2) the word "that" and its variants used in various c |
| spatial_or_directional_prefix | 2 | 13.5 | 27 | 0.00 | 5511 (1/2) frequently occurring articles in a text |
| subject_verb_inversion | 2 | 5.0 | 10 | 0.00 | 15205 (1/2) articles and references to "the" or "a" |
| tag_questions | 2 | 8.5 | 16 | 0.06 | 2151 (2/2) dialogue and conversational phrases |
| telic_atelic | 2 | 23.0 | 45 | 0.02 | 6810 (2/2) terms related to software licensing and legal disc |
| temporal | 2 | 7.5 | 15 | 0.00 | 2520 (1/2) references to scheduled events or invitations |
| transitive_verb | 2 | 4.5 | 9 | 0.00 | 7616 (1/2) references to ranges or measurements in various co |
| turn_taking | 2 | 9.5 | 18 | 0.06 | 6810 (2/2) terms related to software licensing and legal disc |

Reading: small |S_min| with union >> |S_min| and low Jaccard = the command is small PER INSTANCE but instance-specific — the mechanistic explanation for why phenomenon-level selections (FRC r=3, AUROC r=1) fail: no fixed small set serves a phenomenon. High Jaccard phenomena are the opposite: a stable causal core exists and could seed a causally-validated dictionary (better than FRC/AUROC).
