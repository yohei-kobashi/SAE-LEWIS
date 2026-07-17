# S_min 安定核 × FRC top-3(core-frac 0.5, min-pairs 3)

安定核 = 因果検証済みの feature対応集合の候補(事例間で共通に生き残る介入ハンドル)。FRC3列 = LinguaLensの現象同定 top-3。

| phenomenon | pairs | mean\|S_min\| | union | Jaccard | \|core\| | core∩FRC3 | FRC3出現率 | AUROC1出現率 | core特徴 |
|---|---|---|---|---|---|---|---|---|---|
| futurates | 6 | 1.3 | 5 | 0.40 | 1 | 0 | 0.11 | 0.17 | 15596(4/6) forms of the verb "to be" in various tenses |
| cleft_sentences | 5 | 5.4 | 21 | 0.11 | 1 | 0 | 0.07 | 0.00 | 10721(3/5) phrases questioning desires, intentions, or m |
| emphatic_structure | 5 | 1.2 | 4 | 0.30 | 1 | 0 | 0.07 | 0.00 | 15191(3/5) phrases indicating possession or existence of |
| of_genitive | 5 | 4.0 | 16 | 0.33 | 1 | 1 | 0.47 | 0.20 | 5511(4/5) frequently occurring articles in a text |
| passive_voice | 5 | 1.2 | 4 | 0.30 | 1 | 0 | 0.00 | 0.00 | 11438(3/5) phrases that indicate authorship or attributi |
| clausal_subjects | 5 | 8.6 | 40 | 0.04 | 1 | 0 | 0.00 | 0.00 | 384(3/5) instances of the word "what," indicating a fo |
| future_perfect | 5 | 10.4 | 32 | 0.27 | 5 | 2 | 0.60 | 0.20 | 11781(5/5) sentences with future tense and conditional p, 11118(5/5) references to actions related to work and con, 2041(4/5) instances of the word "will" indicating futur, 12220(4/5) words indicating possession or existence |
| interrogative | 5 | 6.8 | 14 | 0.41 | 6 | 0 | 0.00 | 0.80 | 3797(5/5) questions and queries within the text, 11160(5/5) questions related to factors and conditions a, 4744(4/5) affirmative and interrogative modal verbs ind, 7283(4/5) interrogative punctuation marks or questions |
| past | 4 | 4.0 | 8 | 0.30 | 5 | 2 | 0.42 | 0.50 | 11118(3/4) references to actions related to work and con, 607(3/4) actions and operations related to handling re, 3967(3/4) actions related to reporting or presenting in, 13390(2/4) recurring actions or events that signify esse |
| synecdoche | 4 | 5.8 | 23 | 0.00 | 0 | 0 | 0.00 | 0.00 | — |
| past_perfect | 4 | 19.5 | 56 | 0.14 | 15 | 2 | 0.58 | 0.75 | 7876(4/4) instances of the verb "had" in various gramma, 681(3/4) phrases related to significant events or impa, 13390(3/4) recurring actions or events that signify esse, 785(3/4) forms of the verb "to be" |
| subject_auxiliary_inversion | 4 | 8.2 | 32 | 0.01 | 1 | 0 | 0.17 | 0.00 | 8800(2/4) terms associated with economic conditions, pa |
| future_progressive | 3 | 5.0 | 9 | 0.39 | 4 | 1 | 0.44 | 0.33 | 2041(3/3) instances of the word "will" indicating futur, 11781(3/3) sentences with future tense and conditional p, 6810(2/3) terms related to software licensing and legal, 14014(2/3) the verb "do" in various contexts |
| middle_verb | 3 | 10.7 | 27 | 0.11 | 4 | 1 | 0.33 | 0.33 | 3391(3/3) terms related to scientific research and meth, 112(2/3) verbs or phrases that indicate changes or tra, 15596(2/3) forms of the verb "to be" in various tenses, 14454(2/3) instances of the verb "are" and its variation |
| present_progressive | 3 | 2.3 | 5 | 0.32 | 1 | 1 | 0.33 | 1.00 | 14768(3/3) action verbs related to ongoing processes or  |
| transitional | 3 | 16.0 | 42 | 0.05 | 6 | 0 | 0.11 | 0.33 | 6725(2/3) words indicating contrast or opposition in co, 12358(2/3) elements related to numerical data or statist, 1673(2/3) instances of "however" and its variations, in, 6314(2/3) scientific evidence related to health and tre |
| commisive | 3 | 18.0 | 47 | 0.09 | 6 | 0 | 0.00 | 0.00 | 11781(3/3) sentences with future tense and conditional p, 1386(2/3) references to personal possessions and famili, 15888(2/3) action verbs related to processes and events, 2041(2/3) instances of the word "will" indicating futur |
| deontic | 3 | 8.3 | 23 | 0.04 | 2 | 0 | 0.11 | 0.00 | 11118(2/3) references to actions related to work and con, 607(2/3) actions and operations related to handling re |
| epistemic | 3 | 7.3 | 21 | 0.05 | 1 | 0 | 0.00 | 0.00 | 3343(2/3) conditional language indicating possibility o |
| future | 3 | 7.3 | 18 | 0.17 | 2 | 1 | 0.33 | 1.00 | 11781(3/3) sentences with future tense and conditional p, 607(3/3) actions and operations related to handling re |
| non_synecdoche_metonymy | 3 | 3.3 | 10 | 0.00 | 0 | 0 | 0.00 | 0.00 | — |
| split_infinitives | 3 | 9.3 | 26 | 0.07 | 1 | 0 | 0.00 | 0.33 | 8073(3/3) adverbs indicating degree or extent |

## 集計

- 対象現象: 22(pairs ≥ 3)
- 安定核が非空: 20/22
- 安定核がFRC top-3と交わる: 8/22
- 安定核サイズ: median 1.5 / mean 3.2(vs FRC の 3)

読み(免許規則・形(ii)): 安定核が非空でFRC3と交わらない → 因果的に有効な対応集合はFRCの同定と別物。核がFRC3を含みつつ大きい → 対応集合はtop-3より広い。核が空(低Jaccard) → 事例あたり少数だが固定の対応集合は存在しない(現象レベル選択の失敗の機構的説明)。いずれの場合もFRC3出現率の列が「FRCの3本は因果ハンドルとしてどれだけ再利用されているか」を直接測る。
