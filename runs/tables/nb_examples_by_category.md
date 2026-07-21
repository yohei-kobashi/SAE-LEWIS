# 提案手法(nobudgetチャンピオン、L12)の成功・失敗事例 — 4大分類×各10(無作為抽出)

probe500(true条件、復唱枠greedy)より、各分類の成功/失敗プールから無作為抽出(seed=42)。
成功=exact一致(出力=tgt)、失敗=不一致(出力併記)。プール規模: 
morphology 成17/失87、syntax 成67/失162、semantics 成76/失136、pragmatics 成59/失99。

## morphology

### 成功(10/17)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| negation_prefix | 1 | The team is disorganized. | The team is organized. | (=tgt) |
| present_participle | 2 | He is shaving his beard. | He shaves his beard. | (=tgt) |
| comparative | 2 | This hill is rockier than it looks. | This hill is rocky than it looks. | (=tgt) |
| superlative | 1 | My brother is the tallest in his class. | My brother is the tall in his class. | (=tgt) |
| noun_plural | 1 | The houses in this neighborhood are large. | The house in this neighborhood are large. | (=tgt) |
| past_tense_irregular | 1 | They found the keys on the kitchen counter. | They find the keys on the kitchen counter. | (=tgt) |
| past_tense_irregular | 1 | The children built a sandcastle at the beach. | The children build a sandcastle at the beach. | (=tgt) |
| degree_prefix | 2 | The response was supercritical. | The response was critical. | (=tgt) |
| degree_prefix | 3 | The plan was overambitious. | The plan was ambitious. | (=tgt) |
| present_participle | 2 | He is practicing the guitar. | He practices the guitar. | (=tgt) |

### 失敗(10/87)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| adjectival_suffix | 1 | He felt fearless in battle. | He felt fear in battle. | He felt fearless in battle. |
| adjectival_suffix | 1 | She gave a lovable performance. | She gave a love performance. | She gave a loving performance. |
| agentive_suffix | 1 | The performer received applause after the show. | The perform received applause after the show. | The performers received applause after the show. |
| negation_prefix | 1 | The car is uninsured. | The car is insured. | The car is uninsured. |
| past_tense_irregular | 1 | She swam in the pool all afternoon. | She swim in the pool all afternoon. | She swims in the pool all afternoon. |
| past_tense | 1 | We agreed with the decision. | We agree with the decision. | We agreed with the decision. |
| noun_plural | 1 | The birds are flying high. | The bird are flying high. | The bird is flying high. |
| past_participle_irregular | 2 | The house has been built. | The house has be build. | The house has been built. |
| superlative | 2 | It was the roughest road. | It was the rough road. | It was the roughest road. |
| degree_prefix | 3 | He gave an ultrashort reply. | He gave a short reply. | He gave a very short reply. |

## syntax

### 成功(10/67)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| factives | 5 | The fact that we missed the train annoyed her. | Missing the train annoyed her. | (=tgt) |
| active_verbs | 9 | He has cleaned the windows of the office building. | The windows of the office building have been cleaned by him. | (=tgt) |
| subject_auxiliary_inversion | 7 | So clear was her voice that we all heard. | Her voice was so clear that we all heard. | (=tgt) |
| of_genitive | 3 | We defended the thesis of ours. | We defended our thesis. | (=tgt) |
| subject_verb_inversion | 5 | Here comes the mailman with today’s letters. | The mailman comes here with today’s letters. | (=tgt) |
| interrogative | 4 | Did ancient civilizations build irrigation systems? | Ancient civilizations built irrigation systems. | (=tgt) |
| cleft_sentences | 5 | It is thanks to modern medicine that she recovered. | She recovered thanks to modern medicine. | (=tgt) |
| elliptical_sentences | 1 | She plays more than I do. | She plays more than I play. | (=tgt) |
| present_participle | 2 | He is shaving his beard. | He shaves his beard. | (=tgt) |
| futurates | 2 | He is returning the book to the library tomorrow. | He will return the book to the library tomorrow. | (=tgt) |

### 失敗(10/162)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| appositives | 2 | His challenge, climbing Mount Everest, tested him to the limit. | His challenge, which is climbing Mount Everest, tested him to the limit. | His challenge, climbing Mount Everest, tested him to the limit. |
| relative_clauses | 5 | The teacher who inspired me retired last year. | My inspiring teacher retired last year. | My teacher who inspired me retired last year. |
| past_participle | 1 | He had brushed his teeth. | He had brush his teeth. | He brushed his teeth. |
| cleft_sentences | 5 | What I like most is chocolate ice cream. | I like chocolate ice cream most. | I like I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I |
| linking_verb | 7 | The coffee tastes bitter. | Excessive brewing darkens the coffee. | Coffee tastes bitter. |
| appositives | 2 | The novelist, Jane Austen, wrote *Pride and Prejudice*. | The novelist, who is Jane Austen, wrote *Pride and Prejudice*. | The novelist, Jane Austen, wrote *Pride and Prejudice*. |
| third_person_singular | 1 | The author publishes a new book every year. | The author publish a new book every year. | The author publishes a new book every year. |
| optative | 5 | Would that the sun rose early! | The sun rises early. | The sun would rise early! |
| present_participle | 2 | We are traveling to Kyoto this week. | We travel to Kyoto this week. | We traveled to Kyoto this week. |
| universal_quantifiers | 3 | All the files were backed up. | The files were entirely backed up. | The the files were backed up. |

## semantics

### 成功(10/76)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| transitional | 3 | AI diagnoses diseases accurately, yet doctors distrust its recommendations. | AI diagnoses diseases accurately. Doctors distrust its recommendations. | (=tgt) |
| referring | 2 | Our athletes train daily. | Athletes train daily. | (=tgt) |
| futurates | 2 | I am visiting my grandparents during the holidays. | I will visit my grandparents during the holidays. | (=tgt) |
| past | 2 | They rehearsed the play daily. | They rehearse the play daily. | (=tgt) |
| deontic | 3 | He ought to speak quietly. | He speaks quietly. | (=tgt) |
| future_progressive | 3 | You will not be watching those movies. | You do not watch those movies. | (=tgt) |
| future_progressive | 5 | I am going to be working here. | I work here. | (=tgt) |
| futurates | 2 | She is finishing her report by Friday. | She will finish her report by Friday. | (=tgt) |
| middle_verb | 2 | This wood carves smoothly with sharp tools. | This wood is carved smoothly with sharp tools. | (=tgt) |
| given_known | 1 | That car alarm keeps going off. | A car alarm keeps going off. | (=tgt) |

### 失敗(10/136)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| synecdoche | 2 | The blacksmith forged fire in the furnace. | The blacksmith forged horseshoes in the furnace. | The blacksmith forged fire in the furnace. |
| count_nouns | 5 | I bought a box and a bag. | I bought container items. | I bought containers and a container. |
| subjunctive_mood | 2 | I suggest that he be on time. | I suggest he should be on time. | I suggested he be on time. |
| factives | 5 | The knowledge that she speaks five languages impressed us. | Her speaking five languages impressed us. | Her knowledge of five languages impressed us. |
| hyperbole | 10 | He’s so busy he doesn’t have a second to breathe. | He’s very busy. | He is very busy and does not have very much time to breathe. |
| quantifier | 2 | She bought all the ingredients for the recipe. | She bought ingredients for the recipe. | She bought all the ingredients for the recipe. |
| past_perfect | 2 | It had not snowed here often. | It does not snow here often. | It did not snow here often. |
| existential_quantifiers | 1 | I felt some relief. | I felt relief. | I felt some relief. |
| future_perfect | 3 | He will have watched TV in the evenings. | He watches TV in the evenings. | He watched TV in the evenings. |
| future_progressive | 3 | Will you be watching movies? | Do you watch movies? | Do you do watch movies? |

## pragmatics

### 成功(10/59)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| interrogative | 4 | Did ancient civilizations build irrigation systems? | Ancient civilizations built irrigation systems. | (=tgt) |
| anaphor | 1 | The boys played their game happily. | The boys played a game happily. | (=tgt) |
| tag_questions | 7 | You were sleeping, weren’t you? | Were you sleeping? | (=tgt) |
| interrogative | 3 | Were they celebrating their anniversary? | They were celebrating their anniversary. | (=tgt) |
| emphatic_structure | 2 | She did arrive early. | She arrived early. | (=tgt) |
| emphatic_structure | 2 | We did fix the problem. | We fixed the problem. | (=tgt) |
| transitional | 3 | AI diagnoses diseases accurately, yet doctors distrust its recommendations. | AI diagnoses diseases accurately. Doctors distrust its recommendations. | (=tgt) |
| imperative_sentence | 3 | Let's finish this by tomorrow. | We should finish this by tomorrow. | (=tgt) |
| expressive | 4 | We apologize that the train departs late. | The train departs late. | (=tgt) |
| expressive | 4 | We apologize that the alarm rings accidentally. | The alarm rings accidentally. | (=tgt) |

### 失敗(10/99)

| feature | ops | src | tgt | 出力 |
|---|---|---|---|---|
| referring | 3 | The dust everywhere covers furniture. | Dust covers furniture. | Dust everywhere covers furniture. |
| echo_questions | 3 | "Why didn’t you call?" Why didn’t I call? I forgot. | "Why didn’t you call?" I didn’t call because I forgot. | Why didn’t you call?" because I forgot. |
| politeness | 4 | Could you help me carry this upstairs? | Help me carry this upstairs. | Could you help me carry this upstairs? |
| tag_questions | 7 | You have been here before, haven’t you? | Have you been here before? | You've been here before, haven't you? |
| euphemism | 3 | She’s advancing her family. | She’s pregnant. | She’s advancing her career. |
| hyperbole | 10 | He’s so busy he doesn’t have a second to breathe. | He’s very busy. | He is very busy and does not have very much time to breathe. |
| personification | 5 | The curtains tangoed in the wind. | The curtains swayed. | The curtains swayed in the breeze. |
| intensifiers | 1 | They were absolutely thrilled to meet their idol. | They were thrilled to meet their idol. | They were absolutely thrilled to meet their idol. |
| declaration | 3 | A director overrides a policy conflicts with bylaws. | A policy conflicts with bylaws. | A director overrides a policy conflicts with bylaws. |
| deontic | 1 | They should watch the movie. | They watch the movie. | They should watch the movie. |

