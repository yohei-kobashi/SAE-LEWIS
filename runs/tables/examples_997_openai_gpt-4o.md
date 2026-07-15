# LinguaLens transformation examples — focus: ours


**Coverage**: success 136, near 182, fail 99 over 98 features.

Systems: `ours` (routed), `ef32` (k32), `steer` (steer0.5). Pairs where every system has an output: 997.

Categories (= the residual-frontier decomposition):

- **success** — focus system exact-matched the target
- **near** — exact miss, but the judge saw the feature realized in the commanded direction — directionally realizable, not exactly editable
- **fail** — exact miss and not realized — the unreachable end

Outputs are word-diffed against the SOURCE: **added/substituted**, ~~removed~~. `exact` is against the target; `FRR` is the judge's realized verdict (— = not judged for that system).

## active_verbs

### near — idx 0

- **source**: She eats an apple every morning.
- **target**: ~~She eats an~~ **An** apple **is eaten by her** every morning.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~She eats an apple every morning.~~ **is a by a farmer in by the province of of in by the the the of of the of of of of fruit by the the province the province.**
- `ef32` [exact ✗ · FRR ✓]: ~~She eats an apple every morning.~~ **is a by a farmer in by the province of of in by the the the of of the of of of of fruit by the the province the province.**
- `steer` [exact ✗ · FRR ✗]: She eats an apple ~~every~~ **each** morning.


### near — idx 15

- **source**: We fixed the leaky faucet yesterday.
- **target**: ~~We fixed the~~ **The** leaky faucet **was fixed by us** yesterday.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~We fixed the leaky faucet~~ **It was repaired** yesterday.
- `ef32` [exact ✗ · FRR ✓]: ~~We~~ **were not by by by a by us a by by by us by by, by by a by by by by us, by by** fixed ~~the leaky~~ **by us** faucet **by** yesterday.
- `steer` [exact ✗ · FRR ✓]: ~~We fixed the leaky faucet~~ **It was repaired** yesterday.


### fail — idx 20

- **source**: She found a wallet on the sidewalk.
- **target**: ~~She~~ **A wallet was** found ~~a wallet~~ on the ~~sidewalk.~~ **sidewalk by her.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: She ~~found~~ **was** a ~~wallet on the~~ **was a 11 1 a 1 a 1 was a a 1 1 111 111 1** sidewalk.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 22

- **source**: He will read the report carefully tonight.
- **target**: ~~He~~ **The report** will **be** read ~~the report~~ carefully **by him** tonight.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~He will read~~ **The is is a is a is a is a is a is a is a report by** the ~~report~~ **the the by the** carefully **by by by by the by the the** tonight.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## adjectival_suffix

### success — idx 88

- **source**: The student was obedient.
- **target**: The student was ~~obedient.~~ **obey.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The student was ~~obedient.~~ **obey.**
- `ef32` [exact ✓ · FRR ✓]: The student was ~~obedient.~~ **obey.**
- `steer` [exact ✗ · FRR ✓]: The student was ~~obedient.~~ **obediently.**


### success — idx 52

- **source**: She gave a thoughtful reply.
- **target**: She gave a ~~thoughtful~~ **thought** reply.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She gave a ~~thoughtful~~ **thought** reply.
- `ef32` [exact ✓ · FRR ✓]: She gave a ~~thoughtful~~ **thought** reply.
- `steer` [exact ✗ · FRR ✗]: She gave a ~~thoughtful~~ **considered** reply.


### near — idx 71

- **source**: The golden crown sparkled.
- **target**: The ~~golden~~ **gold** crown sparkled.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~golden crown~~ **gold crowns** sparkled.
- `ef32` [exact ✗ · FRR ✓]: The ~~golden crown~~ **gold crowns** sparkled.
- `steer` [exact ✗ · FRR ✓]: The ~~golden~~ **gleaming** crown sparkled.


### near — idx 86

- **source**: Her voice sounded voiceless.
- **target**: Her voice sounded ~~voiceless.~~ **voice.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Her voice ~~sounded voiceless.~~ **sounded.**
- `ef32` [exact ✗ · FRR ✓]: Her voice ~~sounded voiceless.~~ **sounded.**
- `steer` [exact ✗ · FRR ✓]: Her voice sounded ~~voiceless.~~ **voic-less.**


### fail — idx 61

- **source**: They explored cultural differences.
- **target**: They explored ~~cultural~~ **culture** differences.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: They ~~explored~~ **examined** cultural differences.


### fail — idx 80

- **source**: The agile fox escaped.
- **target**: The ~~agile~~ **ag** fox escaped.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: The ~~agile fox~~ **agagagagagagagagagagagagagagagagagagag foxag'ss'sssagag** escaped.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## adverbial_suffix

### success — idx 110

- **source**: She looked sadly at the broken toy.
- **target**: She looked ~~sadly~~ **sad** at the broken toy.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She looked ~~sadly~~ **sad** at the broken toy.
- `ef32` [exact ✓ · FRR ✓]: She looked ~~sadly~~ **sad** at the broken toy.
- `steer` [exact ✗ · FRR ✓]: She looked ~~sadly~~ **forlorn** at the broken toy.


### near — idx 133

- **source**: They responded aggressively.
- **target**: They responded ~~aggressively.~~ **aggressive.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They ~~responded~~ **very** aggressively.
- `ef32` [exact ✗ · FRR ✓]: They ~~responded~~ **very** aggressively.
- `steer` [exact ✗ · FRR ✓]: They responded ~~aggressively.~~ **with aggression.**


### near — idx 146

- **source**: She explained thoughtfully.
- **target**: She explained ~~thoughtfully.~~ **thoughtful.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~explained thoughtfully.~~ **thoughtful.**
- `ef32` [exact ✗ · FRR ✓]: She ~~explained thoughtfully.~~ **thoughtful.**
- `steer` [exact ✗ · FRR ✗]: She ~~explained thoughtfully.~~ **thoughtfully explained.**


### fail — idx 114

- **source**: She sings beautifully.
- **target**: She sings ~~beautifully.~~ **beautiful.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: She sings ~~beautifully.~~ **beautifully songs.**


### fail — idx 101

- **source**: She answered the question confidently.
- **target**: She answered the question ~~confidently.~~ **confident.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: She answered the ~~question~~ **questions** confidently.
- `ef32` [exact ✗ · FRR ✗]: She answered the ~~question~~ **questions** confidently.
- `steer` [exact ✗ · FRR ✓]: She answered the question ~~confidently.~~ **with confidence.**

## agentive_suffix

### success — idx 167

- **source**: The participant answered all the questions.
- **target**: The ~~participant~~ **participate** answered all the questions.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The ~~participant~~ **participate** answered all the questions.
- `ef32` [exact ✓ · FRR ✓]: The ~~participant~~ **participate** answered all the questions.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 170

- **source**: The assistant organized the event.
- **target**: The ~~assistant~~ **assist** organized the event.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~assistant organized~~ **assist scored** the event.
- `ef32` [exact ✗ · FRR ✓]: The ~~assistant organized~~ **assist scored** the event.
- `steer` [exact ✗ · FRR ✗]: The ~~assistant~~ **event was** organized **by** the ~~event.~~ **assistant.**


### near — idx 184

- **source**: The operator fixed the machine.
- **target**: The ~~operator~~ **operate** fixed the machine.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~operator~~ **function** fixed the machine.
- `ef32` [exact ✗ · FRR ✓]: The ~~operator~~ **function** fixed the machine.
- `steer` [exact ✗ · FRR ✗]: The operator ~~fixed~~ **repaired** the machine.


### fail — idx 162

- **source**: They found a good negotiator.
- **target**: They found a good ~~negotiator.~~ **negotiate.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 189

- **source**: The counselor provided helpful advice.
- **target**: The ~~counselor~~ **counsel** provided helpful advice.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✓ · FRR ✓]: The ~~counselor~~ **counsel** provided helpful advice.

## anaphor

### success — idx 202

- **source**: The boys played their game happily.
- **target**: The boys played ~~their~~ **a** game happily.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The boys played ~~their~~ **a** game happily.
- `ef32` [exact ✓ · FRR ✓]: The boys played ~~their~~ **a** game happily.
- `steer` [exact ✓ · FRR ✓]: The boys played ~~their~~ **a** game happily.


### success — idx 206

- **source**: The team celebrated their victory loudly.
- **target**: The team celebrated ~~their~~ **a** victory loudly.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The team celebrated ~~their~~ **a** victory loudly.
- `ef32` [exact ✓ · FRR ✓]: The team celebrated ~~their~~ **a** victory loudly.
- `steer` [exact ✗ · FRR ✗]: The team celebrated their victory ~~loudly.~~ **jubilantly.**


### fail — idx 228

- **source**: The teacher graded their papers carefully.
- **target**: The teacher graded ~~their~~ papers carefully.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~The teacher graded their papers carefully.~~ **rowspan rowspan rowspan**

## appositives

### near — idx 264

- **source**: His hobby, woodworking, relaxes him after work.
- **target**: His hobby, **which is** woodworking, relaxes him after work.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: His ~~hobby,~~ **hobby,,, is is is,, is,,, is,, is a a, is,,, and a,, and very,** woodworking, relaxes him after work.
- `ef32` [exact ✗ · FRR ✓]: His ~~hobby,~~ **hobby,,, is is is,, is,,, is,, is a a, is,,, and a,, and very,** woodworking, relaxes him after work.
- `steer` [exact ✗ · FRR ✓]: ~~His hobby, woodworking, relaxes him after work.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### near — idx 268

- **source**: Their house, a historic landmark, attracts visitors.
- **target**: Their house, **which is** a historic landmark, attracts visitors.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Their house, a historic landmark, attracts visitors.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✓]: Their ~~house,~~ **house,, a, is, is,, is** a **is a,, and is, a, is,, is, very** historic landmark, attracts visitors.
- `steer` [exact ✗ · FRR ✓]: ~~Their house, a historic landmark, attracts visitors.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 256

- **source**: The puppy, a golden retriever, chewed my slippers.
- **target**: The puppy, **which is** a golden retriever, chewed my slippers.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The puppy, a **very** golden retriever, chewed my slippers.
- `ef32` [exact ✗ · FRR ✗]: The puppy, a **very** golden retriever, chewed my slippers.
- `steer` [exact ✗ · FRR ✓]: ~~The puppy, a golden retriever, chewed my slippers.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 294

- **source**: Their decision, a bold strategic move, paid off.
- **target**: Their decision, **which is** a bold strategic move, paid off.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~Their decision, a bold strategic move, paid off.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## clausal_subjects

### success — idx 346

- **source**: What he enjoyed most was the concert.
- **target**: ~~What he~~ **He** enjoyed ~~most was~~ the ~~concert.~~ **concert most.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~What he~~ **He** enjoyed ~~most was~~ the ~~concert.~~ **concert most.**
- `ef32` [exact ✗ · FRR ✓]: ~~What he enjoyed most was the concert.~~ **He enjoyed.**
- `steer` [exact ✓ · FRR ✓]: ~~What he~~ **He** enjoyed ~~most was~~ the ~~concert.~~ **concert most.**


### success — idx 348

- **source**: What stops me is fear of failure.
- **target**: ~~What~~ **Fear of failure** stops ~~me is fear of failure.~~ **me.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~What~~ **Fear of failure** stops ~~me is fear of failure.~~ **me.**
- `ef32` [exact ✗ · FRR ✓]: ~~What stops~~ **and love of,,,,,, and fortune of and love,,, and and- andlife,, and and and** me ~~is fear of failure.~~ **and, and,.**
- `steer` [exact ✓ · FRR ✓]: ~~What~~ **Fear of failure** stops ~~me is fear of failure.~~ **me.**


### near — idx 311

- **source**: What matters is honesty.
- **target**: ~~What matters is honesty.~~ **Honesty matters.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~What matters is~~ **Honionionionededed by him his voice voice own voice own him-ion-rere-re him, his own voice** honesty.
- `ef32` [exact ✗ · FRR ✓]: ~~What matters is~~ **Honionionionededed by him his voice voice own voice own him-ion-rere-re him, his own voice** honesty.
- `steer` [exact ✓ · FRR ✓]: ~~What matters is honesty.~~ **Honesty matters.**


### near — idx 331

- **source**: What she dreams of is success.
- **target**: ~~What she~~ **She** dreams of ~~is~~ success.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~What she dreams of is success.~~ **who.**
- `ef32` [exact ✗ · FRR ✓]: ~~What she dreams of is success.~~ **who.**
- `steer` [exact ✓ · FRR ✓]: ~~What she~~ **She** dreams of ~~is~~ success.


### fail — idx 305

- **source**: What annoys me is loud music.
- **target**: ~~What~~ **Loud music** annoys ~~me is loud music.~~ **me.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~What annoys~~ **air and** me ~~is~~ loud music.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 309

- **source**: What worries me most is climate change.
- **target**: ~~What~~ **Climate change** worries me ~~most is climate change.~~ **most.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: What worries me most is **the** climate change.
- `ef32` [exact ✗ · FRR ✓]: ~~What worries~~ **climate and** me most ~~is climate~~ change.
- `steer` [exact ✗ · FRR ✗]: What worries me most is **the** climate change.

## cleft_sentences

### success — idx 351

- **source**: It was in June that we married.
- **target**: ~~It was~~ **We married** in ~~June that we married.~~ **June.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~It was~~ **We married** in ~~June that we married.~~ **June.**
- `ef32` [exact ✗ · FRR ✓]: ~~It was~~ **Marriage** in June ~~that we married.~~
- `steer` [exact ✓ · FRR ✓]: ~~It was~~ **We married** in ~~June that we married.~~ **June.**


### success — idx 358

- **source**: What we saw was a shooting star.
- **target**: ~~What we~~ **We** saw ~~was~~ a shooting star.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~What we~~ **We** saw ~~was~~ a shooting star.
- `ef32` [exact ✓ · FRR ✓]: ~~What we~~ **We** saw ~~was~~ a shooting star.
- `steer` [exact ✓ · FRR ✓]: ~~What we~~ **We** saw ~~was~~ a shooting star.


### near — idx 392

- **source**: What I cannot accept is dishonesty.
- **target**: ~~What~~ I cannot accept ~~is~~ dishonesty.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~What~~ I ~~cannot accept is dishonesty.~~
- `ef32` [exact ✗ · FRR ✓]: ~~What~~ I ~~cannot accept is dishonesty.~~
- `steer` [exact ✓ · FRR ✓]: ~~What~~ I cannot accept ~~is~~ dishonesty.


### near — idx 352

- **source**: It is Mary who baked the cake.
- **target**: ~~It is~~ Mary ~~who~~ baked the cake.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: It is ~~Mary~~ **Mary,** who baked the cake.
- `ef32` [exact ✗ · FRR ✓]: ~~It is~~ Mary ~~who baked the cake.~~ **baked.**
- `steer` [exact ✗ · FRR ✓]: It is ~~Mary~~ **Mary,** who baked the cake.


### fail — idx 388

- **source**: What worries me is the deadline.
- **target**: ~~What~~ **The deadline** worries ~~me is the deadline.~~ **me.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~What worries me~~ **The National deadline** is ~~the deadline.~~ **for submission is for submission of for submission of of of for submission of of of of for of submission of of of for submission submission of March .**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 362

- **source**: What surprised them was the sudden thunder.
- **target**: ~~What~~ **The sudden thunder** surprised ~~them was the sudden thunder.~~ **them.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~What surprised them was the sudden~~ **The InternationalThe International National Police Police Police Police International Police Police International Police Police Police Police Police International Police Police Police Police Police Police Police International Police International Police Police International thunder** thunder. **The International International**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## commisive

### success — idx 428

- **source**: We ensure the flight will land safely.
- **target**: ~~We ensure the~~ **The** flight ~~will land~~ **lands** safely.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~We ensure the~~ **The** flight ~~will land~~ **lands** safely.
- `ef32` [exact ✗ · FRR ✓]: ~~We ensure~~ **The evening,,** the **when, when London London** flight ~~will land safely.~~ **arrives arrives to from London London London, London which was at London London Airport in London at London, London airport.**
- `steer` [exact ✓ · FRR ✓]: ~~We ensure the~~ **The** flight ~~will land~~ **lands** safely.


### success — idx 418

- **source**: I swear the alarm will sound at dawn.
- **target**: ~~I swear the~~ **The** alarm ~~will sound~~ **sounds** at dawn.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~I swear the~~ **The** alarm ~~will sound~~ **sounds** at dawn.
- `ef32` [exact ✗ · FRR ✓]: ~~I swear the alarm will~~ **The** sound ~~at dawn.~~
- `steer` [exact ✓ · FRR ✓]: ~~I swear the~~ **The** alarm ~~will sound~~ **sounds** at dawn.


### near — idx 401

- **source**: We will have dinner ready by 7.
- **target**: ~~We will have dinner~~ **Dinner is** ready ~~by~~ **at** 7.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~We will have dinner ready by 7.~~
- `ef32` [exact ✗ · FRR ✓]: ~~We will have dinner ready by 7.~~
- `steer` [exact ✗ · FRR ✓]: ~~We~~ **Dinner** will ~~have dinner~~ ready by 7.


### near — idx 412

- **source**: We pledge the concert will start punctually.
- **target**: ~~We pledge the~~ **The** concert ~~will start~~ **starts** punctually.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: We ~~pledge~~ **pledges** the concert will start punctually.
- `ef32` [exact ✗ · FRR ✓]: ~~We pledge the~~ **The** concert ~~will start punctually.~~ **begins at at 77.7...: The 733.3...7..:: at 37.**
- `steer` [exact ✗ · FRR ✓]: We ~~pledge~~ **pledges** the concert will start punctually.

## comparative

### success — idx 453

- **source**: He runs faster than I do.
- **target**: He runs ~~faster~~ **fast** than I do.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He runs ~~faster~~ **fast** than I do.
- `ef32` [exact ✓ · FRR ✓]: He runs ~~faster~~ **fast** than I do.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### success — idx 473

- **source**: That sofa is softer than mine.
- **target**: That sofa is ~~softer~~ **soft** than mine.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: That sofa is ~~softer~~ **soft** than mine.
- `ef32` [exact ✓ · FRR ✓]: That sofa is ~~softer~~ **soft** than mine.
- `steer` [exact ✗ · FRR ✗]: That sofa is softer than ~~mine.~~ **my sofa.**


### near — idx 483

- **source**: She is cleverer than people think.
- **target**: She is ~~cleverer~~ **clever** than people think.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She is ~~cleverer~~ than people think.
- `ef32` [exact ✗ · FRR ✓]: She is ~~cleverer~~ than people think.
- `steer` [exact ✗ · FRR ✓]: ~~She is cleverer than people think.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 472

- **source**: This hill is rockier than it looks.
- **target**: This hill is ~~rockier~~ **rocky** than it looks.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: This hill is ~~rockier~~ **rock** than it looks.
- `ef32` [exact ✗ · FRR ✓]: This hill is ~~rockier~~ **rock** than it looks.
- `steer` [exact ✗ · FRR ✓]: This hill is ~~rockier than it looks.~~ **surprisingly rocky.**


### fail — idx 498

- **source**: His tone was more respectful than usual.
- **target**: His tone was ~~more~~ respectful than usual.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~His tone was more respectful than usual.~~ **rowspan rowspan rowspan**


### fail — idx 489

- **source**: That method is more effective than this one.
- **target**: That method is ~~more~~ effective than this one.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~That method is more effective than this one.~~ **rowspan rowspan rowspan**

## coordination

### success — idx 518

- **source**: She dislikes reading emails let alone writing reports.
- **target**: She dislikes reading ~~emails let alone writing reports.~~ **emails.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She dislikes reading ~~emails let alone writing reports.~~ **emails.**
- `ef32` [exact ✓ · FRR ✓]: She dislikes reading ~~emails let alone writing reports.~~ **emails.**
- `steer` [exact ✗ · FRR ✓]: ~~She dislikes reading emails let alone writing reports.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### success — idx 545

- **source**: The microscope can't resolve cells let alone organelles.
- **target**: The microscope can't resolve ~~cells let alone organelles.~~ **cells.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The microscope can't resolve ~~cells let alone organelles.~~ **cells.**
- `ef32` [exact ✓ · FRR ✓]: The microscope can't resolve ~~cells let alone organelles.~~ **cells.**
- `steer` [exact ✗ · FRR ✓]: ~~The microscope can't resolve cells let alone organelles.~~ **rowspan rowspan rowspan**


### near — idx 514

- **source**: The seedlings barely survived the drought much less the frost.
- **target**: The seedlings barely survived the ~~drought much less the frost.~~ **drought.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~The seedlings barely survived the drought much less the frost.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**
- `ef32` [exact ✗ · FRR ✓]: The seedlings barely survived the drought ~~much~~ less ~~the~~ frost.
- `steer` [exact ✗ · FRR ✓]: ~~The seedlings barely survived the drought much less the frost.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## copular_be

### success — idx 558

- **source**: The exam is tomorrow.
- **target**: The exam ~~is~~ **takes place** tomorrow.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The exam ~~is~~ **takes place** tomorrow.
- `ef32` [exact ✓ · FRR ✓]: The exam ~~is~~ **takes place** tomorrow.
- `steer` [exact ✗ · FRR ✗]: The exam is ~~tomorrow.~~ **the following day.**


### success — idx 593

- **source**: The sun was shining.
- **target**: The sun ~~was shining.~~ **shone.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The sun ~~was shining.~~ **shone.**
- `ef32` [exact ✓ · FRR ✓]: The sun ~~was shining.~~ **shone.**
- `steer` [exact ✓ · FRR ✓]: The sun ~~was shining.~~ **shone.**


### near — idx 551

- **source**: John is a pirate.
- **target**: John ~~is~~ **works as** a pirate.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: John ~~is~~ **as as as as as as as as as as** a pirate.
- `ef32` [exact ✗ · FRR ✓]: John ~~is~~ **as as as as as as as as as as** a pirate.
- `steer` [exact ✗ · FRR ✗]: John is a ~~pirate.~~ **buccaneer.**


### near — idx 571

- **source**: His plan was risky.
- **target**: His plan ~~was risky.~~ **involved risk.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: His ~~plan was risky.~~ **involvement involves involving risk involvement involving of involving involvement of involvement involving.**
- `ef32` [exact ✗ · FRR ✓]: His ~~plan was risky.~~ **involvement involves involving risk involvement involving of involving involvement of involvement involving.**
- `steer` [exact ✗ · FRR ✗]: His plan was ~~risky.~~ **a risk.**


### fail — idx 582

- **source**: The answer is five.
- **target**: ~~The answer is five.~~ **Five equals the answer.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~The~~ **Equal equal equal** answer ~~is~~ **five five equal equal equal equal equal equal equal five equal equal equal equal equal equal equal five equal equal equal** five.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## count_nouns

### near — idx 621

- **source**: She ate two slices of pizza.
- **target**: She ate ~~two slices of pizza.~~ **pizza portions.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ate ~~two slices of pizza.~~ **portions,,,, and was was, as as was was also described considered also in, as described well as was described by in in.**
- `ef32` [exact ✗ · FRR ✓]: She ate ~~two slices of pizza.~~ **portions,,,, and was was, as as was was also described considered also in, as described well as was described by in in.**
- `steer` [exact ✗ · FRR ✗]: She ~~ate~~ **consumed** two slices of pizza.


### near — idx 639

- **source**: I need three forks and spoons.
- **target**: I need ~~three forks and spoons.~~ **cutlery items.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I need ~~three forks and spoons.~~ **to items, items.**
- `ef32` [exact ✗ · FRR ✓]: I need ~~three forks and spoons.~~ **to items, items.**
- `steer` [exact ✗ · FRR ✓]: I need three forks ~~and spoons.~~ **and.**


### fail — idx 601

- **source**: She placed three books on the table.
- **target**: She placed ~~three books~~ **book material** on the table.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: She placed ~~three~~ **material** books on the table.
- `ef32` [exact ✗ · FRR ✓]: She placed ~~three~~ **material** books on the table.
- `steer` [exact ✗ · FRR ✗]: She placed **the** three books on the table.


### fail — idx 640

- **source**: We spotted three squirrels in the tree.
- **target**: We spotted ~~three squirrels~~ **sciurid creatures** in the tree.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: We spotted ~~three~~ **the the CRCR Sci Sci squirrel Sci squirrel and sci Sci Sci Sci Sci Sci Sci Sci Sci Sci Sci Sci Sci sci Sci** squirrels in **in squirrel in the** the tree.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## declaration

### success — idx 683

- **source**: The notary nullifies the partnership ceases operations.
- **target**: The ~~notary nullifies the~~ partnership ceases operations.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The ~~notary nullifies the~~ partnership ceases operations.
- `ef32` [exact ✓ · FRR ✓]: The ~~notary nullifies the~~ partnership ceases operations.
- `steer` [exact ✗ · FRR ✗]: ~~The notary nullifies the partnership ceases operations.~~ **rowspan rowspan**


### success — idx 699

- **source**: An academy confers an award highlights sustainability.
- **target**: An ~~academy confers an~~ award highlights sustainability.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: An ~~academy confers an~~ award highlights sustainability.
- `ef32` [exact ✓ · FRR ✓]: An ~~academy confers an~~ award highlights sustainability.
- `steer` [exact ✗ · FRR ✗]: ~~An academy confers an award highlights sustainability.~~ ******


### near — idx 676

- **source**: The mayor proclaims lands revert to public use.
- **target**: ~~The mayor proclaims lands~~ **Lands** revert to public use.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~The mayor proclaims lands revert~~ **Lands claimed** to public use.
- `ef32` [exact ✗ · FRR ✓]: ~~The mayor proclaims lands revert~~ **Lands claimed** to public use.
- `steer` [exact ✗ · FRR ✓]: The ~~mayor proclaims lands revert to public use.~~ **Lands von Public Use.**


### near — idx 684

- **source**: The decree states the defendant forfeits all assets.
- **target**: The ~~decree states the~~ defendant forfeits all assets.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~decree states the defendant forfeits all~~ assets.
- `ef32` [exact ✗ · FRR ✓]: The ~~decree states the defendant forfeits all~~ assets.
- `steer` [exact ✗ · FRR ✓]: ~~The decree states the defendant forfeits all assets.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### fail — idx 657

- **source**: I specify the ceasefire excludes coastal regions.
- **target**: ~~I specify the~~ **The** ceasefire excludes coastal regions.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~I specify the ceasefire excludes coastal regions.~~ **The**
- `ef32` [exact ✗ · FRR ✗]: ~~I specify the ceasefire excludes coastal regions.~~ **The**
- `steer` [exact ✓ · FRR ✓]: ~~I specify the~~ **The** ceasefire excludes coastal regions.


### fail — idx 666

- **source**: We ratify the pacts ensure mutual aid.
- **target**: ~~We ratify the~~ **The** pacts ensure mutual aid.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~We ratify the pacts ensure mutual aid.~~ **The**
- `ef32` [exact ✗ · FRR ✗]: ~~We ratify the pacts ensure mutual aid.~~ **The**
- `steer` [exact ✗ · FRR ✓]: We ratify the pacts **that** ensure mutual aid.

## degree_prefix

### success — idx 708

- **source**: The speaker was overenthusiastic.
- **target**: The speaker was ~~overenthusiastic.~~ **enthusiastic.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The speaker was ~~overenthusiastic.~~ **enthusiastic.**
- `ef32` [exact ✓ · FRR ✓]: The speaker was ~~overenthusiastic.~~ **enthusiastic.**
- `steer` [exact ✓ · FRR ✓]: The speaker was ~~overenthusiastic.~~ **enthusiastic.**


### success — idx 715

- **source**: This car is superfast.
- **target**: This car is ~~superfast.~~ **fast.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: This car is ~~superfast.~~ **fast.**
- `ef32` [exact ✓ · FRR ✓]: This car is ~~superfast.~~ **fast.**
- `steer` [exact ✗ · FRR ✓]: This ~~car~~ **vehicle** is ~~superfast.~~ **fast.**


### near — idx 729

- **source**: She felt underappreciated.
- **target**: She felt ~~underappreciated.~~ **appreciated.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~felt underappreciated.~~ **feltappreciated.**
- `ef32` [exact ✗ · FRR ✓]: She ~~felt underappreciated.~~ **feltappreciated.**
- `steer` [exact ✗ · FRR ✓]: She felt ~~underappreciated.~~ **valued.**


### near — idx 711

- **source**: The plan was overambitious.
- **target**: The plan was ~~overambitious.~~ **ambitious.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The plan ~~was overambitious.~~ **was.**
- `ef32` [exact ✗ · FRR ✓]: The plan ~~was overambitious.~~ **was.**
- `steer` [exact ✗ · FRR ✓]: The **ambitious** plan ~~was overambitious.~~ **was.**


### fail — idx 741

- **source**: The film depicts an archetypal hero.
- **target**: The film depicts ~~an archetypal~~ **a typical** hero.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The film depicts an ~~archetypal hero.~~ **ordinary life of a,, of a normal person,,,, a.**
- `ef32` [exact ✗ · FRR ✗]: The film depicts an ~~archetypal hero.~~ **ordinary life of a,, of a normal person,,,, a.**
- `steer` [exact ✗ · FRR ✗]: The film ~~depicts an archetypal hero.~~ **features a hero archetype.**

## deixis

### success — idx 769

- **source**: She enjoys hiking on weekends.
- **target**: ~~She~~ **Emma** enjoys hiking on weekends.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~She~~ **Emma** enjoys hiking on weekends.
- `ef32` [exact ✓ · FRR ✓]: ~~She~~ **Emma** enjoys hiking on weekends.
- `steer` [exact ✓ · FRR ✓]: ~~She~~ **Emma** enjoys hiking on weekends.


### success — idx 776

- **source**: You should arrive fifteen minutes early.
- **target**: ~~You~~ **John** should arrive fifteen minutes early.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~You~~ **John** should arrive fifteen minutes early.
- `ef32` [exact ✓ · FRR ✓]: ~~You~~ **John** should arrive fifteen minutes early.
- `steer` [exact ✓ · FRR ✓]: ~~You~~ **John** should arrive fifteen minutes early.


### near — idx 774

- **source**: We celebrated our anniversary together.
- **target**: ~~We~~ **David and Emma** celebrated ~~our~~ **the** anniversary together.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~We celebrated our anniversary together.~~ **The**
- `ef32` [exact ✗ · FRR ✓]: ~~We~~ **and** celebrated ~~our~~ **the and David David Emma and and David and Emma Emma the David and and Emma David and David David David and and Emma** anniversary **and, and David David** together.
- `steer` [exact ✗ · FRR ✓]: ~~We celebrated our anniversary together.~~ **The**


### near — idx 775

- **source**: We booked the conference room.
- **target**: ~~We~~ **Alice and Frank** booked ~~the~~ **a** conference room.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~We booked the conference room.~~ **A**
- `ef32` [exact ✗ · FRR ✓]: ~~We~~ **and** booked ~~the~~ **a a a F.. and F.. and. F and a. and F. F. F F and and F a** conference **and** room.
- `steer` [exact ✗ · FRR ✓]: ~~We booked the conference room.~~ **A**

## deontic

### success — idx 809

- **source**: He should buy the tickets.
- **target**: He ~~should buy~~ **buys** the tickets.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~should buy~~ **buys** the tickets.
- `ef32` [exact ✓ · FRR ✓]: He ~~should buy~~ **buys** the tickets.
- `steer` [exact ✓ · FRR ✓]: He ~~should buy~~ **buys** the tickets.


### success — idx 813

- **source**: She might visit her grandparents.
- **target**: She ~~might visit~~ **visits** her grandparents.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~might visit~~ **visits** her grandparents.
- `ef32` [exact ✓ · FRR ✓]: She ~~might visit~~ **visits** her grandparents.
- `steer` [exact ✗ · FRR ✗]: She ~~might visit~~ **possibly visits** her grandparents.


### near — idx 833

- **source**: She can take the bus.
- **target**: She ~~can take~~ **takes** the bus.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~can take~~ **gives** the bus.
- `ef32` [exact ✗ · FRR ✓]: She ~~can take~~ **gives** the bus.
- `steer` [exact ✓ · FRR ✓]: She ~~can take~~ **takes** the bus.


### near — idx 830

- **source**: We are supposed to help each other.
- **target**: We ~~are supposed to~~ help each other.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~are supposed to help~~ each other.
- `ef32` [exact ✗ · FRR ✓]: We ~~are supposed to help~~ each other.
- `steer` [exact ✗ · FRR ✓]: ~~We are supposed to help each other.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## direct_object

### near — idx 902

- **source**: She ate the cake.
- **target**: ~~She ate the cake.~~ **The cake was eaten by her.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~She ate~~ **was by the taken by the the the by the the the by the the by the was the by the by the the by the** the cake.
- `ef32` [exact ✗ · FRR ✓]: ~~She ate~~ **was by the taken by the the the by the the the by the the by the was the by the by the the by the** the cake.
- `steer` [exact ✗ · FRR ✗]: She ~~ate~~ **devoured** the cake.


### near — idx 914

- **source**: She found the keys.
- **target**: ~~She~~ **The keys were** found ~~the keys.~~ **by her.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~found the~~ **was was her was a her a her her her her her her her own own keyboard hands own keyboard own keyboard hands keyboard her hand by** keys.
- `ef32` [exact ✗ · FRR ✓]: She ~~found the~~ **was was her was a her a her her her her her her her own own keyboard hands own keyboard own keyboard hands keyboard her hand by** keys.
- `steer` [exact ✗ · FRR ✗]: She ~~found~~ **discovered** the keys.


### fail — idx 943

- **source**: She chooses the winner.
- **target**: ~~She chooses the winner.~~ **The winner is chosen by her.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: She ~~chooses~~ **selected** the winner.
- `ef32` [exact ✗ · FRR ✓]: ~~She chooses~~ **The 22 by012,, the01 by winner is** the ~~winner.~~ **by the a by a by by the a in by the the of the**
- `steer` [exact ✗ · FRR ✗]: She ~~chooses~~ **selected** the winner.

## directive

### success — idx 854

- **source**: I command students to memorize this formula.
- **target**: ~~I command students to~~ **Students** memorize this formula.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~I command students to~~ **Students** memorize this formula.
- `ef32` [exact ✓ · FRR ✓]: ~~I command students to~~ **Students** memorize this formula.
- `steer` [exact ✗ · FRR ✓]: ~~I command students to memorize this formula.~~ **Students:**


### success — idx 872

- **source**: It is prescribed the thermostat maintains 22°C.
- **target**: ~~It is prescribed the~~ **The** thermostat maintains 22°C.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~It is prescribed the~~ **The** thermostat maintains 22°C.
- `ef32` [exact ✓ · FRR ✓]: ~~It is prescribed the~~ **The** thermostat maintains 22°C.
- `steer` [exact ✗ · FRR ✗]: ~~It~~ **The thermostat** is prescribed ~~the thermostat maintains~~ **to maintain** 22°C.


### near — idx 857

- **source**: I require you to apologize immediately.
- **target**: ~~I require you to~~ **You** apologize immediately.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I ~~require~~ **knew** you to apologize immediately.
- `ef32` [exact ✗ · FRR ✓]: I ~~require~~ **knew** you to apologize immediately.
- `steer` [exact ✗ · FRR ✓]: ~~I~~ **You** require you to apologize immediately.


### near — idx 860

- **source**: I insist you finish the soup.
- **target**: ~~I insist you~~ **You** finish the soup.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~I~~ **You** insist you finish the soup.
- `ef32` [exact ✗ · FRR ✓]: ~~I~~ **You** insist you finish the soup.
- `steer` [exact ✓ · FRR ✓]: ~~I insist you~~ **You** finish the soup.

## discourse_markers

### success — idx 962

- **source**: However, it’s still worth a try.
- **target**: ~~However, it’s~~ **It’s** still worth a try.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~However,~~ it’s still worth a try.
- `ef32` [exact ✓ · FRR ✓]: ~~However,~~ it’s still worth a try.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 983

- **source**: Seriously, I’m not joking.
- **target**: ~~Seriously,~~ I’m not joking.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Seriously, I’m~~ **I am** not ~~joking.~~ **joking,**
- `ef32` [exact ✗ · FRR ✓]: ~~Seriously, I’m not joking.~~ **Seriously joking**
- `steer` [exact ✗ · FRR ✓]: ~~Seriously, I’m~~ **I am** not ~~joking.~~ **joking,**


### near — idx 955

- **source**: Seriously, this is the best option.
- **target**: ~~Seriously, this~~ **This** is the best option.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Seriously, this is the best option.~~ **This**
- `ef32` [exact ✗ · FRR ✓]: ~~Seriously, this is the best option.~~ **This**
- `steer` [exact ✓ · FRR ✓]: ~~Seriously, this~~ **This** is the best option.


### fail — idx 977

- **source**: Basically, it’s all about timing.
- **target**: ~~Basically, it’s~~ **It’s** all about timing.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Basically,~~ **Basically** it’s all about timing.
- `ef32` [exact ✗ · FRR ✗]: ~~Basically,~~ **Basically** it’s all about timing.
- `steer` [exact ✗ · FRR ✓]: ~~Basically, it’s all~~ **Basic** about ~~timing.~~ **timing**


### fail — idx 995

- **source**: By the way, did you finish the report?
- **target**: ~~By the way, did~~ **Did** you finish the report?

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~By the way, did~~ **said** you finish the ~~report?~~ **report.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## echo_questions

### near — idx 1029

- **source**: "What did he say?" What did he say? “Goodbye.”
- **target**: "What did he say?" ~~What did he say?~~ **He said** “Goodbye.”

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~"What did he say?" What did he say? “Goodbye.”~~ **He said goodbye**
- `ef32` [exact ✗ · FRR ✓]: "What ~~did~~ he ~~say?" What did~~ **said,.?" said,, and** he ~~say? “Goodbye.”~~ **said, " "I'm, "IGoodbye,**
- `steer` [exact ✗ · FRR ✓]: ~~"What did he say?" What did he say? “Goodbye.”~~ **He said goodbye**


### near — idx 1041

- **source**: "How many people attended?" How many people attended? Fifty.
- **target**: "How many people attended?" ~~How many~~ **Fifty** people ~~attended? Fifty.~~ **attended.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~"How many people attended?"~~ How many people ~~attended?~~ **attended?,"** Fifty.
- `ef32` [exact ✗ · FRR ✓]: "How many people attended?" ~~How many people attended?~~ **attended** Fifty.
- `steer` [exact ✗ · FRR ✓]: ~~"How many people attended?"~~ How many people ~~attended?~~ **attended?,"** Fifty.

## elliptical_sentences

### near — idx 1077

- **source**: What did you buy? Books.
- **target**: What did you buy? ~~Books.~~ **I bought books.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: What did you buy? ~~Books.~~ **I I bought I I I I I I I I bought I I bought I I bought I I I I I I I I I I I bought**
- `ef32` [exact ✗ · FRR ✓]: What did you buy? ~~Books.~~ **I I bought I I I I I I I I bought I I bought I I bought I I I I I I I I I I I bought**
- `steer` [exact ✗ · FRR ✓]: What ~~did you buy?~~ **bought?** Books.


### near — idx 1078

- **source**: I prefer tea, John coffee.
- **target**: I prefer tea, **and** John **prefers** coffee.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I prefer tea, **and I I do I do do I do do I do do I do do I, do I do, do not** John coffee.
- `ef32` [exact ✗ · FRR ✓]: I prefer tea, **and I I do I do do I do do I do do I do do I, do I do, do not** John coffee.
- `steer` [exact ✗ · FRR ✓]: ~~I prefer tea, John coffee.~~ **" " " " " " " " " " " " " " and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and**


### fail — idx 1065

- **source**: She plays more than I do.
- **target**: She plays more than I ~~do.~~ **play.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 1061

- **source**: Alice fixed the computer and Bob the printer.
- **target**: Alice fixed the computer and Bob **fixed** the printer.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~Alice fixed the computer and Bob the printer.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## emphatic_structure

### success — idx 1102

- **source**: You do look tired.
- **target**: You ~~do~~ look tired.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: You ~~do~~ look tired.
- `ef32` [exact ✓ · FRR ✓]: You ~~do~~ look tired.
- `steer` [exact ✗ · FRR ✓]: ~~You do look tired.~~ **rowspan rowspan rowspan**


### success — idx 1104

- **source**: She does enjoy painting.
- **target**: She ~~does enjoy~~ **enjoys** painting.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~does enjoy~~ **enjoys** painting.
- `ef32` [exact ✓ · FRR ✓]: She ~~does enjoy~~ **enjoys** painting.
- `steer` [exact ✓ · FRR ✓]: She ~~does enjoy~~ **enjoys** painting.


### near — idx 1128

- **source**: She did arrive early.
- **target**: She ~~did arrive~~ **arrived** early.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~did arrive~~ **joined** early.
- `ef32` [exact ✗ · FRR ✓]: She ~~did arrive~~ **joined** early.
- `steer` [exact ✓ · FRR ✓]: She ~~did arrive~~ **arrived** early.


### near — idx 1121

- **source**: He did mention the issue.
- **target**: He ~~did mention~~ **mentioned** the issue.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~did mention~~ **explained** the issue.
- `ef32` [exact ✗ · FRR ✓]: He ~~did mention~~ **explained** the issue.
- `steer` [exact ✗ · FRR ✓]: He ~~did mention~~ **brought** the ~~issue.~~ **issue to attention.**

## epistemic

### success — idx 1166

- **source**: They must arrive at 5 PM.
- **target**: They ~~must~~ **will** arrive at 5 PM.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They ~~must~~ **will** arrive at 5 PM.
- `ef32` [exact ✓ · FRR ✓]: They ~~must~~ **will** arrive at 5 PM.
- `steer` [exact ✓ · FRR ✓]: They ~~must~~ **will** arrive at 5 PM.


### success — idx 1181

- **source**: The dog must be barking loudly.
- **target**: The dog ~~must be~~ **is** barking loudly.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The dog ~~must be~~ **is** barking loudly.
- `ef32` [exact ✓ · FRR ✓]: The dog ~~must be~~ **is** barking loudly.
- `steer` [exact ✓ · FRR ✓]: The dog ~~must be~~ **is** barking loudly.


### near — idx 1168

- **source**: They could arrive at 5 PM.
- **target**: They ~~could~~ **will** arrive at 5 PM.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They ~~could~~ **will come** arrive at 5 PM.
- `ef32` [exact ✗ · FRR ✓]: They ~~could~~ **will come** arrive at 5 PM.
- `steer` [exact ✓ · FRR ✓]: They ~~could~~ **will** arrive at 5 PM.


### near — idx 1155

- **source**: She can’t be coming to the party tomorrow.
- **target**: She ~~can’t be~~ **is** coming to the party tomorrow.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~can’t be~~ **is not** coming to the party tomorrow.
- `ef32` [exact ✗ · FRR ✓]: She ~~can’t be~~ **is not** coming to the party tomorrow.
- `steer` [exact ✗ · FRR ✓]: She ~~can’t be~~ **is not** coming to the party tomorrow.

## euphemism

### near — idx 1206

- **source**: He crossed over.
- **target**: He ~~crossed over.~~ **died.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~crossed over.~~ **died in.**
- `ef32` [exact ✗ · FRR ✓]: He ~~crossed over.~~ **died in.**
- `steer` [exact ✓ · FRR ✓]: He ~~crossed over.~~ **died.**


### near — idx 1232

- **source**: He’s freshening up.
- **target**: He’s ~~freshening up.~~ **going to the bathroom.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He’s ~~freshening up.~~ **a the the the bathroom, going to to to to the to the the bathroom to the the bathroom the bathroom bathroom to the bathroom the.**
- `ef32` [exact ✗ · FRR ✓]: He’s ~~freshening up.~~ **a the the the bathroom, going to to to to the to the the bathroom to the the bathroom the bathroom bathroom to the bathroom the.**
- `steer` [exact ✗ · FRR ✓]: ~~He’s freshening up.~~ **He is going to the bathroom.**

## existential

### success — idx 1259

- **source**: There exists a solution for this equation.
- **target**: ~~There~~ **A solution** exists ~~a solution~~ for this equation.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~There~~ **A solution** exists ~~a solution~~ for this equation.
- `ef32` [exact ✗ · FRR ✓]: ~~There exists a solution for this equation.~~ **A water water is is a-waterwater which is is.**
- `steer` [exact ✓ · FRR ✓]: ~~There~~ **A solution** exists ~~a solution~~ for this equation.


### success — idx 1272

- **source**: There exist multiple pathways leading to the summit.
- **target**: ~~There exist multiple~~ **Multiple** pathways ~~leading~~ **lead** to the summit.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~There exist multiple~~ **Multiple** pathways ~~leading~~ **lead** to the summit.
- `ef32` [exact ✗ · FRR ✓]: ~~There exist multiple pathways~~ leading to **either** the ~~summit.~~ **route or or to or to.**
- `steer` [exact ✓ · FRR ✓]: ~~There exist multiple~~ **Multiple** pathways ~~leading~~ **lead** to the summit.


### near — idx 1274

- **source**: There present themselves three options clearly.
- **target**: ~~There~~ **Three options** present themselves ~~three options~~ clearly.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~There~~ **is** present themselves three options clearly.
- `ef32` [exact ✗ · FRR ✓]: ~~There~~ **is** present themselves three options clearly.
- `steer` [exact ✗ · FRR ✗]: There ~~present themselves~~ **are** three **clear** options ~~clearly.~~ **present.**


### near — idx 1275

- **source**: There occurs a chemical reaction spontaneously.
- **target**: ~~There occurs a~~ **A** chemical reaction **occurs** spontaneously.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~There occurs~~ **Spontaneously,** a chemical reaction ~~spontaneously.~~ **occurs.**
- `ef32` [exact ✗ · FRR ✓]: ~~There occurs a chemical reaction spontaneously.~~ **A person who has been taken spontaneously from.**
- `steer` [exact ✗ · FRR ✓]: ~~There occurs~~ **Spontaneously,** a chemical reaction ~~spontaneously.~~ **occurs.**


### fail — idx 1283

- **source**: There remain two possible outcomes plausible.
- **target**: ~~There remain two~~ **Two** possible outcomes **remain** plausible.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~There remain two possible~~ **Two plausible** outcomes ~~plausible.~~ **remain.**
- `ef32` [exact ✗ · FRR ✓]: ~~There~~ **Either** remain two ~~possible~~ outcomes plausible.
- `steer` [exact ✗ · FRR ✗]: ~~There remain two possible~~ **Two plausible** outcomes ~~plausible.~~ **remain.**


### fail — idx 1291

- **source**: There confront several challenges new employees.
- **target**: ~~There~~ **Several challenges** confront ~~several challenges~~ new employees.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~There confront several challenges~~ **Over some cases eight** new employees.
- `ef32` [exact ✗ · FRR ✗]: ~~There confront several challenges~~ **Over some cases eight** new employees.
- `steer` [exact ✗ · FRR ✗]: ~~There~~ **Several new employees** confront ~~several challenges new employees.~~ **challenges.**

## existential_quantifiers

### success — idx 1334

- **source**: We experienced some delays.
- **target**: We experienced ~~some~~ delays.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: We experienced ~~some~~ delays.
- `ef32` [exact ✓ · FRR ✓]: We experienced ~~some~~ delays.
- `steer` [exact ✗ · FRR ✓]: ~~We experienced some delays.~~ **rowspan rowspan rowspan**


### success — idx 1335

- **source**: I felt some relief.
- **target**: I felt ~~some~~ relief.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I felt ~~some~~ relief.
- `ef32` [exact ✓ · FRR ✓]: I felt ~~some~~ relief.
- `steer` [exact ✗ · FRR ✓]: ~~I felt some relief.~~ **rowspan rowspan rowspan**


### fail — idx 1339

- **source**: They detected some movement.
- **target**: They detected ~~some~~ movement.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~They detected some movement.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### fail — idx 1347

- **source**: He answered some questions.
- **target**: He answered ~~some~~ questions.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~He answered some questions.~~ **rowspan rowspan rowspan**

## expletive

### success — idx 1394

- **source**: There followed a long pause.
- **target**: ~~There followed a~~ **A** long ~~pause.~~ **pause followed.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~There followed a~~ **A** long ~~pause.~~ **pause followed.**
- `ef32` [exact ✗ · FRR ✓]: ~~There followed a long pause.~~ **A pause,,,, when I,,, and and I I when I, I, and I, I stopped, I and, stopped**
- `steer` [exact ✓ · FRR ✓]: ~~There followed a~~ **A** long ~~pause.~~ **pause followed.**


### success — idx 1371

- **source**: It is said honesty is the best policy.
- **target**: ~~It is said~~ **People say** honesty is the best policy.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~It is said~~ **people say** honesty is the best policy.
- `ef32` [exact ✓ · FRR ✓]: ~~It is said~~ **people say** honesty is the best policy.
- `steer` [exact ✓ · FRR ✓]: ~~It is said~~ **People say** honesty is the best policy.


### near — idx 1375

- **source**: It looks like rain.
- **target**: ~~It~~ **Rain** looks ~~like rain.~~ **imminent.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~It looks like~~ **storm imminent danger, danger, and,, and danger, and and immediately,,, and rain, and rain, and danger, and and** rain.
- `ef32` [exact ✗ · FRR ✓]: ~~It looks like~~ **storm imminent danger, danger, and,, and danger, and and immediately,,, and rain, and rain, and danger, and and** rain.
- `steer` [exact ✗ · FRR ✗]: It ~~looks like rain.~~ **appears to be raining.**


### near — idx 1368

- **source**: There was a loud crash.
- **target**: ~~There was a~~ **A** loud ~~crash.~~ **crash occurred.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~There was a loud crash.~~ **A woman screamed**
- `ef32` [exact ✗ · FRR ✓]: ~~There was a loud crash.~~ **A woman screamed**
- `steer` [exact ✓ · FRR ✓]: ~~There was a~~ **A** loud ~~crash.~~ **crash occurred.**

## expressive

### success — idx 1409

- **source**: I’m sorry the window remains open.
- **target**: ~~I’m sorry the~~ **The** window remains open.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~I’m sorry the~~ **The** window remains open.
- `ef32` [exact ✗ · FRR ✓]: ~~I’m sorry the window remains open.~~ **I open**
- `steer` [exact ✓ · FRR ✓]: ~~I’m sorry the~~ **The** window remains open.


### success — idx 1407

- **source**: We apologize that the train departs late.
- **target**: ~~We apologize that the~~ **The** train departs late.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~We apologize that the~~ **The** train departs late.
- `ef32` [exact ✓ · FRR ✓]: ~~We apologize that the~~ **The** train departs late.
- `steer` [exact ✗ · FRR ✗]: We ~~apologize that~~ **apologize,** the train departs late.


### near — idx 1405

- **source**: I’m grateful you water my plants.
- **target**: ~~I’m grateful you~~ **You** water my plants.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~I’m grateful you water my plants.~~ **You**
- `ef32` [exact ✗ · FRR ✓]: ~~I’m grateful you water my plants.~~ **You**
- `steer` [exact ✗ · FRR ✓]: ~~I’m grateful you water my plants.~~ **Water your plants**


### near — idx 1449

- **source**: We regret the road getting icy.
- **target**: ~~We regret the~~ **The** road ~~getting~~ **gets** icy.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~We regret the~~ **The** road getting icy.
- `ef32` [exact ✗ · FRR ✓]: ~~We regret the~~ **The** road getting icy.
- `steer` [exact ✗ · FRR ✓]: ~~We regret the~~ **The** road ~~getting~~ **got** icy.


### fail — idx 1433

- **source**: We apologize for the microphone feedbacking.
- **target**: ~~We apologize for the~~ **The** microphone ~~feedbacking.~~ **feedbacks.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~We apologize for the microphone feedbacking.~~ **The sentences:**
- `ef32` [exact ✗ · FRR ✗]: ~~We apologize for the microphone feedbacking.~~ **The radio stations feedback signals.**
- `steer` [exact ✗ · FRR ✗]: ~~We apologize for the microphone feedbacking.~~ **The sentences:**

## extraposition

### near — idx 1499

- **source**: It was lamented that prices increased.
- **target**: ~~It~~ **That prices increased** was ~~lamented that prices increased.~~ **lamented.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: It was lamented ~~that~~ **because** prices increased.
- `ef32` [exact ✗ · FRR ✓]: ~~It was lamented~~ **That is means to to** that prices ~~increased.~~ **was lamented.**
- `steer` [exact ✗ · FRR ✓]: It was lamented ~~that~~ **because** prices increased.


### near — idx 1459

- **source**: It was rumored that she would resign.
- **target**: ~~It was rumored that~~ **That** she would ~~resign.~~ **resign was rumored.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~It was rumored that~~ **Rumored to be resigning,** she ~~would resign.~~ **was.**
- `ef32` [exact ✗ · FRR ✓]: ~~It~~ **That** was rumored ~~that she would resign.~~ **that.**
- `steer` [exact ✗ · FRR ✓]: ~~It was rumored that~~ **Rumored to be resigning,** she ~~would resign.~~ **was.**


### fail — idx 1458

- **source**: It is crucial that everyone participates.
- **target**: ~~It~~ **That everyone participates** is ~~crucial that everyone participates.~~ **crucial.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: It is crucial ~~that~~ **for** everyone ~~participates.~~ **to participate.**
- `ef32` [exact ✗ · FRR ✓]: ~~It~~ **That** is ~~crucial~~ **is that,, which is is, and is is** that ~~everyone~~ **is is being used used is being is being which is being is being used** participates.
- `steer` [exact ✗ · FRR ✗]: It is crucial ~~that~~ **for** everyone ~~participates.~~ **to participate.**


### fail — idx 1465

- **source**: It is true that knowledge is power.
- **target**: ~~It is true that~~ **That** knowledge is ~~power.~~ **power is true.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~It is true~~ **That means** that knowledge **which** is ~~power.~~ **in is power of power power of power of knowledge which that is which is is in which is in in is in in in reality.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## factives

### near — idx 1532

- **source**: The realization that money was missing panicked him.
- **target**: ~~The realization that money was~~ **Money** missing panicked him.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~The realization that money was missing~~ panicked him.
- `ef32` [exact ✗ · FRR ✓]: ~~The realization that money was missing~~ panicked him.
- `steer` [exact ✗ · FRR ✓]: The ~~realization that money was missing panicked him.~~ **Money Missing Panic Him.**


### near — idx 1501

- **source**: The fact that we missed the train annoyed her.
- **target**: ~~The fact that we missed~~ **Missing** the train annoyed her.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~The fact that we missed~~ **Missing** the ~~train annoyed her.~~ **train,**
- `ef32` [exact ✗ · FRR ✓]: ~~The fact that we missed the train annoyed her.~~ **her**
- `steer` [exact ✗ · FRR ✓]: ~~The fact that we missed~~ **Missing** the ~~train annoyed her.~~ **train,**


### fail — idx 1513

- **source**: The claim that money grows on trees is false.
- **target**: ~~The claim that money grows~~ **Money growing** on trees is false.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~The claim that~~ **Time and and** money **and and and plant and and and and** grows **and and and and plant and and and and on and and and on and and and** on trees is false.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## first_conditional

### near — idx 1580

- **source**: If they're playing outside, close the window.
- **target**: ~~If they're~~ **They’re** playing outside, **so** close the window.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~If they're~~ **So they’re** playing outside, close the window.
- `ef32` [exact ✗ · FRR ✓]: ~~If they're playing outside,~~ **So,,'tt’ss,, so so so so playing, so so so that and so on, so so so that so that that of,** close the window.
- `steer` [exact ✗ · FRR ✓]: ~~If they're~~ **So they’re** playing outside, close the window.


### near — idx 1578

- **source**: If I'm feeling tired, I'll call you tomorrow.
- **target**: ~~If~~ I'm feeling tired, ~~I'll~~ **so I** call you tomorrow.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~If I'm feeling tired, I'll~~ **I will** call you tomorrow.
- `ef32` [exact ✗ · FRR ✓]: ~~If I'm feeling tired, I'll call you~~ **I am, and I I am am so I so I so, so so I am am so happy going, so and so so I and, so I am am am so** tomorrow.
- `steer` [exact ✗ · FRR ✓]: ~~If I'm feeling tired, I'll~~ **I will** call you tomorrow.


### fail — idx 1589

- **source**: If they've decided, they will inform us.
- **target**: ~~If they've~~ **They’ve** decided, **and** they ~~will~~ inform us.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: If ~~they've~~ **they’ve** decided, they will inform us.
- `ef32` [exact ✗ · FRR ✓]: If they've decided, ~~they will~~ **they’rerere and’ theyrerere’rerere're’rere** inform **of and and and of and** us.
- `steer` [exact ✗ · FRR ✗]: If ~~they've~~ **they’ve** decided, they will inform us.


### fail — idx 1552

- **source**: If it rains tomorrow, we will cancel the picnic.
- **target**: ~~If it~~ **It** rains tomorrow, **and** we ~~will~~ cancel the picnic.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: If it rains tomorrow, ~~we will~~ **we'll** cancel the picnic.
- `ef32` [exact ✗ · FRR ✓]: ~~If it~~ **And that and and and and and day and and and, and, and and that and that and that** rains **and rain rain** tomorrow, ~~we~~ **and and that that** will cancel the picnic.
- `steer` [exact ✗ · FRR ✗]: If it rains tomorrow, ~~we will~~ **we'll** cancel the picnic.

## futurates

### success — idx 1603

- **source**: They are starting the project on Monday.
- **target**: They ~~are starting~~ **will start** the project on Monday.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They ~~are starting~~ **will start** the project on Monday.
- `ef32` [exact ✓ · FRR ✓]: They ~~are starting~~ **will start** the project on Monday.
- `steer` [exact ✓ · FRR ✓]: They ~~are starting~~ **will start** the project on Monday.


### success — idx 1608

- **source**: She is finishing her report by Friday.
- **target**: She ~~is finishing~~ **will finish** her report by Friday.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~is finishing~~ **will finish** her report by Friday.
- `ef32` [exact ✓ · FRR ✓]: She ~~is finishing~~ **will finish** her report by Friday.
- `steer` [exact ✓ · FRR ✓]: She ~~is finishing~~ **will finish** her report by Friday.


### near — idx 1607

- **source**: I am visiting my grandparents during the holidays.
- **target**: I ~~am visiting~~ **will visit** my grandparents during the holidays.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I ~~am~~ **will go** visiting my grandparents during the holidays.
- `ef32` [exact ✗ · FRR ✓]: I ~~am~~ **will go** visiting my grandparents during the holidays.
- `steer` [exact ✓ · FRR ✓]: I ~~am visiting~~ **will visit** my grandparents during the holidays.

## future

### success — idx 1653

- **source**: He will teach chemistry.
- **target**: He ~~will teach~~ **teaches** chemistry.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~will teach~~ **teaches** chemistry.
- `ef32` [exact ✓ · FRR ✓]: He ~~will teach~~ **teaches** chemistry.
- `steer` [exact ✓ · FRR ✓]: He ~~will teach~~ **teaches** chemistry.


### success — idx 1652

- **source**: I will practice piano daily.
- **target**: I ~~will~~ practice piano daily.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I ~~will~~ practice piano daily.
- `ef32` [exact ✓ · FRR ✓]: I ~~will~~ practice piano daily.
- `steer` [exact ✗ · FRR ✓]: ~~I will practice piano daily.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 1676

- **source**: She’ll answer emails.
- **target**: ~~She’ll answer~~ **She answers** emails.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~She’ll answer emails.~~ **She asks.**
- `ef32` [exact ✗ · FRR ✓]: ~~She’ll answer emails.~~ **She asks.**
- `steer` [exact ✓ · FRR ✓]: ~~She’ll answer~~ **She answers** emails.


### near — idx 1668

- **source**: He’ll take a taxi.
- **target**: ~~He’ll take~~ **He takes** a taxi.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~He’ll take a taxi.~~ **He has**
- `ef32` [exact ✗ · FRR ✓]: ~~He’ll take a taxi.~~ **He has**
- `steer` [exact ✓ · FRR ✓]: ~~He’ll take~~ **He takes** a taxi.


### fail — idx 1685

- **source**: Are they announcing the winner?
- **target**: ~~Are~~ **Do** they ~~announcing~~ **announce** the winner?

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Are~~ **Do** they announcing the winner?
- `ef32` [exact ✗ · FRR ✗]: ~~Are~~ **Do** they announcing the winner?
- `steer` [exact ✗ · FRR ✗]: ~~Are~~ **Will** they ~~announcing~~ **announce** the winner?

## future_perfect

### success — idx 1704

- **source**: They will have eaten dinner together.
- **target**: They ~~will have eaten~~ **eat** dinner together.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They ~~will have eaten~~ **eat** dinner together.
- `ef32` [exact ✓ · FRR ✓]: They ~~will have eaten~~ **eat** dinner together.
- `steer` [exact ✓ · FRR ✓]: They ~~will have eaten~~ **eat** dinner together.


### success — idx 1727

- **source**: She will have painted beautiful pictures.
- **target**: She ~~will have painted~~ **paints** beautiful pictures.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~will have painted~~ **paints** beautiful pictures.
- `ef32` [exact ✓ · FRR ✓]: She ~~will have painted~~ **paints** beautiful pictures.
- `steer` [exact ✓ · FRR ✓]: She ~~will have painted~~ **paints** beautiful pictures.


### near — idx 1718

- **source**: Will he have played football?
- **target**: ~~Will~~ **Does** he ~~have played~~ **play** football?

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Will he~~ **Does Does play** have played football?
- `ef32` [exact ✗ · FRR ✓]: ~~Will he~~ **Does Does play** have played football?
- `steer` [exact ✓ · FRR ✓]: ~~Will~~ **Does** he ~~have played~~ **play** football?


### near — idx 1713

- **source**: Will you have called him every day?
- **target**: ~~Will~~ **Do** you ~~have called~~ **call** him every day?

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Will~~ **Do** you have called him every day?
- `ef32` [exact ✗ · FRR ✓]: ~~Will~~ **Do** you have called him every day?
- `steer` [exact ✓ · FRR ✓]: ~~Will~~ **Do** you ~~have called~~ **call** him every day?

## future_progressive

### success — idx 1771

- **source**: Will they be playing games?
- **target**: ~~Will~~ **Do** they ~~be playing~~ **play** games?

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Will~~ **Do** they ~~be playing~~ **play** games?
- `ef32` [exact ✗ · FRR ✓]: ~~Will they~~ **Do** be playing ~~games?~~ **games.**
- `steer` [exact ✓ · FRR ✓]: ~~Will~~ **Do** they ~~be playing~~ **play** games?


### success — idx 1774

- **source**: Will you be watching movies?
- **target**: ~~Will~~ **Do** you ~~be watching~~ **watch** movies?

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Will~~ **Do** you ~~be watching~~ **watch** movies?
- `ef32` [exact ✗ · FRR ✓]: ~~Will~~ you ~~be watching movies?~~ **do.**
- `steer` [exact ✓ · FRR ✓]: ~~Will~~ **Do** you ~~be watching~~ **watch** movies?


### near — idx 1776

- **source**: Will I be working here?
- **target**: ~~Will~~ **Do** I ~~be working~~ **work** here?

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Will I be~~ **Do you** working here?
- `ef32` [exact ✗ · FRR ✓]: ~~Will I be~~ **Do you** working here?
- `steer` [exact ✓ · FRR ✓]: ~~Will~~ **Do** I ~~be working~~ **work** here?


### near — idx 1778

- **source**: When will she be singing?
- **target**: When ~~will~~ **does** she ~~be singing?~~ **sing?**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: When ~~will~~ **does not sing** she be singing?
- `ef32` [exact ✗ · FRR ✓]: When ~~will~~ **does not sing** she be singing?
- `steer` [exact ✗ · FRR ✓]: When will she ~~be singing?~~ **sing?**

## given_known

### success — idx 1835

- **source**: That tree fell during storm.
- **target**: ~~That~~ **A** tree fell during storm.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~That~~ **A** tree fell during storm.
- `ef32` [exact ✓ · FRR ✓]: ~~That~~ **A** tree fell during storm.
- `steer` [exact ✗ · FRR ✗]: ~~That~~ **A** tree fell during **the** storm.


### success — idx 1846

- **source**: That car alarm keeps going off.
- **target**: ~~That~~ **A** car alarm keeps going off.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~That~~ **A** car alarm keeps going off.
- `ef32` [exact ✓ · FRR ✓]: ~~That~~ **A** car alarm keeps going off.
- `steer` [exact ✗ · FRR ✗]: ~~That~~ **The** car alarm keeps going off.


### near — idx 1812

- **source**: We should visit that museum tomorrow.
- **target**: We should visit ~~that~~ museum tomorrow.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~should~~ visit that museum tomorrow.
- `ef32` [exact ✗ · FRR ✓]: We ~~should~~ visit that museum tomorrow.
- `steer` [exact ✗ · FRR ✓]: ~~We should visit that museum tomorrow.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### fail — idx 1802

- **source**: We need to buy the milk today.
- **target**: We need to buy ~~the~~ milk today.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~We need to buy the milk today.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## hyperbole

### near — idx 1859

- **source**: The lecture lasted an eternity.
- **target**: The lecture lasted ~~an eternity.~~ **very long.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The lecture lasted ~~an eternity.~~ **very very long.**
- `ef32` [exact ✗ · FRR ✓]: The lecture lasted ~~an eternity.~~ **very very long.**
- `steer` [exact ✗ · FRR ✓]: The lecture ~~lasted an eternity.~~ **was lengthy.**


### near — idx 1898

- **source**: She runs like the wind.
- **target**: She runs ~~like the wind.~~ **very fast.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She runs ~~like the wind.~~ **very very very very fast very fast very fast,,,, very fast fast,,,,, very fast fast,,, and,.**
- `ef32` [exact ✗ · FRR ✓]: She runs ~~like the wind.~~ **very very very very fast very fast very fast,,,, very fast fast,,,,, very fast fast,,, and,.**
- `steer` [exact ✗ · FRR ✓]: She ~~runs like the wind.~~ **is fast.**

## imperative_sentence

### success — idx 1929

- **source**: Let's finish this by tomorrow.
- **target**: ~~Let's~~ **We should** finish this by tomorrow.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Let's~~ **We should** finish this by tomorrow.
- `ef32` [exact ✓ · FRR ✓]: ~~Let's~~ **We should** finish this by tomorrow.
- `steer` [exact ✓ · FRR ✓]: ~~Let's~~ **We should** finish this by tomorrow.


### near — idx 1913

- **source**: Stand up.
- **target**: ~~Stand~~ **You are required to stand** up.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Stand up.~~ ****to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to**
- `ef32` [exact ✗ · FRR ✓]: ~~Stand up.~~ **is a not to to a 1 1121 to to the 1 122 01 up to to.**
- `steer` [exact ✗ · FRR ✓]: ~~Stand up.~~ ****to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to**


### near — idx 1916

- **source**: Drive carefully.
- **target**: ~~Drive~~ **You must drive** carefully.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Drive~~ **must have** carefully.
- `ef32` [exact ✗ · FRR ✓]: ~~Drive~~ **must have** carefully.
- `steer` [exact ✗ · FRR ✓]: ~~Drive carefully.~~ **Please provide me with the context of the sentence.**

## indirect_speech

### near — idx 1953

- **source**: They mentioned they would arrive late.
- **target**: They ~~mentioned they would~~ **mentioned, “We will** arrive ~~late.~~ **late.”**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~They mentioned~~ **We will arrive late,”** they ~~would arrive late.~~ **said.”**
- `ef32` [exact ✗ · FRR ✓]: ~~They~~ **We** mentioned ~~they would arrive late.~~ **we' we'rere'rererere we going to'sre're'srere'ssres going here.**
- `steer` [exact ✗ · FRR ✓]: ~~They mentioned~~ **We will arrive late,”** they ~~would arrive late.~~ **said.”**


### near — idx 1992

- **source**: He announced he had been promoted.
- **target**: He ~~announced he had~~ **announced, “I have** been ~~promoted.~~ **promoted.”**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He announced he had been ~~promoted.~~ **promoted, " " "I "I''m "mmII'mm " "mmI'mI'm'mmm**
- `ef32` [exact ✗ · FRR ✓]: He announced he had been ~~promoted.~~ **promoted, " " "I "I''m "mmII'mm " "mmI'mI'm'mmm**
- `steer` [exact ✗ · FRR ✓]: ~~He announced he had been promoted.~~ **I waspromoted.**


### fail — idx 1956

- **source**: He warned that it might rain later.
- **target**: He ~~warned that it~~ **warned, “It** might rain ~~later.~~ **later.”**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: He warned that it might rain ~~later.~~ **later,, ' ' ' ', ' ' ' ' ' ' ' ' ' ', ' ' ' ' ' ' ' ' '**
- `ef32` [exact ✗ · FRR ✗]: He warned that it might rain ~~later.~~ **later,, ' ' ' ', ' ' ' ' ' ' ' ' ' ', ' ' ' ' ' ' ' ' '**
- `steer` [exact ✗ · FRR ✓]: He ~~warned that it~~ **cautioned, 'It** might rain ~~later.~~ **later.'**

## intensifiers

### success — idx 2007

- **source**: They were totally exhausted after the hike.
- **target**: They were ~~totally~~ exhausted after the hike.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They were ~~totally~~ exhausted after the hike.
- `ef32` [exact ✓ · FRR ✓]: They were ~~totally~~ exhausted after the hike.
- `steer` [exact ✗ · FRR ✓]: ~~They were totally exhausted after the hike.~~ **rowspan rowspan rowspan**


### success — idx 2022

- **source**: He really wants to learn a new language this year.
- **target**: He ~~really~~ wants to learn a new language this year.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~really~~ wants to learn a new language this year.
- `ef32` [exact ✓ · FRR ✓]: He ~~really~~ wants to learn a new language this year.
- `steer` [exact ✗ · FRR ✓]: ~~He really wants to learn a new language this year.~~ **rowspan rowspan rowspan**


### near — idx 2035

- **source**: I don’t very much like spicy food, to be honest.
- **target**: I don’t ~~very much~~ like spicy food, to be honest.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I don’t ~~very~~ much like spicy food, to be honest.
- `ef32` [exact ✗ · FRR ✓]: I don’t ~~very~~ much like spicy food, to be honest.
- `steer` [exact ✗ · FRR ✓]: ~~I don’t very much like spicy food, to be honest.~~ **rowspan rowspan rowspan**


### fail — idx 2043

- **source**: They were absolutely thrilled to meet their idol.
- **target**: They were ~~absolutely~~ thrilled to meet their idol.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~They were absolutely thrilled to meet their idol.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### fail — idx 2012

- **source**: I really need to finish this work by noon.
- **target**: I ~~really~~ need to finish this work by noon.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~I really need to finish this work by noon.~~ **rowspan rowspan rowspan**

## interrogative

### success — idx 2064

- **source**: Were they celebrating their anniversary?
- **target**: ~~Were they~~ **They were** celebrating their ~~anniversary?~~ **anniversary.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Were they~~ **They were** celebrating their ~~anniversary?~~ **anniversary.**
- `ef32` [exact ✗ · FRR ✓]: ~~Were they celebrating~~ **They celebrate** their ~~anniversary?~~ **anniversary.**
- `steer` [exact ✓ · FRR ✓]: ~~Were they~~ **They were** celebrating their ~~anniversary?~~ **anniversary.**


### success — idx 2052

- **source**: Does the engine require regular maintenance?
- **target**: ~~Does the~~ **The** engine ~~require~~ **requires** regular ~~maintenance?~~ **maintenance.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Does the~~ **The** engine ~~require~~ **requires** regular ~~maintenance?~~ **maintenance.**
- `ef32` [exact ✗ · FRR ✓]: ~~Does the~~ **The** engine ~~require regular maintenance?~~ **provides.. The 111111,11,,,1, 11,,, responsible for the.**
- `steer` [exact ✓ · FRR ✓]: ~~Does the~~ **The** engine ~~require~~ **requires** regular ~~maintenance?~~ **maintenance.**


### near — idx 2091

- **source**: When was the novel written?
- **target**: ~~When~~ **The novel** was **written during** the ~~novel written?~~ **19th century.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~When was the novel written?~~ **The 119191 119199191119991.19 111919**
- `ef32` [exact ✗ · FRR ✓]: ~~When was the novel written?~~ **The 119191 119199191119991.19 111919**
- `steer` [exact ✗ · FRR ✓]: ~~When was the novel written?~~ **19th century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century**


### near — idx 2095

- **source**: Why did the experiment fail?
- **target**: ~~Why did the~~ **The** experiment ~~fail?~~ **failed because of contaminated samples.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Why did the experiment fail?~~ **The.**
- `ef32` [exact ✗ · FRR ✓]: ~~Why did~~ **The sampleThe sample of of of of of** the ~~experiment fail?~~ **the the of the the the the the of the the the the the the sample of. The sample**
- `steer` [exact ✗ · FRR ✓]: ~~Why did the experiment fail?~~ **The.**

## intransitive_verb

### near — idx 2103

- **source**: He resigned.
- **target**: He ~~resigned.~~ **quit his job.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~resigned.~~ **his job job job job his job job job job job job job job job job job job his job job job job job job job job job.**
- `ef32` [exact ✗ · FRR ✓]: He ~~resigned.~~ **his job job job job his job job job job job job job job job job job job his job job job job job job job job job.**
- `steer` [exact ✗ · FRR ✓]: He ~~resigned.~~ **was laid off.**


### near — idx 2115

- **source**: The dancer danced.
- **target**: The dancer ~~danced.~~ **began a dance performance.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The ~~dancer danced.~~ **dancer's dance.**
- `ef32` [exact ✗ · FRR ✓]: The dancer ~~danced.~~ **danced a a a performance performance performance a performance performance performance performance performance performance performance performance performance performance performance performance. a performance performance performance performance performance**
- `steer` [exact ✗ · FRR ✓]: The ~~dancer danced.~~ **dancer's dance.**

## linking_verb

### near — idx 2165

- **source**: The child appears hungry.
- **target**: The child ~~appears hungry.~~ **asks for food.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The child ~~appears hungry.~~ **asked for food for.**
- `ef32` [exact ✗ · FRR ✓]: The child ~~appears hungry.~~ **asked for food for.**
- `steer` [exact ✗ · FRR ✗]: The child ~~appears~~ **is** hungry.


### near — idx 2186

- **source**: The project became challenging.
- **target**: ~~The project became challenging.~~ **Obstacles emerged during the process.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~The project became challenging.~~
- `ef32` [exact ✗ · FRR ✓]: ~~The project became challenging.~~
- `steer` [exact ✗ · FRR ✓]: The ~~project became challenging.~~ **sentences below.**


### fail — idx 2151

- **source**: The sky is blue.
- **target**: The sky ~~is blue.~~ **reflects blue light.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: The ~~sky is blue.~~ **light blue light.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 2184

- **source**: The sky grew cloudy.
- **target**: ~~The sky grew cloudy.~~ **Clouds obscured the sky.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The sky ~~grew~~ **became** cloudy.
- `ef32` [exact ✗ · FRR ✓]: ~~The sky grew cloudy.~~ **the sky, the the blindness, cloud of, the, the, the, the blindness, the blindness clouds above the obstacles ocean vision,,.**
- `steer` [exact ✗ · FRR ✗]: The sky ~~grew~~ **became** cloudy.

## mass_noun

### success — idx 2227

- **source**: He has evidence to support his claim.
- **target**: He has ~~evidence~~ **facts** to support his claim.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He has ~~evidence~~ **facts** to support his claim.
- `ef32` [exact ✓ · FRR ✓]: He has ~~evidence~~ **facts** to support his claim.
- `steer` [exact ✗ · FRR ✗]: He has **supporting** evidence ~~to support~~ **for** his claim.


### success — idx 2204

- **source**: She showed great patience with the struggling student.
- **target**: She ~~showed great patience~~ **was very patient** with the struggling student.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: She ~~showed great patience~~ **was very patient** with the struggling student.
- `ef32` [exact ✗ · FRR ✓]: She ~~showed great patience~~ **was was very very very was very very very very very was very very very very very very very very very very very very very very very very very** with **very very** the **very** struggling student.
- `steer` [exact ✓ · FRR ✓]: She ~~showed great patience~~ **was very patient** with the struggling student.


### near — idx 2232

- **source**: The recipe calls for rice and vegetables.
- **target**: The recipe calls for ~~rice~~ **a grain** and vegetables.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The recipe calls for ~~rice and vegetables.~~ **a grain a.**
- `ef32` [exact ✗ · FRR ✓]: The recipe calls for ~~rice and vegetables.~~ **a grain a.**
- `steer` [exact ✗ · FRR ✓]: The recipe calls for ~~rice~~ **a mix of grains** and vegetables.


### near — idx 2222

- **source**: He studied literature as part of his degree.
- **target**: He studied ~~literature~~ **novels and poems** as part of his degree.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He studied ~~literature~~ **poetry and and and poetry and poetry poems of and poems poems poetry and poetry and as poems and of and poems poetry poetry and poems and as** as part of his degree.
- `ef32` [exact ✗ · FRR ✓]: He studied ~~literature~~ **poetry and and and poetry and poetry poems of and poems poems poetry and poetry and as poems and of and poems poetry poetry and poems and as** as part of his degree.
- `steer` [exact ✗ · FRR ✗]: He ~~studied literature as part~~ **delved into the realms** of **literature, pursuing their study within** his ~~degree.~~ **academic pursuits.**


### fail — idx 2244

- **source**: He studied music theory in college.
- **target**: He studied ~~music theory~~ **song structures** in college.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: He ~~studied~~ **learned about** music ~~theory~~ in college.
- `ef32` [exact ✗ · FRR ✓]: He studied ~~music theory~~ **song song song song song song song song song song song song singing and singing song songs and songs singing** in **the songs songs in song the song song song** college.
- `steer` [exact ✗ · FRR ✗]: He ~~studied~~ **learned about** music ~~theory~~ in college.


### fail — idx 2202

- **source**: The report contains valuable information about trends.
- **target**: The report ~~contains~~ **lists** valuable ~~information~~ **details** about trends.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The report ~~contains~~ **details** valuable information about trends.
- `ef32` [exact ✗ · FRR ✗]: The report ~~contains~~ **details** valuable information about trends.
- `steer` [exact ✗ · FRR ✗]: The report ~~contains~~ **details** valuable information about trends.

## metaphor

### near — idx 2283

- **source**: Patience's hourglass emptied.
- **target**: ~~Patience's hourglass emptied.~~ **The audience grew restless.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Patience's hourglass emptied.~~ **The audience''''The rest of the the the audience''''' of the the rest of of the of the the audience'''' interest in**
- `ef32` [exact ✗ · FRR ✓]: ~~Patience's hourglass emptied.~~ **The audience''''The rest of the the the audience''''' of the the rest of of the of the the audience'''' interest in**
- `steer` [exact ✗ · FRR ✓]: ~~Patience's hourglass emptied.~~ **The crowd grew restless.**


### near — idx 2268

- **source**: Sunrise painted heaven's canvas.
- **target**: ~~Sunrise painted heaven's canvas.~~ **Dawn colored the sky.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: Sunrise painted ~~heaven's canvas.~~ **the sky.**
- `ef32` [exact ✗ · FRR ✓]: Sunrise ~~painted heaven's canvas.~~ **heaven color of of the of the of the the of the the the the the color of of the morning sky color of and the the color of of of the of the of the.**
- `steer` [exact ✗ · FRR ✓]: Sunrise painted ~~heaven's canvas.~~ **the sky.**

## middle_verb

### success — idx 2328

- **source**: This wood carves smoothly with sharp tools.
- **target**: This wood ~~carves~~ **is carved** smoothly with sharp tools.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: This wood ~~carves~~ **is carved** smoothly with sharp tools.
- `ef32` [exact ✗ · FRR ✓]: This wood ~~carves smoothly with sharp tools.~~ **is is is is a is a is a very is a very a very a smoothly.**
- `steer` [exact ✓ · FRR ✓]: This wood ~~carves~~ **is carved** smoothly with sharp tools.


### near — idx 2304

- **source**: Fresh bread smells wonderful when baking.
- **target**: Fresh bread ~~smells~~ **is smelled as** wonderful when baking.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Fresh bread ~~smells~~ **is is as is a is as a as is a a as a is a as baked as as a as fresh as as a as as** wonderful when baking.
- `ef32` [exact ✗ · FRR ✓]: Fresh bread ~~smells~~ **is is as is a is as a as is a a as a is a as baked as as a as fresh as as a as as** wonderful when baking.
- `steer` [exact ✗ · FRR ✓]: Fresh bread smells wonderful when **it is** baking.


### near — idx 2333

- **source**: Fresh bread toasts perfectly in a classic toaster.
- **target**: Fresh bread ~~toasts~~ **is toasted** perfectly in a classic toaster.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Fresh bread ~~toasts~~ **is is is is is made in is made in in is made in in in is made in in in is made in in a** perfectly in a classic toaster.
- `ef32` [exact ✗ · FRR ✓]: Fresh bread ~~toasts~~ **is is is is is made in is made in in is made in in in is made in in in is made in in a** perfectly in a classic toaster.
- `steer` [exact ✗ · FRR ✗]: Fresh bread ~~toasts~~ **is** perfectly **toasted** in a classic toaster.


### fail — idx 2313

- **source**: Fresh vegetables cook quickly in a steamer.
- **target**: Fresh vegetables ~~cook~~ **are cooked** quickly in a steamer.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: Fresh vegetables **are** cook quickly in a steamer.
- `ef32` [exact ✗ · FRR ✗]: Fresh vegetables **are** cook quickly in a steamer.
- `steer` [exact ✓ · FRR ✗]: Fresh vegetables ~~cook~~ **are cooked** quickly in a steamer.


### fail — idx 2331

- **source**: Leather jackets wear comfortably in cool weather.
- **target**: Leather jackets ~~wear~~ **are worn** comfortably in cool weather.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: Leather jackets ~~wear~~ **are are are are are are used** comfortably in cool weather.
- `ef32` [exact ✗ · FRR ✗]: Leather jackets ~~wear~~ **are are are are are are used** comfortably in cool weather.
- `steer` [exact ✗ · FRR ✓]: Leather jackets ~~wear comfortably~~ **are comfortable** in cool weather.

## negation_prefix

### success — idx 2356

- **source**: He is an inexperienced driver.
- **target**: He is an ~~inexperienced~~ **experienced** driver.

- `ours` via ef32 [exact ✓ · FRR —] ←focus: He is an ~~inexperienced~~ **experienced** driver.
- `ef32` [exact ✓ · FRR —]: He is an ~~inexperienced~~ **experienced** driver.
- `steer` [exact ✗ · FRR —]: He is ~~an inexperienced~~ **a seasoned** driver.


### success — idx 2372

- **source**: He is an illogical thinker.
- **target**: He is ~~an illogical~~ **a logical** thinker.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He is ~~an illogical~~ **a logical** thinker.
- `ef32` [exact ✓ · FRR ✓]: He is ~~an illogical~~ **a logical** thinker.
- `steer` [exact ✓ · FRR ✓]: He is ~~an illogical~~ **a logical** thinker.


### fail — idx 2381

- **source**: The team is disorganized.
- **target**: The team is ~~disorganized.~~ **organized.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The team is ~~disorganized.~~ **organized in by the a by team a in the by the the the a team order of by the the a of the the the to a**
- `ef32` [exact ✗ · FRR ✗]: The team is ~~disorganized.~~ **organized in by the a by team a in the by the the the a team order of by the the a of the the the to a**
- `steer` [exact ✗ · FRR ✗]: The team is ~~disorganized.~~ **unorganized.**


### fail — idx 2384

- **source**: The car is uninsured.
- **target**: The car is ~~uninsured.~~ **insured.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: The car is ~~uninsured.~~ **not insured.**

## nominal_adverbials

### success — idx 2436

- **source**: She wrote quickly to meet the deadline.
- **target**: She wrote ~~quickly~~ **with speed** to meet the deadline.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She wrote ~~quickly~~ **with speed** to meet the deadline.
- `ef32` [exact ✓ · FRR ✓]: She wrote ~~quickly~~ **with speed** to meet the deadline.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 2420

- **source**: She studied abroad last semester.
- **target**: She studied ~~abroad~~ **in a foreign country** last semester.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She studied ~~abroad~~ **a a school foreign country country a school country foreign country school or in a a school country school in a a school foreign country a country** last semester.
- `ef32` [exact ✗ · FRR ✓]: She studied ~~abroad~~ **a a school foreign country country a school country foreign country school or in a a school country school in a a school foreign country a country** last semester.
- `steer` [exact ✗ · FRR ✓]: She studied ~~abroad~~ **foreign country** last semester.


### near — idx 2443

- **source**: We picnicked downtown on Saturday.
- **target**: We picnicked **in the** downtown **district** on Saturday.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~picnicked~~ **pic verynicked** downtown on Saturday.
- `ef32` [exact ✗ · FRR ✓]: We ~~picnicked~~ **pic verynicked** downtown on Saturday.
- `steer` [exact ✗ · FRR ✓]: ~~We picnicked downtown on Saturday.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 2414

- **source**: They hiked uphill for miles.
- **target**: They hiked ~~uphill~~ **up the hill** for **several** miles.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: They ~~hiked uphill for miles.~~ **trekked miles up the hill.**
- `ef32` [exact ✗ · FRR ✓]: They hiked ~~uphill~~ **the the the the up the the the the the other other up the the the the other town the city the other other end the the streets of** for **the the the** miles.
- `steer` [exact ✗ · FRR ✗]: They ~~hiked uphill for miles.~~ **trekked miles up the hill.**


### fail — idx 2413

- **source**: He slept late on Saturday morning.
- **target**: He slept **until** late on Saturday morning.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~He slept late on Saturday morning.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✗]: He ~~slept late~~ **slept, until the very late,, he, and,, to, ( to he has, has not, and and, and has** on Saturday ~~morning.~~ **morning,.**
- `steer` [exact ✗ · FRR ✗]: ~~He slept late on Saturday morning.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## nominal_suffix

### success — idx 2462

- **source**: She showed great kindness to strangers.
- **target**: She showed great ~~kindness~~ **kind** to strangers.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She showed great ~~kindness~~ **kind** to strangers.
- `ef32` [exact ✓ · FRR ✓]: She showed great ~~kindness~~ **kind** to strangers.
- `steer` [exact ✗ · FRR ✗]: She ~~showed~~ **exhibited** great kindness to strangers.


### success — idx 2473

- **source**: He found a duckling near the pond.
- **target**: He found a ~~duckling~~ **duck** near the pond.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He found a ~~duckling~~ **duck** near the pond.
- `ef32` [exact ✓ · FRR ✓]: He found a ~~duckling~~ **duck** near the pond.
- `steer` [exact ✗ · FRR ✓]: ~~He found a duckling near the pond.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 2461

- **source**: His argument was convincing.
- **target**: His ~~argument~~ **argue** was convincing.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: His ~~argument~~ **arguments** was convincing.
- `ef32` [exact ✗ · FRR ✓]: His ~~argument~~ **arguments** was convincing.
- `steer` [exact ✗ · FRR ✗]: His argument was ~~convincing.~~ **compelling.**


### near — idx 2485

- **source**: He gave a convincing explanation.
- **target**: He gave a convincing ~~explanation.~~ **explain.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He gave a convincing ~~explanation.~~ **recount.**
- `ef32` [exact ✗ · FRR ✓]: He gave a convincing ~~explanation.~~ **recount.**
- `steer` [exact ✗ · FRR ✗]: He ~~gave~~ **provided** a convincing explanation.


### fail — idx 2495

- **source**: She majors in linguistics.
- **target**: She majors in ~~linguistics.~~ **lingual.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: She ~~majors~~ **is majoring** in linguistics.


### fail — idx 2451

- **source**: The teacher encouraged every student.
- **target**: The ~~teacher~~ **teach** encouraged every student.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The ~~teacher~~ **teachers** encouraged every student.
- `ef32` [exact ✗ · FRR ✗]: The ~~teacher~~ **teachers** encouraged every student.
- `steer` [exact ✗ · FRR ✗]: The ~~teacher~~ **teachers** encouraged every student.

## non_defining_relative_clauses

### near — idx 2509

- **source**: The beach, where we met, holds special memories.
- **target**: The ~~beach, where we met,~~ **beach** holds special memories. **We met there.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~The beach, where we met, holds special memories.~~ **The.. They'rere here.........re.. There.. There...'tt..'ts**
- `ef32` [exact ✗ · FRR ✓]: ~~The beach, where we met, holds special memories.~~ **The.. They'rere here.........re.. There.. There...'tt..'ts**
- `steer` [exact ✗ · FRR ✓]: The ~~beach,~~ **beach** where we ~~met, holds special memories.~~ **met.**


### near — idx 2528

- **source**: The hotel, where we stayed, offered free breakfast.
- **target**: The ~~hotel, where we stayed,~~ **hotel** offered free breakfast. **We stayed there.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The ~~hotel,~~ **hotel** where we ~~stayed,~~ **stayed. It** offered free breakfast.
- `ef32` [exact ✗ · FRR ✓]: ~~The hotel, where we stayed, offered free breakfast.~~ **There''ss something there.....'tt ... It''s'ss. There'ss something**
- `steer` [exact ✗ · FRR ✓]: The ~~hotel,~~ **hotel** where we ~~stayed,~~ **stayed. It** offered free breakfast.

## non_synecdoche_metonymy

### success — idx 2562

- **source**: The teacher graded forty minds.
- **target**: The teacher graded forty ~~minds.~~ **papers.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The teacher graded forty ~~minds.~~ **papers.**
- `ef32` [exact ✓ · FRR ✓]: The teacher graded forty ~~minds.~~ **papers.**
- `steer` [exact ✗ · FRR ✓]: The teacher graded forty ~~minds.~~ **student papers.**


### success — idx 2582

- **source**: The engineers designed electric vehicles.
- **target**: The ~~engineers~~ **company** designed electric vehicles.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The ~~engineers~~ **company** designed electric vehicles.
- `ef32` [exact ✓ · FRR ✓]: The ~~engineers~~ **company** designed electric vehicles.
- `steer` [exact ✓ · FRR ✓]: The ~~engineers~~ **company** designed electric vehicles.


### near — idx 2578

- **source**: The lab coats discovered a new element.
- **target**: The ~~lab coats~~ **university** discovered a new element.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~lab coats discovered a new element.~~ **University**
- `ef32` [exact ✗ · FRR ✓]: The ~~lab coats discovered a new element.~~ **University**
- `steer` [exact ✓ · FRR ✓]: The ~~lab coats~~ **university** discovered a new element.

## noun_clauses

### success — idx 2645

- **source**: I doubt that he will apologize.
- **target**: I doubt ~~that he will apologize.~~ **his apology.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I doubt ~~that he will apologize.~~ **his apology.**
- `ef32` [exact ✓ · FRR ✓]: I doubt ~~that he will apologize.~~ **his apology.**
- `steer` [exact ✗ · FRR ✓]: I doubt ~~that he~~ **his** will **to** apologize.


### near — idx 2604

- **source**: She explained why she left early.
- **target**: She explained ~~why~~ **the reason** she left early.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She explained ~~why~~ **the** she left early.
- `ef32` [exact ✗ · FRR ✓]: She explained ~~why~~ **the** she left early.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 2610

- **source**: I don’t understand why she cried.
- **target**: I don’t understand ~~why she cried.~~ **the cause of her tears.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I don’t understand ~~why she cried.~~ **the the cause of their their respective respective causes the their of their own respective causes of their.**
- `ef32` [exact ✗ · FRR ✓]: I don’t understand ~~why she cried.~~ **the the cause of their their respective respective causes the their of their own respective causes of their.**
- `steer` [exact ✗ · FRR ✓]: **The reason for her tears,** I ~~don’t understand why she cried.~~ **don confusion.**


### fail — idx 2605

- **source**: He doesn’t know how I solved it.
- **target**: He doesn’t know ~~how~~ **the method** I ~~solved~~ **used to solve** it.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: He ~~doesn’t~~ **doesn't** know how I solved it.
- `ef32` [exact ✗ · FRR ✓]: He doesn’t know ~~how~~ **the the method** I ~~solved it.~~ **use to use to to use the to to use the to the method to the method method of to use the use of the the.**
- `steer` [exact ✗ · FRR ✗]: He ~~doesn’t~~ **doesn't** know how I solved it.

## noun_plural

### success — idx 2658

- **source**: The chairs are comfortable.
- **target**: The ~~chairs~~ **chair** are comfortable.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The ~~chairs~~ **chair** are comfortable.
- `ef32` [exact ✓ · FRR ✓]: The ~~chairs~~ **chair** are comfortable.
- `steer` [exact ✗ · FRR ✓]: The chairs ~~are~~ **is** comfortable.


### success — idx 2686

- **source**: The wishes were granted.
- **target**: The ~~wishes~~ **wish** were granted.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The ~~wishes~~ **wish** were granted.
- `ef32` [exact ✓ · FRR ✓]: The ~~wishes~~ **wish** were granted.
- `steer` [exact ✗ · FRR ✗]: The ~~wishes~~ **desires** were granted.


### near — idx 2675

- **source**: The songs are popular.
- **target**: The ~~songs~~ **song** are popular.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~songs are~~ **song is** popular.
- `ef32` [exact ✗ · FRR ✓]: The ~~songs are~~ **song is** popular.
- `steer` [exact ✗ · FRR ✓]: The ~~songs are~~ **song is** popular.


### near — idx 2674

- **source**: The keys are in my pocket.
- **target**: The ~~keys~~ **key** are in my pocket.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~keys are~~ **key is** in my pocket.
- `ef32` [exact ✗ · FRR ✓]: The ~~keys are~~ **key is** in my pocket.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## object_expletives

### near — idx 2713

- **source**: I hate it when people interrupt others.
- **target**: I hate ~~it when~~ people ~~interrupt~~ **interrupting** others.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: I ~~hate it when people interrupt others.~~ **dislike interruptions.**
- `ef32` [exact ✗ · FRR ✓]: I ~~hate it when~~ **interrupt with** people ~~interrupt~~ others.
- `steer` [exact ✗ · FRR ✓]: I ~~hate it when people interrupt others.~~ **dislike interruptions.**


### near — idx 2702

- **source**: She made it clear that she prefers coffee.
- **target**: She made ~~it~~ clear ~~that she prefers~~ **her preference for** coffee.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~She made it clear that she prefers~~ **Her preference for** coffee.
- `ef32` [exact ✗ · FRR ✓]: ~~She made it~~ **For** clear ~~that she prefers~~ **reasons of preference for for the for the opinion of the opinion of the** coffee.
- `steer` [exact ✗ · FRR ✓]: ~~She made it clear that she prefers~~ **Her preference for** coffee.

## of_genitive

### success — idx 2756

- **source**: He lost the key of his.
- **target**: He lost ~~the key of his.~~ **his key.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: He lost ~~the key of his.~~ **his key.**
- `ef32` [exact ✗ · FRR ✓]: He lost ~~the key of his.~~ **his eye.**
- `steer` [exact ✓ · FRR ✓]: He lost ~~the key of his.~~ **his key.**


### success — idx 2772

- **source**: She will modify the recipe of hers.
- **target**: She will modify ~~the recipe of hers.~~ **her recipe.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: She will modify ~~the recipe of hers.~~ **her recipe.**
- `ef32` [exact ✓ · FRR ✓]: She will modify ~~the recipe of hers.~~ **her recipe.**
- `steer` [exact ✓ · FRR ✓]: She will modify ~~the recipe of hers.~~ **her recipe.**


### near — idx 2752

- **source**: I found a friend of hers.
- **target**: I found ~~a friend of hers.~~ **her friend.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I found ~~a friend of hers.~~ **her.**
- `ef32` [exact ✗ · FRR ✓]: I found ~~a friend of hers.~~ **her.**
- `steer` [exact ✓ · FRR ✓]: I found ~~a friend of hers.~~ **her friend.**


### near — idx 2758

- **source**: He appreciated the talent of hers.
- **target**: He appreciated ~~the talent of hers.~~ **her talent.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He appreciated ~~the talent of~~ **her and** hers.
- `ef32` [exact ✗ · FRR ✓]: He appreciated ~~the talent of~~ **her and** hers.
- `steer` [exact ✓ · FRR ✓]: He appreciated ~~the talent of hers.~~ **her talent.**


### fail — idx 2773

- **source**: He assembled the furniture of his.
- **target**: He assembled ~~the furniture of his.~~ **his furniture.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✓ · FRR ✓]: He assembled ~~the furniture of his.~~ **his furniture.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## optative

### success — idx 2822

- **source**: Would that fire spread quickly!
- **target**: ~~Would that fire spread quickly!~~ **Fire spreads quickly.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Would that fire spread quickly!~~ **Fire spreads quickly.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would that~~ fire ~~spread quickly!~~ **spread,, and is fire and and is,,,, and and fire and and,, and fire and and, and is in!.**
- `steer` [exact ✓ · FRR ✓]: ~~Would that fire spread quickly!~~ **Fire spreads quickly.**


### success — idx 2806

- **source**: Would that the sun rose early!
- **target**: ~~Would that the~~ **The** sun ~~rose early!~~ **rises early.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Would that the~~ **The** sun ~~rose early!~~ **rises early.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would that~~ **TheTheThe evening,, the,** the ~~sun rose early!~~ **the, and the, the, the sun, and and and the, and the, and the, and the**
- `steer` [exact ✓ · FRR ✓]: ~~Would that the~~ **The** sun ~~rose early!~~ **rises early.**


### near — idx 2826

- **source**: Would that owl hooted at midnight!
- **target**: ~~Would that owl hooted~~ **Owl hoots** at ~~midnight!~~ **midnight.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Would that owl hooted at midnight!~~ **.. Co. and and and and..... Co. and and and Co. and and and.. Co. and and and and Co.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would that owl hooted at midnight!~~ **.. Co. and and and and..... Co. and and and Co. and and and.. Co. and and and and Co.**
- `steer` [exact ✗ · FRR ✓]: Would ~~that~~ **the** owl ~~hooted~~ **hoots** at ~~midnight!~~ **midnight.**


### near — idx 2830

- **source**: Would that fog blanketed the valley!
- **target**: ~~Would that fog blanketed~~ **Fog blankets** the ~~valley!~~ **valley.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Would that~~ **The** fog ~~blanketed~~ **blankets** the ~~valley!~~ **valley.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would that fog blanketed~~ **Snow fog,,,,, and the, and** the **blan,, and, and the, and the, and the,keted, and and by the the the,** valley!
- `steer` [exact ✗ · FRR ✓]: ~~Would that~~ **The** fog ~~blanketed~~ **blankets** the ~~valley!~~ **valley.**


### fail — idx 2818

- **source**: Would that fish swam upstream!
- **target**: ~~Would that fish swam upstream!~~ **Fish swim upstream.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: Would that fish ~~swam upstream!~~ **swim upstream.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would that fish swam upstream!~~ **Swim swimming upstream,,,,, and, and,,,,, and swim swimming and swimming, and swimming swimming, and swimming and swimming**
- `steer` [exact ✗ · FRR ✗]: Would that fish ~~swam upstream!~~ **swim upstream.**

## passive_voice

### success — idx 2896

- **source**: The song was composed by the musician.
- **target**: The ~~song was~~ **musician** composed ~~by~~ the ~~musician.~~ **song.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: The ~~song was~~ **musician** composed ~~by~~ the ~~musician.~~ **song.**
- `ef32` [exact ✗ · FRR ✓]: The **singer wrote the** song ~~was composed by the musician.~~ **" " " " " music " " " " " " " " " " " " " " " " " " "The music " " " " "**
- `steer` [exact ✓ · FRR ✓]: The ~~song was~~ **musician** composed ~~by~~ the ~~musician.~~ **song.**


### success — idx 2862

- **source**: The message is being delivered by the postman.
- **target**: The ~~message~~ **postman** is ~~being delivered by~~ **delivering** the ~~postman.~~ **message.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: The ~~message~~ **postman** is ~~being delivered by~~ **delivering** the ~~postman.~~ **message.**
- `ef32` [exact ✗ · FRR ✓]: The ~~message~~ **shipping** is ~~being delivered by the postman.~~ **shipping.**
- `steer` [exact ✓ · FRR ✓]: The ~~message~~ **postman** is ~~being delivered by~~ **delivering** the ~~postman.~~ **message.**


### near — idx 2867

- **source**: The software has been updated by the technician.
- **target**: The ~~software~~ **technician** has ~~been~~ updated ~~by~~ the ~~technician.~~ **software.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The ~~software has been~~ **technician** updated ~~by~~ the ~~technician.~~ **software.**
- `ef32` [exact ✗ · FRR ✓]: The ~~software has been~~ **phone is phone phone** updated ~~by~~ **and and with and and-levellevel phone with phone and** the ~~technician.~~ **phone is phone and is phone phone really really is phone really with.**
- `steer` [exact ✗ · FRR ✓]: The ~~software has been~~ **technician** updated ~~by~~ the ~~technician.~~ **software.**


### near — idx 2879

- **source**: The mistake must be corrected by you immediately.
- **target**: ~~The~~ **You must correct the** mistake ~~must be corrected by you~~ immediately.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~The~~ **You must correct this** mistake ~~must be corrected by you~~ immediately.
- `ef32` [exact ✗ · FRR ✓]: ~~The mistake~~ **What you the problem you have want to have the the want the to have the want the the to have the problem can solve change the the the** must ~~be corrected by you immediately.~~ **the your.**
- `steer` [exact ✗ · FRR ✓]: ~~The~~ **You must correct this** mistake ~~must be corrected by you~~ immediately.


### fail — idx 2875

- **source**: The furniture is moved by the movers tomorrow.
- **target**: The **movers will move the** furniture ~~is moved by the movers~~ tomorrow.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The furniture ~~is~~ **will be** moved by the movers tomorrow.
- `ef32` [exact ✗ · FRR ✓]: The ~~furniture is moved by~~ **will take will to take it account** the ~~movers~~ **level amount of of to the take it account account of of the will to the the point it account account to the** tomorrow.
- `steer` [exact ✗ · FRR ✗]: The furniture ~~is~~ **will be** moved by the movers tomorrow.


### fail — idx 2854

- **source**: The movie will be shown at the theater next week.
- **target**: The **theater will show the** movie ~~will be shown at the theater~~ next week.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: The movie ~~will be~~ **is** shown ~~at the~~ **in to in to** theater ~~next week.~~ **and and and and theater is is is theater and show show and is show show is theater and and and and is show theater theater and and is.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## past

### success — idx 2912

- **source**: Children built sandcastles.
- **target**: Children ~~built~~ **build** sandcastles.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Children ~~built~~ **build** sandcastles.
- `ef32` [exact ✓ · FRR ✓]: Children ~~built~~ **build** sandcastles.
- `steer` [exact ✗ · FRR ✓]: Children ~~built~~ **construct** sandcastles.


### success — idx 2904

- **source**: She wrote poems daily.
- **target**: She ~~wrote~~ **writes** poems daily.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~wrote~~ **writes** poems daily.
- `ef32` [exact ✓ · FRR ✓]: She ~~wrote~~ **writes** poems daily.
- `steer` [exact ✓ · FRR ✓]: She ~~wrote~~ **writes** poems daily.


### near — idx 2942

- **source**: The kettle whistled when boiling.
- **target**: The kettle ~~whistled~~ **whistles** when boiling.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~kettle~~ **whistles** whistled when boiling.
- `ef32` [exact ✗ · FRR ✓]: The ~~kettle~~ **whistles** whistled when boiling.
- `steer` [exact ✓ · FRR ✓]: The kettle ~~whistled~~ **whistles** when boiling.


### near — idx 2945

- **source**: They rehearsed the play daily.
- **target**: They ~~rehearsed~~ **rehearse** the play daily.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They ~~rehearsed~~ **rehehearsed** the play daily.
- `ef32` [exact ✗ · FRR ✓]: They ~~rehearsed~~ **rehehearsed** the play daily.
- `steer` [exact ✓ · FRR ✓]: They ~~rehearsed~~ **rehearse** the play daily.


### fail — idx 2948

- **source**: Shadows lengthened at dusk.
- **target**: Shadows ~~lengthened~~ **lengthen** at dusk.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: Shadows ~~lengthened~~ **stretched** at dusk.


### fail — idx 2915

- **source**: Rivers flowed to the sea.
- **target**: Rivers ~~flowed~~ **flow** to the sea.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~Rivers flowed~~ **Water flows** to the sea.

## past_participle

### success — idx 2986

- **source**: He had brushed his teeth.
- **target**: He had ~~brushed~~ **brush** his teeth.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He had ~~brushed~~ **brush** his teeth.
- `ef32` [exact ✓ · FRR ✓]: He had ~~brushed~~ **brush** his teeth.
- `steer` [exact ✗ · FRR ✓]: He ~~had~~ brushed his teeth.


### success — idx 2970

- **source**: They have included everyone in the game.
- **target**: They have ~~included~~ **include** everyone in the game.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They have ~~included~~ **include** everyone in the game.
- `ef32` [exact ✓ · FRR ✓]: They have ~~included~~ **include** everyone in the game.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 2994

- **source**: She had folded the clothes.
- **target**: She had ~~folded~~ **fold** the clothes.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~had folded~~ **has fold** the clothes.
- `ef32` [exact ✗ · FRR ✓]: She ~~had folded~~ **has fold** the clothes.
- `steer` [exact ✗ · FRR ✓]: She ~~had~~ folded the clothes.


### near — idx 2974

- **source**: The artist had sketched a portrait.
- **target**: The artist had ~~sketched~~ **sketch** a portrait.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The artist ~~had sketched~~ **was sketch** a portrait.
- `ef32` [exact ✗ · FRR ✓]: The artist ~~had sketched~~ **was sketch** a portrait.
- `steer` [exact ✗ · FRR ✓]: The artist ~~had~~ sketched a portrait.


### fail — idx 2958

- **source**: I have visited that museum before.
- **target**: I have ~~visited~~ **visit** that museum before.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: I ~~have~~ **had** visited that museum before.
- `ef32` [exact ✗ · FRR ✗]: I ~~have~~ **had** visited that museum before.
- `steer` [exact ✗ · FRR ✗]: ~~I have~~ **I've** visited that museum before.

## past_participle_irregular

### near — idx 3031

- **source**: She has fallen asleep.
- **target**: She has ~~fallen~~ **fall** asleep.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~has~~ **spring** fallen asleep.
- `ef32` [exact ✗ · FRR ✓]: She ~~has~~ **spring** fallen asleep.
- `steer` [exact ✗ · FRR ✓]: She ~~has fallen~~ **is** asleep.


### near — idx 3015

- **source**: They have caught the bus.
- **target**: They have ~~caught~~ **catch** the bus.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~They have caught~~ **They' catch** the bus.
- `ef32` [exact ✗ · FRR ✓]: ~~They have caught~~ **They' catch** the bus.
- `steer` [exact ✗ · FRR ✓]: They ~~have~~ caught the bus.


### fail — idx 3018

- **source**: The meeting has been held.
- **target**: The meeting has been ~~held.~~ **hold.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The ~~meeting has~~ **meetings have** been held.
- `ef32` [exact ✗ · FRR ✗]: The ~~meeting has~~ **meetings have** been held.
- `steer` [exact ✗ · FRR ✓]: The meeting has ~~been held.~~ **taken place.**


### fail — idx 3020

- **source**: She has found the solution.
- **target**: She has ~~found~~ **find** the solution.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: She ~~has~~ **had** found the solution.
- `ef32` [exact ✗ · FRR ✗]: She ~~has~~ **had** found the solution.
- `steer` [exact ✗ · FRR ✓]: She ~~has~~ found the solution.

## past_perfect

### success — idx 3071

- **source**: I had eaten breakfast.
- **target**: I ~~had eaten~~ **eat** breakfast.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I ~~had eaten~~ **eat** breakfast.
- `ef32` [exact ✓ · FRR ✓]: I ~~had eaten~~ **eat** breakfast.
- `steer` [exact ✗ · FRR ✓]: I ~~had eaten~~ **ate** breakfast.


### success — idx 3077

- **source**: It had flown south.
- **target**: It ~~had flown~~ **flies** south.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: It ~~had flown~~ **flies** south.
- `ef32` [exact ✓ · FRR ✓]: It ~~had flown~~ **flies** south.
- `steer` [exact ✓ · FRR ✓]: It ~~had flown~~ **flies** south.


### near — idx 3051

- **source**: He had walked to school.
- **target**: He ~~had walked~~ **walks** to school.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~had walked~~ **goes** to school.
- `ef32` [exact ✗ · FRR ✓]: He ~~had walked~~ **goes** to school.
- `steer` [exact ✗ · FRR ✓]: He ~~had~~ walked to school.


### near — idx 3062

- **source**: The computer had processed data.
- **target**: The computer ~~had processed~~ **processes** data.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~computer had processed~~ **provides** data.
- `ef32` [exact ✗ · FRR ✓]: The ~~computer had processed~~ **provides** data.
- `steer` [exact ✓ · FRR ✓]: The computer ~~had processed~~ **processes** data.

## past_progressive

### success — idx 3131

- **source**: She was writing emails.
- **target**: She ~~was writing~~ **writes** emails.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~was writing~~ **writes** emails.
- `ef32` [exact ✓ · FRR ✓]: She ~~was writing~~ **writes** emails.
- `steer` [exact ✓ · FRR ✓]: She ~~was writing~~ **writes** emails.


### success — idx 3146

- **source**: Fans were recording videos.
- **target**: Fans ~~were recording~~ **record** videos.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Fans ~~were recording~~ **record** videos.
- `ef32` [exact ✓ · FRR ✓]: Fans ~~were recording~~ **record** videos.
- `steer` [exact ✓ · FRR ✓]: Fans ~~were recording~~ **record** videos.


### near — idx 3120

- **source**: We were compressing files.
- **target**: We ~~were compressing~~ **compress** files.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~were compressing files.~~
- `ef32` [exact ✗ · FRR ✓]: We ~~were compressing files.~~
- `steer` [exact ✗ · FRR ✓]: We ~~were compressing~~ **compressed** files.


### near — idx 3140

- **source**: Everyone was evacuating calmly.
- **target**: Everyone ~~was evacuating~~ **evacuates** calmly.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Everyone ~~was evacuating~~ **leaves** calmly.
- `ef32` [exact ✗ · FRR ✓]: Everyone ~~was evacuating~~ **leaves** calmly.
- `steer` [exact ✓ · FRR ✓]: Everyone ~~was evacuating~~ **evacuates** calmly.

## past_tense

### success — idx 3167

- **source**: She danced all night.
- **target**: She ~~danced~~ **dance** all night.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~danced~~ **dance** all night.
- `ef32` [exact ✓ · FRR ✓]: She ~~danced~~ **dance** all night.
- `steer` [exact ✗ · FRR ✗]: She danced all ~~night.~~ **night long.**


### success — idx 3156

- **source**: He worried about the results.
- **target**: He ~~worried~~ **worry** about the results.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~worried~~ **worry** about the results.
- `ef32` [exact ✓ · FRR ✓]: He ~~worried~~ **worry** about the results.
- `steer` [exact ✗ · FRR ✗]: He ~~worried~~ **fretted** about the results.


### near — idx 3180

- **source**: We continued the discussion.
- **target**: We ~~continued~~ **continue** the discussion.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~continued~~ **will discuss** the discussion.
- `ef32` [exact ✗ · FRR ✓]: We ~~continued~~ **will discuss** the discussion.
- `steer` [exact ✗ · FRR ✗]: We ~~continued~~ **resumed** the discussion.


### near — idx 3173

- **source**: He grabbed her bag and left.
- **target**: He ~~grabbed~~ **grab** her bag and left.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~grabbed~~ **grabs** her bag and left.
- `ef32` [exact ✗ · FRR ✓]: He ~~grabbed~~ **grabs** her bag and left.
- `steer` [exact ✗ · FRR ✗]: He ~~grabbed~~ **took** her bag and left.

## past_tense_irregular

### success — idx 3219

- **source**: She swam in the pool all afternoon.
- **target**: She ~~swam~~ **swim** in the pool all afternoon.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~swam~~ **swim** in the pool all afternoon.
- `ef32` [exact ✓ · FRR ✓]: She ~~swam~~ **swim** in the pool all afternoon.
- `steer` [exact ✗ · FRR ✗]: She swam in the pool all ~~afternoon.~~ **day.**


### success — idx 3224

- **source**: He flew to Paris for a vacation.
- **target**: He ~~flew~~ **fly** to Paris for a vacation.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~flew~~ **fly** to Paris for a vacation.
- `ef32` [exact ✓ · FRR ✓]: He ~~flew~~ **fly** to Paris for a vacation.
- `steer` [exact ✗ · FRR ✗]: He ~~flew~~ **went** to Paris for a vacation.


### near — idx 3204

- **source**: He took the book from the shelf.
- **target**: He ~~took~~ **take** the book from the shelf.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~took~~ **takes** the book from the shelf.
- `ef32` [exact ✗ · FRR ✓]: He ~~took~~ **takes** the book from the shelf.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 3207

- **source**: The team won the championship last year.
- **target**: The team ~~won~~ **win** the championship last year.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~team won~~ **teams win** the championship last year.
- `ef32` [exact ✗ · FRR ✓]: The ~~team won~~ **teams win** the championship last year.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 3205

- **source**: The children built a sandcastle at the beach.
- **target**: The children ~~built~~ **build** a sandcastle at the beach.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: The children ~~built~~ **construct** a sandcastle at the beach.

## personification

### near — idx 3327

- **source**: Hope quietly stirred.
- **target**: ~~Hope quietly stirred.~~ **People felt hopeful.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Hope quietly stirred.~~ **people excited to people's feelingre to of people for people''s.**
- `ef32` [exact ✗ · FRR ✓]: ~~Hope quietly stirred.~~ **people excited to people's feelingre to of people for people''s.**
- `steer` [exact ✗ · FRR ✓]: Hope ~~quietly stirred.~~ **felt hopeful.**


### near — idx 3301

- **source**: Time is flying by.
- **target**: Time is ~~flying by.~~ **passing quickly.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Time is ~~flying by.~~ **quickly passing.**
- `ef32` [exact ✗ · FRR ✓]: Time is ~~flying by.~~ **quickly passing.**
- `steer` [exact ✗ · FRR ✓]: Time ~~is flying~~ **slips** by.

## politeness

### success — idx 3391

- **source**: Please write your name clearly.
- **target**: ~~Please write~~ **Write** your name clearly.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Please write~~ **Write** your name clearly.
- `ef32` [exact ✓ · FRR ✓]: ~~Please write~~ **Write** your name clearly.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### success — idx 3390

- **source**: Would you kindly stay a little longer?
- **target**: ~~Would you kindly stay~~ **Stay** a little ~~longer?~~ **longer.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Would you kindly stay~~ **Stay** a little ~~longer?~~ **longer.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would you kindly stay~~ **Stay and stop** a ~~little longer?~~ **little.**
- `steer` [exact ✓ · FRR ✓]: ~~Would you kindly stay~~ **Stay** a little ~~longer?~~ **longer.**


### near — idx 3369

- **source**: Please listen carefully.
- **target**: ~~Please listen~~ **Listen** carefully.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Please listen~~ **Watch** carefully.
- `ef32` [exact ✗ · FRR ✓]: ~~Please listen~~ **Watch** carefully.
- `steer` [exact ✓ · FRR ✓]: ~~Please listen~~ **Listen** carefully.


### near — idx 3355

- **source**: Could you please explain that again?
- **target**: ~~Could you please explain~~ **Explain** that ~~again?~~ **again.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Could~~ **Tell** you ~~please explain~~ that ~~again?~~ **again.**
- `ef32` [exact ✗ · FRR ✓]: ~~Could you please explain~~ **What means,, to you, me is** that ~~again?~~ **that to me.**
- `steer` [exact ✗ · FRR ✓]: ~~Could~~ **Tell** you ~~please explain~~ that ~~again?~~ **again.**

## possessive_form

### success — idx 3428

- **source**: They studied the philosopher's writings.
- **target**: They studied the ~~philosopher's~~ **philosopher** writings.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They studied the ~~philosopher's~~ **philosopher** writings.
- `ef32` [exact ✓ · FRR ✓]: They studied the ~~philosopher's~~ **philosopher** writings.
- `steer` [exact ✗ · FRR ✓]: ~~They studied the philosopher's writings.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### success — idx 3437

- **source**: He admired the musician's skills.
- **target**: He admired the ~~musician's~~ **musician** skills.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He admired the ~~musician's~~ **musician** skills.
- `ef32` [exact ✓ · FRR ✓]: He admired the ~~musician's~~ **musician** skills.
- `steer` [exact ✗ · FRR ✓]: ~~He admired the musician's skills.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 3431

- **source**: We followed the director's instructions.
- **target**: We followed the ~~director's~~ **director** instructions.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We followed ~~the director's~~ instructions.
- `ef32` [exact ✗ · FRR ✓]: We followed ~~the director's~~ instructions.
- `steer` [exact ✗ · FRR ✓]: ~~We followed the director's instructions.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 3413

- **source**: He repaired the car's broken mirror.
- **target**: He repaired the ~~car's~~ **car** broken mirror.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He repaired the ~~car's broken~~ **car** mirror.
- `ef32` [exact ✗ · FRR ✓]: He repaired the ~~car's broken~~ **car** mirror.
- `steer` [exact ✗ · FRR ✓]: ~~He repaired the car's broken mirror.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## present_participle

### success — idx 3468

- **source**: She is wearing a red dress.
- **target**: She ~~is wearing~~ **wears** a red dress.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~is wearing~~ **wears** a red dress.
- `ef32` [exact ✓ · FRR ✓]: She ~~is wearing~~ **wears** a red dress.
- `steer` [exact ✗ · FRR ✓]: She ~~is wearing~~ **donned** a red dress.


### success — idx 3479

- **source**: The chef is tasting the soup.
- **target**: The chef ~~is tasting~~ **tastes** the soup.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The chef ~~is tasting~~ **tastes** the soup.
- `ef32` [exact ✓ · FRR ✓]: The chef ~~is tasting~~ **tastes** the soup.
- `steer` [exact ✓ · FRR ✓]: The chef ~~is tasting~~ **tastes** the soup.


### near — idx 3457

- **source**: The children are singing loudly.
- **target**: The children ~~are singing~~ **sing** loudly.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~children are singing~~ **sings** loudly.
- `ef32` [exact ✗ · FRR ✓]: The ~~children are singing~~ **sings** loudly.
- `steer` [exact ✗ · FRR ✓]: The children ~~are singing~~ **sang** loudly.


### near — idx 3483

- **source**: He is practicing the guitar.
- **target**: He ~~is practicing~~ **practices** the guitar.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~is practicing~~ **plays** the guitar.
- `ef32` [exact ✗ · FRR ✓]: He ~~is practicing~~ **plays** the guitar.
- `steer` [exact ✓ · FRR ✓]: He ~~is practicing~~ **practices** the guitar.

## present_perfect

### success — idx 3507

- **source**: This machine has printed documents quickly.
- **target**: This machine ~~has printed~~ **prints** documents quickly.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: This machine ~~has printed~~ **prints** documents quickly.
- `ef32` [exact ✓ · FRR ✓]: This machine ~~has printed~~ **prints** documents quickly.
- `steer` [exact ✓ · FRR ✓]: This machine ~~has printed~~ **prints** documents quickly.


### success — idx 3542

- **source**: Have you fed the stray cats?
- **target**: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?
- `ef32` [exact ✓ · FRR ✓]: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?
- `steer` [exact ✓ · FRR ✓]: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?


### near — idx 3504

- **source**: The dog has barked at strangers.
- **target**: The dog ~~has barked~~ **barks** at strangers.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The dog ~~has barked~~ at strangers.
- `ef32` [exact ✗ · FRR ✓]: The dog ~~has barked~~ at strangers.
- `steer` [exact ✓ · FRR ✓]: The dog ~~has barked~~ **barks** at strangers.


### near — idx 3510

- **source**: Birds have chirped loudly at dawn.
- **target**: Birds ~~have chirped~~ **chirp** loudly at dawn.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Birds ~~have chirped loudly~~ **chipped** at dawn.
- `ef32` [exact ✗ · FRR ✓]: Birds ~~have chirped loudly~~ **chipped** at dawn.
- `steer` [exact ✗ · FRR ✓]: Birds ~~have~~ chirped loudly at dawn.

## present_progressive

### success — idx 3579

- **source**: Composers are arranging scores.
- **target**: Composers ~~are arranging~~ **arrange** scores.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Composers ~~are arranging~~ **arrange** scores.
- `ef32` [exact ✓ · FRR ✓]: Composers ~~are arranging~~ **arrange** scores.
- `steer` [exact ✓ · FRR ✓]: Composers ~~are arranging~~ **arrange** scores.


### success — idx 3596

- **source**: Ink is bleeding through.
- **target**: Ink ~~is bleeding~~ **bleeds** through.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Ink ~~is bleeding~~ **bleeds** through.
- `ef32` [exact ✓ · FRR ✓]: Ink ~~is bleeding~~ **bleeds** through.
- `steer` [exact ✗ · FRR ✓]: Ink ~~is bleeding~~ **seeped** through.


### near — idx 3555

- **source**: I am proofreading documents.
- **target**: I ~~am proofreading~~ **proofread** documents.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I ~~am proofreading~~ **checked** documents.
- `ef32` [exact ✗ · FRR ✓]: I ~~am proofreading~~ **checked** documents.
- `steer` [exact ✓ · FRR ✓]: I ~~am proofreading~~ **proofread** documents.


### near — idx 3556

- **source**: Bees are pollinating flowers.
- **target**: Bees ~~are pollinating~~ **pollinate** flowers.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: Bees ~~are pollinating~~ **pollinated** flowers.
- `ef32` [exact ✗ · FRR ✓]: Bees ~~are pollinating flowers.~~ **pollinize.**
- `steer` [exact ✗ · FRR ✓]: Bees ~~are pollinating~~ **pollinated** flowers.

## punctual_durative

### success — idx 3604

- **source**: He shivers nonstop.
- **target**: He ~~shivers nonstop.~~ **shivers.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~shivers nonstop.~~ **shivers.**
- `ef32` [exact ✓ · FRR ✓]: He ~~shivers nonstop.~~ **shivers.**
- `steer` [exact ✗ · FRR ✓]: ~~He shivers nonstop.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### success — idx 3609

- **source**: Fire sparks intermittently.
- **target**: Fire ~~sparks intermittently.~~ **sparks.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Fire ~~sparks intermittently.~~ **sparks.**
- `ef32` [exact ✓ · FRR ✓]: Fire ~~sparks intermittently.~~ **sparks.**
- `steer` [exact ✗ · FRR ✓]: ~~Fire sparks intermittently.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 3637

- **source**: She sits.
- **target**: She ~~sits.~~ **sits down.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~She sits.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✓]: She ~~sits.~~ **has sits down.**
- `steer` [exact ✗ · FRR ✓]: ~~She sits.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### near — idx 3629

- **source**: We stay awake.
- **target**: We ~~stay awake.~~ **wake up.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~stay awake.~~ **wake up morning.**
- `ef32` [exact ✗ · FRR ✓]: We ~~stay awake.~~ **wake up morning.**
- `steer` [exact ✗ · FRR ✓]: We ~~stay~~ **are** awake.


### fail — idx 3617

- **source**: A car honks incessantly.
- **target**: A car ~~honks incessantly.~~ **honks.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~A car honks incessantly.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### fail — idx 3624

- **source**: A wire twangs constantly.
- **target**: A wire ~~twangs constantly.~~ **twangs.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~A wire twangs constantly.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## quantifier

### success — idx 3672

- **source**: They finished all the work by noon.
- **target**: They finished ~~all the~~ work by noon.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They finished ~~all the~~ work by noon.
- `ef32` [exact ✓ · FRR ✓]: They finished ~~all the~~ work by noon.
- `steer` [exact ✗ · FRR ✓]: ~~They finished all the work by noon.~~ ******


### success — idx 3677

- **source**: We planted some vegetables in the garden.
- **target**: We planted ~~some~~ vegetables in the garden.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: We planted ~~some~~ vegetables in the garden.
- `ef32` [exact ✓ · FRR ✓]: We planted ~~some~~ vegetables in the garden.
- `steer` [exact ✗ · FRR ✓]: ~~We planted some vegetables in the garden.~~ **rowspan rowspan rowspan**


### near — idx 3687

- **source**: They need each document for the application.
- **target**: They need ~~each document~~ **documents** for the application.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They need ~~each~~ document for the application.
- `ef32` [exact ✗ · FRR ✓]: They need ~~each~~ document for the application.
- `steer` [exact ✓ · FRR ✓]: They need ~~each document~~ **documents** for the application.


### near — idx 3696

- **source**: There is any evidence to support that.
- **target**: ~~There is any evidence to support~~ **Evidence supports** that.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~There is any evidence to support that.~~ **Evidence**
- `ef32` [exact ✗ · FRR ✓]: ~~There is any evidence to support that.~~ **Evidence**
- `steer` [exact ✓ · FRR ✓]: ~~There is any evidence to support~~ **Evidence supports** that.


### fail — idx 3654

- **source**: We saw many birds flying south.
- **target**: We saw ~~many~~ birds flying south.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~We saw many birds flying south.~~ **rowspan rowspan**


### fail — idx 3685

- **source**: Did you take many photos on vacation?
- **target**: Did you take ~~many~~ photos on vacation?

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~Did you take many photos on vacation?~~ **rowspan rowspan**

## quantitative_prefix

### near — idx 3707

- **source**: The solution was semitransparent.
- **target**: The solution was ~~semitransparent.~~ **partially transparent.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The solution was ~~semitransparent.~~ **partially, in, in transparent,, partially,,, open,, in and partially,, transparent, in in partially transparent transparent,, partially with a**
- `ef32` [exact ✗ · FRR ✓]: The solution was ~~semitransparent.~~ **partially, in, in transparent,, partially,,, open,, in and partially,, transparent, in in partially transparent transparent,, partially with a**
- `steer` [exact ✗ · FRR ✓]: The solution was ~~semitransparent.~~ **semi see through.**


### near — idx 3710

- **source**: The quadrennial event was postponed.
- **target**: The ~~quadrennial~~ event **held every four years** was postponed.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The ~~quadrennial~~ **annual** event was postponed.
- `ef32` [exact ✗ · FRR ✗]: The ~~quadrennial event was postponed.~~ **every event,, held every in every year in in in year,,,, every day year the year every year year in in every year year held.**
- `steer` [exact ✗ · FRR ✗]: The ~~quadrennial~~ **annual** event was postponed.


### fail — idx 3740

- **source**: The factory produces kilowatt machines.
- **target**: The factory produces ~~kilowatt machines.~~ **machines rated at one thousand watts.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The factory produces ~~kilowatt~~ **kilowatt-horsepower** machines.
- `ef32` [exact ✗ · FRR ✗]: The factory produces **at one one one million hundred thousand million one million million a year at hundred million thousand thousand** kilowatt ~~machines.~~ **per of a a one million hour machines per at.**
- `steer` [exact ✗ · FRR ✗]: The factory produces ~~kilowatt~~ **kilowatt-horsepower** machines.


### fail — idx 3742

- **source**: The training program is unisex.
- **target**: The training program is ~~unisex.~~ **designed for one gender or both without distinction.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The training program is ~~unisex.~~ **unisex both in for both a men or or one or one a of or for a a a one one of or one for a or gender of of a or.**
- `ef32` [exact ✗ · FRR ✗]: The training program is ~~unisex.~~ **unisex both in for both a men or or one or one a of or for a a a one one of or one for a or gender of of a or.**
- `steer` [exact ✗ · FRR ✗]: ~~The training program is unisex.~~ **of" genders.**

## referring

### success — idx 3752

- **source**: This music brings joy.
- **target**: ~~This music~~ **Music** brings joy.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~This music~~ **Music** brings joy.
- `ef32` [exact ✓ · FRR ✓]: ~~This music~~ **Music** brings joy.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### success — idx 3762

- **source**: These poems touch hearts.
- **target**: ~~These poems~~ **Poems** touch hearts.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~These poems~~ **Poems** touch hearts.
- `ef32` [exact ✓ · FRR ✓]: ~~These poems~~ **Poems** touch hearts.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 3777

- **source**: Their babies cry often.
- **target**: ~~Their babies~~ **Babies** cry often.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Their babies~~ **baby** cry often.
- `ef32` [exact ✗ · FRR ✓]: ~~Their babies~~ **baby** cry often.
- `steer` [exact ✗ · FRR ✓]: ~~Their babies cry often.~~ **Babies of the other often cry.**


### near — idx 3778

- **source**: Those mountains challenge climbers.
- **target**: ~~Those mountains~~ **Mountains** challenge climbers.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Those mountains challenge climbers.~~ **mountains.**
- `ef32` [exact ✗ · FRR ✓]: ~~Those mountains challenge climbers.~~ **mountains.**
- `steer` [exact ✓ · FRR ✓]: ~~Those mountains~~ **Mountains** challenge climbers.

## relative_clauses

### success — idx 3848

- **source**: The teacher who inspired me retired last year.
- **target**: ~~The~~ **My inspiring** teacher ~~who inspired me~~ retired last year.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~The~~ **My inspiring** teacher ~~who inspired me~~ retired last year.
- `ef32` [exact ✗ · FRR ✓]: ~~The teacher who inspired me retired~~ **My own family life life, beautiful and beautiful beautiful beautiful beautiful beautiful inspiring beautiful family, my own family own family family teacher, my family own family family** last year.
- `steer` [exact ✓ · FRR ✓]: ~~The~~ **My inspiring** teacher ~~who inspired me~~ retired last year.


### near — idx 3821

- **source**: The child whose toy broke started crying.
- **target**: The ~~child whose~~ **child’s** toy broke **and the child** started crying.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The ~~child whose~~ **child, the** toy broke **and the child** started crying.
- `ef32` [exact ✗ · FRR ✓]: The ~~child whose~~ **child'ss'sss thes child'sss** toy **ands''ss and the** broke ~~started~~ **the'ss** crying.
- `steer` [exact ✗ · FRR ✓]: The ~~child whose~~ **child, the** toy broke **and the child** started crying.


### near — idx 3838

- **source**: The student whose experiment failed was disappointed.
- **target**: ~~The student whose~~ **His** experiment failed **and the student** was disappointed.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The student ~~whose~~ **was disappointed because his** experiment ~~failed was disappointed.~~ **failed.**
- `ef32` [exact ✗ · FRR ✓]: ~~The student whose experiment failed was~~ **His and his his his his other experiment, he and and, and his he and his very very his other other very other very very very very his other other failure** disappointed.
- `steer` [exact ✗ · FRR ✓]: The student ~~whose~~ **was disappointed because his** experiment ~~failed was disappointed.~~ **failed.**


### fail — idx 3832

- **source**: The garden which Sarah tends is beautiful.
- **target**: ~~The garden which~~ Sarah tends **the garden and it** is beautiful.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The garden ~~which~~ **that** Sarah tends is beautiful.
- `ef32` [exact ✗ · FRR ✓]: ~~The garden which Sarah tends~~ is ~~beautiful.~~ **a very very a very good a a good a very very a good very a a good and a a beautiful beautiful and a a a and a and a**
- `steer` [exact ✗ · FRR ✗]: The garden ~~which~~ **that** Sarah tends is beautiful.

## representative

### near — idx 3873

- **source**: Geologists assert earthquakes cause tectonic shifts.
- **target**: ~~Geologists assert earthquakes~~ **Earthquakes** cause tectonic shifts.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Geologists~~ **Earthologists** assert earthquakes cause tectonic shifts.
- `ef32` [exact ✗ · FRR ✓]: ~~Geologists~~ **Earthologists** assert earthquakes cause tectonic shifts.
- `steer` [exact ✗ · FRR ✗]: ~~Geologists assert earthquakes~~ **Earthquakes** cause tectonic ~~shifts.~~ **shifts, according to geologists.**


### near — idx 3857

- **source**: It is evident she practices piano daily.
- **target**: ~~It is evident she~~ **She** practices piano daily.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~It is evident she practices piano daily.~~ **He**
- `ef32` [exact ✗ · FRR ✓]: ~~It is evident she practices piano daily.~~ **He**
- `steer` [exact ✗ · FRR ✓]: ~~It is evident she~~ **She** practices piano ~~daily.~~ **daily?**


### fail — idx 3891

- **source**: Science asserts mercury is a liquid metal.
- **target**: ~~Science asserts mercury~~ **Mercury** is a liquid metal.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: Science asserts **that** mercury is a liquid metal.
- `ef32` [exact ✗ · FRR ✓]: ~~Science asserts mercury~~ **Mercury** is a ~~liquid~~ metal.
- `steer` [exact ✗ · FRR ✗]: Science asserts **that** mercury is a liquid metal.

## resultative

### near — idx 3935

- **source**: He cut the rope free.
- **target**: He cut the **rope. The** rope **was** free.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~He cut the rope free.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✓]: He **has** cut the **very** rope free.
- `steer` [exact ✗ · FRR ✓]: ~~He cut the rope free.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### near — idx 3941

- **source**: She rinsed the soap off.
- **target**: She rinsed the **soap. The** soap **was** off.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She rinsed the soap **off** off.
- `ef32` [exact ✗ · FRR ✓]: She rinsed the soap **off** off.
- `steer` [exact ✗ · FRR ✓]: ~~She rinsed the soap off.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 3916

- **source**: She tied the knot tight.
- **target**: She tied the **knot. The** knot **was** tight.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~She tied the knot tight.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 3919

- **source**: He folded the paper neat.
- **target**: He folded the **paper. The** paper **was** neat.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~He folded the paper neat.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## s_genitive

### success — idx 4407

- **source**: Laura’s presentation impressed everyone.
- **target**: ~~Laura’s~~ **Her** presentation impressed everyone.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Laura’s~~ **Her** presentation impressed everyone.
- `ef32` [exact ✓ · FRR ✓]: ~~Laura’s~~ **Her** presentation impressed everyone.
- `steer` [exact ✗ · FRR ✓]: ~~Laura’s~~ **Her** presentation impressed everyone. **.**


### success — idx 4423

- **source**: The cat’s owner arrived.
- **target**: The ~~cat’s~~ owner **of the cat** arrived.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~The cat’s~~ **the** owner **of the cat** arrived.
- `ef32` [exact ✗ · FRR ✓]: The ~~cat’s~~ **of the the the the the the the the the the the the the of the the the the the the the the the** owner ~~arrived.~~ **of of the the the the the the the the the.**
- `steer` [exact ✓ · FRR ✓]: ~~The cat’s~~ **the** owner **of the cat** arrived.


### near — idx 4416

- **source**: Harry’s birthday is tomorrow.
- **target**: ~~Harry’s~~ **His** birthday is tomorrow.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Harry’s~~ **His** birthday is tomorrow. **. . . . .**
- `ef32` [exact ✗ · FRR ✓]: ~~Harry’s birthday~~ **His own father's** is tomorrow.
- `steer` [exact ✗ · FRR ✓]: ~~Harry’s~~ **His** birthday is tomorrow. **. . . . .**

## spatial

### success — idx 3974

- **source**: The phone rang while I was cooking dinner. I didn’t answer it because my hands were full.
- **target**: The phone rang while I was cooking dinner. I didn’t answer ~~it~~ **the phone** because my hands were full.

- `ours` via ef32 [exact ✓ · FRR —] ←focus: The phone rang while I was cooking dinner. I didn’t answer ~~it~~ **the phone** because my hands were full.
- `ef32` [exact ✓ · FRR —]: The phone rang while I was cooking dinner. I didn’t answer ~~it~~ **the phone** because my hands were full.
- `steer` [exact ✗ · FRR —]: ~~The phone~~ **Phone** rang while I was cooking dinner. I ~~didn’t~~ **didn't** answer ~~it~~ **the phone** because my hands were full.


### near — idx 3979

- **source**: She went to the library to study because the library was quiet. She could concentrate better.
- **target**: She went to the library to study because ~~the library~~ **it** was quiet. She could concentrate better.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She went to the library to study because the ~~library~~ **it** was quiet. She could concentrate better.
- `ef32` [exact ✗ · FRR ✓]: She went to the library to study because the ~~library~~ **it** was quiet. She could concentrate better.
- `steer` [exact ✗ · FRR ✗]: She went to the library to ~~study because the library was quiet. She could concentrate better.~~ **focus on her work.**

## spatial_or_directional_prefix

### success — idx 4041

- **source**: The event was an extra-ordinary success.
- **target**: The event was an ~~extra-ordinary~~ **exceptional** success.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The event was an ~~extra-ordinary~~ **exceptional** success.
- `ef32` [exact ✓ · FRR ✓]: The event was an ~~extra-ordinary~~ **exceptional** success.
- `steer` [exact ✗ · FRR ✗]: The event was an ~~extra-ordinary~~ **extraordinary** success.


### near — idx 4029

- **source**: She studied intercellular communication.
- **target**: She studied ~~intercellular communication.~~ **communication between cells.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: She studied ~~intercellular communication.~~ **the communication between cells.**
- `ef32` [exact ✗ · FRR ✓]: She studied ~~intercellular communication.~~ **the between the the cells communication between the the and the.**
- `steer` [exact ✗ · FRR ✓]: She studied ~~intercellular communication.~~ **the communication between cells.**


### near — idx 4023

- **source**: They mapped the subcontinental terrain.
- **target**: They mapped the ~~subcontinental terrain.~~ **terrain beneath the continent.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: They ~~mapped~~ **charted** the ~~subcontinental terrain.~~ **land beneath.**
- `ef32` [exact ✗ · FRR ✓]: They mapped the ~~subcontinental terrain.~~ **the terrain the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the.**
- `steer` [exact ✗ · FRR ✓]: They ~~mapped~~ **charted** the ~~subcontinental terrain.~~ **land beneath.**


### fail — idx 4040

- **source**: The scientist examined intrauterine growth.
- **target**: The scientist examined ~~intrauterine growth.~~ **growth inside the uterus.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The scientist examined intrauterine ~~growth.~~ **growth the the the the the the the the the the the the the the the the the the the the the the the the the the the the the.**
- `ef32` [exact ✗ · FRR ✗]: The scientist examined intrauterine ~~growth.~~ **growth the the the the the the the the the the the the the the the the the the the the the the the the the the the the the.**
- `steer` [exact ✗ · FRR ✓]: The scientist ~~examined intrauterine~~ **studied fetal** growth.


### fail — idx 4035

- **source**: They studied the transnational network systems.
- **target**: They studied the ~~transnational~~ network ~~systems.~~ **systems across nations.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: They studied the transnational network ~~systems.~~ **systems of across the the the the countries of the countries of the the the across the the countries the the country of the the across the the**
- `ef32` [exact ✗ · FRR ✗]: They studied the transnational network ~~systems.~~ **systems of across the the the the countries of the countries of the the the across the the countries the the country of the the across the the**
- `steer` [exact ✗ · FRR ✓]: They studied the ~~transnational~~ **global** network systems.

## split_infinitives

### success — idx 4086

- **source**: He had to suddenly adjust the settings to fix the machine.
- **target**: He had to ~~suddenly~~ adjust the settings **suddenly** to fix the machine.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: He had to ~~suddenly~~ adjust the settings **suddenly** to fix the machine.
- `ef32` [exact ✗ · FRR ✓]: He had to ~~suddenly~~ adjust the settings **automatically** to fix the machine.
- `steer` [exact ✓ · FRR ✓]: He had to ~~suddenly~~ adjust the settings **suddenly** to fix the machine.


### success — idx 4087

- **source**: The team hopes to easily win the championship with hard work.
- **target**: The team hopes to ~~easily~~ win the championship **easily** with hard work.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: The team hopes to ~~easily~~ win the championship **easily** with hard work.
- `ef32` [exact ✗ · FRR ✓]: The team hopes to ~~easily~~ win **in** the championship **in** with hard work.
- `steer` [exact ✓ · FRR ✓]: The team hopes to ~~easily~~ win the championship **easily** with hard work.


### near — idx 4079

- **source**: They plan to more effectively communicate with their team members.
- **target**: They plan to ~~more effectively~~ communicate with their team ~~members.~~ **members more effectively.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: They plan to **communicate** more effectively ~~communicate~~ with their team members.
- `ef32` [exact ✗ · FRR ✓]: They plan to ~~more effectively~~ communicate with their team ~~members.~~ **members regularly and and and with.**
- `steer` [exact ✗ · FRR ✓]: They plan to **communicate** more effectively ~~communicate~~ with their team members.


### near — idx 4092

- **source**: I want to clearly express my ideas in the presentation.
- **target**: I want to ~~clearly~~ express my ideas **clearly** in the presentation.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: I ~~want to clearly express~~ **will articulate** my ideas **effectively** in the presentation.
- `ef32` [exact ✗ · FRR ✓]: I want to ~~clearly express~~ **expressing** my ideas in **in in in in in in in in in in in in verbally in and in in and in and in and in and verbally** the presentation.
- `steer` [exact ✗ · FRR ✓]: I ~~want to clearly express~~ **will articulate** my ideas **effectively** in the presentation.


### fail — idx 4070

- **source**: He hopes to easily pass the exam with regular study.
- **target**: He hopes to ~~easily~~ pass the exam **easily** with regular study.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: He hopes to easily pass the exam ~~with~~ **through** regular study.
- `ef32` [exact ✗ · FRR ✓]: He hopes to ~~easily~~ pass the exam **only** with regular study.
- `steer` [exact ✗ · FRR ✗]: He hopes to easily pass the exam ~~with~~ **through** regular study.

## static_dynamic

### success — idx 4125

- **source**: A yogi maintains postures.
- **target**: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.
- `ef32` [exact ✓ · FRR ✓]: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.
- `steer` [exact ✓ · FRR ✓]: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.


### success — idx 4118

- **source**: An actor learns lines by heart.
- **target**: ~~An~~ **The** actor ~~learns lines by heart.~~ **memorizes lines.**

- `ours` via steer [exact ✓ · FRR —] ←focus: ~~An~~ **The** actor ~~learns lines by heart.~~ **memorizes lines.**
- `ef32` [exact ✗ · FRR —]: ~~An~~ **The The** actor ~~learns~~ **Mor Mor Mor Mor Mor Mor Mor** lines ~~by heart.~~ **the the the the the actor Mor of the the Mor the Mor Mor the Mor Mor the Mor actor Mor Mor.**
- `steer` [exact ✓ · FRR —]: ~~An~~ **The** actor ~~learns lines by heart.~~ **memorizes lines.**


### near — idx 4145

- **source**: Ice becomes solid.
- **target**: ~~Ice becomes~~ **The ice remains** solid.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Ice becomes~~ **The 22 22'2''s 2222's'ss remains''s's** solid.
- `ef32` [exact ✗ · FRR ✓]: ~~Ice becomes~~ **The 22 22'2''s 2222's'ss remains''s's** solid.
- `steer` [exact ✗ · FRR ✗]: Ice ~~becomes~~ **turns** solid.


### near — idx 4104

- **source**: He trembles during thunderstorms.
- **target**: He ~~trembles during~~ **fears** thunderstorms.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~trembles during thunderstorms.~~ **fears that his he would would would would would he fear the risk fear risk he would would not would would would be afraid will be afraid afraid.**
- `ef32` [exact ✗ · FRR ✓]: He ~~trembles during thunderstorms.~~ **fears that his he would would would would would he fear the risk fear risk he would would not would would would be afraid will be afraid afraid.**
- `steer` [exact ✓ · FRR ✓]: He ~~trembles during~~ **fears** thunderstorms.


### fail — idx 4124

- **source**: An acrobat maintains equilibrium midair.
- **target**: ~~An~~ **The** acrobat ~~maintains equilibrium midair.~~ **balances on beams.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~An acrobat maintains equilibrium midair.~~ **The acrobats balance on the air.**
- `ef32` [exact ✗ · FRR ✗]: ~~An~~ **The** acrobat ~~maintains equilibrium midair.~~ **on the the on the on the on on the the the on the on the the on mida the the the on the the on the the on the.**
- `steer` [exact ✗ · FRR ✗]: ~~An acrobat maintains equilibrium midair.~~ **The acrobats balance on the air.**


### fail — idx 4110

- **source**: A bridge crosses over the river.
- **target**: A bridge ~~crosses over~~ **spans** the river.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: A bridge ~~crosses over~~ **spans across** the river.
- `ef32` [exact ✗ · FRR ✗]: A bridge ~~crosses over~~ **spans across** the river.
- `steer` [exact ✗ · FRR ✗]: A ~~bridge crosses over~~ **structure spans** the river.

## subject_auxiliary_inversion

### success — idx 4171

- **source**: Can you believe what he just said?
- **target**: ~~Can you~~ **You can** believe what he just ~~said?~~ **said.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Can you~~ **You can** believe what he just ~~said?~~ **said.**
- `ef32` [exact ✗ · FRR ✓]: ~~Can~~ **You' can't''t'tss's believe that** you ~~believe what he just said?~~ **is said.**
- `steer` [exact ✓ · FRR ✓]: ~~Can you~~ **You can** believe what he just ~~said?~~ **said.**


### success — idx 4197

- **source**: So bright was the moon that we could walk without lights.
- **target**: ~~So~~ **The moon was so** bright ~~was the moon~~ that we could walk without lights.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~So~~ **The moon was so** bright ~~was the moon~~ that we could walk without lights.
- `ef32` [exact ✗ · FRR ✓]: ~~So bright was the~~ **The is also also also also also also is called is Moon is also also called also** moon ~~that~~ we ~~could~~ **are also also not are not not also not are really not also** walk **is** without lights.
- `steer` [exact ✓ · FRR ✓]: ~~So~~ **The moon was so** bright ~~was the moon~~ that we could walk without lights.


### near — idx 4193

- **source**: May you stay healthy and happy!
- **target**: ~~May~~ **I hope** you stay healthy and ~~happy!~~ **happy.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~May you stay~~ **I'''m'mm'mmm' Immm' I Immm' Immmm** healthy and happy!
- `ef32` [exact ✗ · FRR ✓]: ~~May you stay~~ **I'''m'mm'mmm' Immm' I Immm' Immmm** healthy and happy!
- `steer` [exact ✓ · FRR ✓]: ~~May~~ **I hope** you stay healthy and ~~happy!~~ **happy.**


### near — idx 4163

- **source**: Seldom has he spoken about his past.
- **target**: ~~Seldom~~ **He** has ~~he~~ **seldom** spoken about his past.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Seldom~~ **He** has ~~he~~ **rarely** spoken about his past.
- `ef32` [exact ✗ · FRR ✓]: ~~Seldom has~~ **Similarly,** he ~~spoken~~ **had, similarly similarly, similarly similar, similarly similarly similar, similarly similar similarly similar, similarly similar similarly to, similarly similarly similarly to** about his past.
- `steer` [exact ✗ · FRR ✓]: ~~Seldom~~ **He** has ~~he~~ **rarely** spoken about his past.

## subject_verb_inversion

### success — idx 4202

- **source**: There the car goes.
- **target**: ~~There the~~ **The** car ~~goes.~~ **goes there.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~There the~~ **The** car ~~goes.~~ **goes there.**
- `ef32` [exact ✗ · FRR ✓]: ~~There the car goes.~~ **They went to and to to them a and to them, parking. and to to parking to, to them and and to to from them.**
- `steer` [exact ✓ · FRR ✓]: ~~There the~~ **The** car ~~goes.~~ **goes there.**


### success — idx 4210

- **source**: Loudly laughed the children in the park.
- **target**: ~~Loudly~~ **The children** laughed ~~the children~~ **loudly** in the park.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Loudly~~ **The children** laughed ~~the children~~ **loudly** in the park.
- `ef32` [exact ✗ · FRR ✓]: ~~Loudly~~ **The children'sThe's's children's'ss's's's** laughed ~~the children~~ in ~~the~~ park.
- `steer` [exact ✓ · FRR ✓]: ~~Loudly~~ **The children** laughed ~~the children~~ **loudly** in the park.


### near — idx 4212

- **source**: There goes his comment.
- **target**: ~~There~~ **His comment** goes ~~his comment.~~ **there.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~There goes~~ **is is also also also his is also also also here, known as as as the as the here,,, is, and** his comment.
- `ef32` [exact ✗ · FRR ✓]: ~~There goes~~ **is is also also also his is also also also here, known as as as the as the here,,, is, and** his comment.
- `steer` [exact ✗ · FRR ✓]: ~~There goes~~ **That was** his comment.


### near — idx 4244

- **source**: Only by trusting can love truly grow.
- **target**: ~~Only~~ **Love can truly grow only** by ~~trusting can love truly grow.~~ **trusting.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: Only by ~~trusting~~ **trusting, truly love** can ~~love truly~~ grow.
- `ef32` [exact ✗ · FRR ✓]: ~~Only~~ **only** by ~~trusting can love~~ **a person person,, love,,,,,,, the only way love,,, for a, a person, found in the heart** truly grow.
- `steer` [exact ✗ · FRR ✓]: Only by ~~trusting~~ **trusting, truly love** can ~~love truly~~ grow.


### fail — idx 4233

- **source**: Here comes the mailman with today’s letters.
- **target**: ~~Here~~ **The mailman** comes ~~the mailman~~ **here** with today’s letters.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: Here comes the mailman with ~~today’s~~ **today's** letters.
- `ef32` [exact ✗ · FRR ✓]: ~~Here comes the mailman~~ **The house, and** with **him which they was was, as which they was was here, and came, as as was came here to me with him** today’s letters.
- `steer` [exact ✗ · FRR ✗]: Here comes the mailman with ~~today’s~~ **today's** letters.


### fail — idx 4208

- **source**: There goes the alarm clock we set earlier.
- **target**: ~~There goes the~~ **The** alarm clock we set ~~earlier.~~ **earlier goes there.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~There goes the~~ **The** alarm ~~clock we set earlier.~~ **was, and then then was then was was then stopped by, was then and then stopped immediately, and then then again stopped to and then was turned to off to on set.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## subjunctive_mood

### success — idx 4287

- **source**: I wish I could paint pictures.
- **target**: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.
- `ef32` [exact ✓ · FRR ✓]: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.
- `steer` [exact ✓ · FRR ✓]: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.


### success — idx 4272

- **source**: If the temperature dropped, we would need warm clothes.
- **target**: If the temperature ~~dropped,~~ **drops,** we ~~would~~ **will** need warm clothes.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: If the temperature ~~dropped,~~ **drops,** we ~~would~~ **will** need warm clothes.
- `ef32` [exact ✓ · FRR ✓]: If the temperature ~~dropped,~~ **drops,** we ~~would~~ **will** need warm clothes.
- `steer` [exact ✓ · FRR ✓]: If the temperature ~~dropped,~~ **drops,** we ~~would~~ **will** need warm clothes.


### near — idx 4283

- **source**: I wish I had more money.
- **target**: I ~~wish~~ **hope** I ~~had~~ **can get** more money.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: I ~~wish I had~~ **hope to get** more money.
- `ef32` [exact ✗ · FRR ✓]: I ~~wish~~ **hope get** I ~~had~~ **can get get** more money.
- `steer` [exact ✗ · FRR ✓]: I ~~wish I had~~ **hope to get** more money.


### near — idx 4251

- **source**: I suggest that he be on time.
- **target**: I suggest ~~that~~ he **should** be on time.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I ~~suggest that~~ **should be** he be on time.
- `ef32` [exact ✗ · FRR ✓]: I ~~suggest that~~ **should be** he be on time.
- `steer` [exact ✗ · FRR ✓]: ~~I suggest that~~ **Suggest** he be on time.

## superlative

### success — idx 4324

- **source**: He showed the happiest smile.
- **target**: He showed the ~~happiest~~ **happy** smile.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He showed the ~~happiest~~ **happy** smile.
- `ef32` [exact ✓ · FRR ✓]: He showed the ~~happiest~~ **happy** smile.
- `steer` [exact ✗ · FRR ✓]: He ~~showed the happiest smile.~~ **beamed with happiness.**


### success — idx 4337

- **source**: He took the safest route.
- **target**: He took the ~~safest~~ **safe** route.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He took the ~~safest~~ **safe** route.
- `ef32` [exact ✓ · FRR ✓]: He took the ~~safest~~ **safe** route.
- `steer` [exact ✗ · FRR ✓]: He took the ~~safest~~ **secure** route.


### near — idx 4323

- **source**: That was the silliest mistake ever.
- **target**: That was the ~~silliest~~ **silly** mistake ever.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: That was the ~~silliest~~ mistake ever.
- `ef32` [exact ✗ · FRR ✓]: That was the ~~silliest~~ mistake ever.
- `steer` [exact ✓ · FRR ✓]: That was the ~~silliest~~ **silly** mistake ever.


### fail — idx 4331

- **source**: He had the most memorable vacation.
- **target**: He had the ~~most~~ memorable vacation.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 4316

- **source**: He is the most respectful student here.
- **target**: He is the ~~most~~ respectful student here.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~He is the most respectful student here.~~ **rowspan rowspan rowspan**

## synecdoche

### success — idx 4400

- **source**: Tailors measured shoulders for custom suits.
- **target**: Tailors measured ~~shoulders~~ **customers** for custom suits.

- `ours` via ef32 [exact ✓ · FRR ✗] ←focus: Tailors measured ~~shoulders~~ **customers** for custom suits.
- `ef32` [exact ✓ · FRR ✗]: Tailors measured ~~shoulders~~ **customers** for custom suits.
- `steer` [exact ✗ · FRR ✗]: Tailors measured **customers'** shoulders for custom suits.


### success — idx 4353

- **source**: The orchestra tuned their strings before the concert.
- **target**: The orchestra tuned their ~~strings~~ **instruments** before the concert.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The orchestra tuned their ~~strings~~ **instruments** before the concert.
- `ef32` [exact ✓ · FRR ✓]: The orchestra tuned their ~~strings~~ **instruments** before the concert.
- `steer` [exact ✓ · FRR ✓]: The orchestra tuned their ~~strings~~ **instruments** before the concert.


### near — idx 4362

- **source**: The museum acquired a Renaissance brushstroke.
- **target**: The museum acquired a Renaissance ~~brushstroke.~~ **painting.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The museum acquired a ~~Renaissance brushstroke.~~ **collection painting of the by painting the Museum of painting in of the the of painting of the**
- `ef32` [exact ✗ · FRR ✓]: The museum acquired a ~~Renaissance brushstroke.~~ **collection painting of the by painting the Museum of painting in of the the of painting of the**
- `steer` [exact ✓ · FRR ✓]: The museum acquired a Renaissance ~~brushstroke.~~ **painting.**


### near — idx 4371

- **source**: Pens interviewed protesters at the rally.
- **target**: ~~Pens~~ **Journalists** interviewed protesters at the rally.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Pens~~ **The journalist** interviewed **the** protesters at the rally.
- `ef32` [exact ✗ · FRR ✓]: ~~Pens interviewed protesters~~ **' interviewed'ss'sss'sss Journal of of' Journalsss'ss Journal of of Journal of protesters'** at the rally.
- `steer` [exact ✗ · FRR ✓]: ~~Pens~~ **The journalist** interviewed **the** protesters at the rally.


### fail — idx 4378

- **source**: Divers explored colors near the atoll.
- **target**: Divers explored ~~colors~~ **coral reefs** near the atoll.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: Divers explored ~~colors~~ **reefs** near the atoll.
- `ef32` [exact ✗ · FRR ✗]: Divers explored ~~colors~~ **reefs** near the atoll.
- `steer` [exact ✓ · FRR ✗]: Divers explored ~~colors~~ **coral reefs** near the atoll.


### fail — idx 4382

- **source**: Pilots checked the skies before takeoff.
- **target**: Pilots checked ~~the skies~~ **weather reports** before takeoff.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~Pilots checked the skies~~ **Pilots,, weather,,, checked,, weather and, and weather weather, weather weather and weather weather** before ~~takeoff.~~ **and and and and and takeoff,,.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## tag_questions

### success — idx 4482

- **source**: You were sleeping, weren’t you?
- **target**: ~~You were sleeping, weren’t you?~~ **Were you sleeping?**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~You were sleeping, weren’t you?~~ **Were you sleeping?**
- `ef32` [exact ✗ · FRR ✓]: ~~You~~ **Did your house have own house or** were ~~sleeping, weren’t you?~~ **a own house?**
- `steer` [exact ✓ · FRR ✓]: ~~You were sleeping, weren’t you?~~ **Were you sleeping?**


### success — idx 4479

- **source**: You have been here before, haven’t you?
- **target**: ~~You have~~ **Have you** been here ~~before, haven’t you?~~ **before?**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~You have~~ **Have you** been here ~~before, haven’t you?~~ **before?**
- `ef32` [exact ✗ · FRR ✓]: ~~You have been here before, haven’t you?~~ **Do you have?**
- `steer` [exact ✓ · FRR ✓]: ~~You have~~ **Have you** been here ~~before, haven’t you?~~ **before?**


### near — idx 4459

- **source**: You’re coming too, aren’t you?
- **target**: You’re coming ~~too, aren’t you?~~ **too.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: You’re coming ~~too, aren’t you?~~ **too. going with me,,,. each of them other places coming to me with with.**
- `ef32` [exact ✗ · FRR ✓]: You’re coming ~~too, aren’t you?~~ **too. going with me,,,. each of them other places coming to me with with.**
- `steer` [exact ✗ · FRR ✓]: ~~You’re~~ **You are** coming ~~too, aren’t you?~~ **too,.**


### near — idx 4466

- **source**: Julia is singing, isn’t she?
- **target**: **Is** Julia ~~is singing, isn’t she?~~ **singing?**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Julia is singing, isn’t she?~~ **Is Java's name India's?**
- `ef32` [exact ✗ · FRR ✓]: ~~Julia is singing, isn’t she?~~ **Is Java's name India's?**
- `steer` [exact ✓ · FRR ✓]: **Is** Julia ~~is singing, isn’t she?~~ **singing?**

## telic_atelic

### success — idx 4510

- **source**: Students memorized a formula.
- **target**: Students memorized ~~a formula.~~ **formulas.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Students memorized ~~a formula.~~ **formulas.**
- `ef32` [exact ✓ · FRR ✓]: Students memorized ~~a formula.~~ **formulas.**
- `steer` [exact ✗ · FRR ✗]: Students memorized ~~a~~ **the** formula.


### near — idx 4536

- **source**: Leaves were rustling.
- **target**: Leaves ~~were rustling.~~ **rustled.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Leaves ~~were rustling.~~ **iron.**
- `ef32` [exact ✗ · FRR ✓]: Leaves ~~were rustling.~~ **iron.**
- `steer` [exact ✓ · FRR ✓]: Leaves ~~were rustling.~~ **rustled.**


### near — idx 4506

- **source**: Children solved a puzzle.
- **target**: Children solved ~~a puzzle.~~ **puzzles.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Children ~~solved a puzzle.~~
- `ef32` [exact ✗ · FRR ✓]: Children ~~solved a puzzle.~~
- `steer` [exact ✗ · FRR ✗]: Children solved ~~a~~ **the** puzzle.


### fail — idx 4533

- **source**: Ice was cracking.
- **target**: Ice ~~was cracking.~~ **cracked.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Ice was cracking.~~ **He cracked.**
- `ef32` [exact ✗ · FRR ✗]: ~~Ice was cracking.~~ **He cracked.**
- `steer` [exact ✗ · FRR ✗]: Ice ~~was cracking.~~ **shattered.**


### fail — idx 4516

- **source**: The team practiced a drill.
- **target**: The team practiced ~~a drill.~~ **drills.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The team ~~practiced a drill.~~ **drills.**
- `ef32` [exact ✗ · FRR ✗]: The team ~~practiced a drill.~~ **drills.**
- `steer` [exact ✗ · FRR ✓]: The team practiced ~~a drill.~~ **their drills.**

## temporal

### success — idx 4595

- **source**: I watched the game that night.
- **target**: I watched the game ~~that~~ **last** night.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I watched the game ~~that~~ **last** night.
- `ef32` [exact ✓ · FRR ✓]: I watched the game ~~that~~ **last** night.
- `steer` [exact ✓ · FRR ✓]: I watched the game ~~that~~ **last** night.


### success — idx 4588

- **source**: They are celebrating their anniversary that week.
- **target**: They are celebrating their anniversary ~~that~~ **next** week.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They are celebrating their anniversary ~~that~~ **next** week.
- `ef32` [exact ✓ · FRR ✓]: They are celebrating their anniversary ~~that~~ **next** week.
- `steer` [exact ✓ · FRR ✓]: They are celebrating their anniversary ~~that~~ **next** week.


### near — idx 4559

- **source**: They will arrive that day.
- **target**: They will arrive ~~that day.~~ **on Monday.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They will arrive ~~that day.~~ **on Monday Monday on March.**
- `ef32` [exact ✗ · FRR ✓]: They will arrive ~~that day.~~ **on Monday Monday on March.**
- `steer` [exact ✓ · FRR ✓]: They will arrive ~~that day.~~ **on Monday.**


### near — idx 4586

- **source**: We will have a meeting that day.
- **target**: We will have a meeting ~~that day.~~ **next Wednesday.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We will have a meeting ~~that day.~~ **Wednesday next.**
- `ef32` [exact ✗ · FRR ✓]: We will have a meeting ~~that day.~~ **Wednesday next.**
- `steer` [exact ✗ · FRR ✓]: ~~We~~ **Next Wednesday, we** will have a ~~meeting that day.~~ **meeting.**


### fail — idx 4578

- **source**: We went to the conference that year.
- **target**: We went to the conference ~~that year.~~ **in 2022.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~We went to the conference that year.~~ **20232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323**
- `ef32` [exact ✗ · FRR ✗]: We went to the conference ~~that year.~~ **2 22220000 222020020000000000.0**
- `steer` [exact ✗ · FRR ✗]: ~~We went to the conference that year.~~ **20232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323**

## temporal_prefix

### near — idx 4604

- **source**: They held a post-match press conference.
- **target**: They held a ~~post-match~~ press ~~conference.~~ **conference after the match.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They held ~~a post-match press~~ **the same the same same the same the same the the same the the the same the same the the the the same the same the the same the the same the same last** conference.
- `ef32` [exact ✗ · FRR ✓]: They held ~~a post-match press~~ **the same the same same the same the same the the same the the the same the same the the the the same the same the the same the the same the same last** conference.
- `steer` [exact ✗ · FRR ✓]: They held a ~~post-match~~ **post- match** press conference.


### near — idx 4643

- **source**: They observed a mid-decade trend shift.
- **target**: They observed a ~~mid-decade~~ trend ~~shift.~~ **shift in the middle of the decade.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: They observed a ~~mid-decade trend shift.~~ **mid- decade.**
- `ef32` [exact ✗ · FRR ✓]: They observed a ~~mid-decade~~ **the the the the the the the** trend ~~shift.~~ **in the the of the the the the in the the the the the the the of the the the the the the the the the.**
- `steer` [exact ✗ · FRR ✓]: They observed a ~~mid-decade trend shift.~~ **mid- decade.**


### fail — idx 4625

- **source**: The midterm break was much-needed.
- **target**: The ~~midterm~~ break **halfway through the term** was much-needed.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The ~~midterm~~ **mid-term** break was ~~much-needed.~~ **much needed.**
- `ef32` [exact ✗ · FRR ✗]: The ~~midterm~~ **term " " " the** break ~~was much-needed.~~ **the the the term " the term " " the term " " the half the the year much of the the-needed.**
- `steer` [exact ✗ · FRR ✗]: The ~~midterm~~ **mid-term** break was ~~much-needed.~~ **much needed.**


### fail — idx 4648

- **source**: The post-interview reflections were positive.
- **target**: The ~~post-interview~~ reflections **after the interview** were positive.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The ~~post-interview reflections were positive.~~
- `ef32` [exact ✗ · FRR ✗]: The ~~post-interview reflections were positive.~~
- `steer` [exact ✗ · FRR ✗]: The ~~post-interview~~ **post- interview** reflections were positive.

## third_person_singular

### success — idx 4668

- **source**: The river flows gently.
- **target**: The river ~~flows~~ **flow** gently.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The river ~~flows~~ **flow** gently.
- `ef32` [exact ✓ · FRR ✓]: The river ~~flows~~ **flow** gently.
- `steer` [exact ✗ · FRR ✓]: The river ~~flows~~ **flowed** gently.


### success — idx 4658

- **source**: The dog barks at strangers.
- **target**: The dog ~~barks~~ **bark** at strangers.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The dog ~~barks~~ **bark** at strangers.
- `ef32` [exact ✓ · FRR ✓]: The dog ~~barks~~ **bark** at strangers.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 4669

- **source**: She enjoys reading novels.
- **target**: She ~~enjoys~~ **enjoy** reading novels.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~enjoys reading~~ **read** novels.
- `ef32` [exact ✗ · FRR ✓]: She ~~enjoys reading~~ **read** novels.
- `steer` [exact ✗ · FRR ✗]: She ~~enjoys~~ **likes** reading novels.


### near — idx 4672

- **source**: She washes her hands frequently.
- **target**: She ~~washes~~ **wash** her hands frequently.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~washes~~ **was going** her hands frequently.
- `ef32` [exact ✗ · FRR ✓]: She ~~washes~~ **was going** her hands frequently.
- `steer` [exact ✗ · FRR ✓]: She ~~washes~~ **frequently washed** her ~~hands frequently.~~ **hands.**


### fail — idx 4665

- **source**: He plays guitar beautifully.
- **target**: He ~~plays~~ **play** guitar beautifully.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: He plays **the** guitar beautifully.


### fail — idx 4657

- **source**: She teaches English at the university.
- **target**: She ~~teaches~~ **teach** English at the university.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: She teaches English at ~~the~~ **a** university.

## transitional

### success — idx 4709

- **source**: He apologized sincerely; nonetheless, she refused forgiveness.
- **target**: He apologized ~~sincerely; nonetheless, she~~ **sincerely. She** refused forgiveness.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: He apologized ~~sincerely; nonetheless, she~~ **sincerely. She** refused forgiveness.
- `ef32` [exact ✗ · FRR ✓]: He ~~apologized sincerely; nonetheless, she refused forgiveness.~~ **apologized. She said that**
- `steer` [exact ✓ · FRR ✓]: He apologized ~~sincerely; nonetheless, she~~ **sincerely. She** refused forgiveness.


### success — idx 4748

- **source**: Architects designed earthquake-resistant buildings; however, foundations cracked.
- **target**: Architects designed earthquake-resistant ~~buildings; however, foundations~~ **buildings. Foundations** cracked.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: Architects designed earthquake-resistant ~~buildings; however, foundations~~ **buildings. Foundations** cracked.
- `ef32` [exact ✗ · FRR ✓]: Architects designed ~~earthquake-resistant buildings; however, foundations cracked.~~ **buildings..**
- `steer` [exact ✓ · FRR ✓]: Architects designed earthquake-resistant ~~buildings; however, foundations~~ **buildings. Foundations** cracked.


### near — idx 4718

- **source**: Children laughed joyfully while parents looked exhausted.
- **target**: Children laughed ~~joyfully while parents~~ **joyfully. Parents** looked exhausted.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Children laughed joyfully ~~while~~ **and** parents looked exhausted.
- `ef32` [exact ✗ · FRR ✓]: Children laughed joyfully ~~while~~ **and** parents looked exhausted.
- `steer` [exact ✓ · FRR ✓]: Children laughed ~~joyfully while parents~~ **joyfully. Parents** looked exhausted.


### near — idx 4713

- **source**: They rehearsed tirelessly, but the performance was canceled.
- **target**: They rehearsed ~~tirelessly, but the~~ **tirelessly. The** performance was canceled.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They ~~rehearsed tirelessly, but the~~ **rehearsed. The** performance was canceled.
- `ef32` [exact ✗ · FRR ✓]: They ~~rehearsed tirelessly, but the~~ **rehearsed. The** performance was canceled.
- `steer` [exact ✓ · FRR ✓]: They rehearsed ~~tirelessly, but the~~ **tirelessly. The** performance was canceled.


### fail — idx 4722

- **source**: Mountains appear tranquil even though avalanches occur frequently.
- **target**: Mountains appear ~~tranquil even though avalanches~~ **tranquil. Avalanches** occur frequently.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: Mountains appear ~~tranquil even though avalanches occur frequently.~~ **tranquil, despite frequent avalanches.**
- `ef32` [exact ✗ · FRR ✓]: Mountains appear tranquil ~~even though avalanches occur frequently.~~ **and and are avalaveded by.**
- `steer` [exact ✗ · FRR ✗]: Mountains appear ~~tranquil even though avalanches occur frequently.~~ **tranquil, despite frequent avalanches.**

## transitive_verb

### success — idx 4767

- **source**: She lost consciousness.
- **target**: She ~~lost consciousness.~~ **fainted.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~lost consciousness.~~ **fainted.**
- `ef32` [exact ✓ · FRR ✓]: She ~~lost consciousness.~~ **fainted.**
- `steer` [exact ✓ · FRR ✓]: She ~~lost consciousness.~~ **fainted.**


### success — idx 4782

- **source**: The system stopped operating.
- **target**: The system ~~stopped operating.~~ **crashed.**

- `ours` via ef32 [exact ✓ · FRR —] ←focus: The system ~~stopped operating.~~ **crashed.**
- `ef32` [exact ✓ · FRR —]: The system ~~stopped operating.~~ **crashed.**
- `steer` [exact ✗ · FRR —]: The system ~~stopped operating.~~ **ceased operation.**


### near — idx 4787

- **source**: The novel announced its conclusion.
- **target**: The novel ~~announced its conclusion.~~ **ended.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The novel ~~announced its conclusion.~~ **is in the 10000,0,0,,0,0,0,, end of of the the, book with the.**
- `ef32` [exact ✗ · FRR ✓]: The novel ~~announced its conclusion.~~ **is in the 10000,0,0,,0,0,0,, end of of the the, book with the.**
- `steer` [exact ✗ · FRR ✓]: The novel ~~announced its conclusion.~~ **concluded.**


### near — idx 4761

- **source**: The phone made a ringing sound.
- **target**: The phone ~~made a ringing sound.~~ **rang.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The phone ~~made a ringing sound.~~ **rang,.**
- `ef32` [exact ✗ · FRR ✓]: The phone ~~made a ringing sound.~~ **rang,.**
- `steer` [exact ✓ · FRR ✓]: The phone ~~made a ringing sound.~~ **rang.**

## turn_taking

### success — idx 4838

- **source**: It’s your turn to voice concerns about scalability.
- **target**: ~~It’s your turn to voice~~ **Voice** concerns about scalability.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~It’s your turn to voice~~ **Voice** concerns about scalability.
- `ef32` [exact ✓ · FRR ✓]: ~~It’s your turn to voice~~ **Voice** concerns about scalability.
- `steer` [exact ✗ · FRR ✓]: ~~It’s your turn to voice concerns about scalability.~~ **Voice**


### success — idx 4823

- **source**: Sorry to interrupt, who will chair the next meeting?
- **target**: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?
- `ef32` [exact ✓ · FRR ✓]: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?
- `steer` [exact ✓ · FRR ✓]: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?


### near — idx 4834

- **source**: Sorry, quick interruption: the staging environment is offline.
- **target**: ~~Sorry, quick interruption: the~~ **The** staging environment is offline.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Sorry, quick interruption: the staging~~ **The** environment is offline.
- `ef32` [exact ✗ · FRR ✓]: ~~Sorry, quick interruption: the staging~~ **The** environment is offline.
- `steer` [exact ✗ · FRR ✓]: ~~Sorry, quick interruption: the staging environment is offline.~~ **```**


### near — idx 4844

- **source**: I’m done, please outline the user onboarding flow.
- **target**: ~~I’m done, please outline~~ **Outline** the user onboarding flow.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~I’m done, please outline the user onboarding flow.~~ **Overview**
- `ef32` [exact ✗ · FRR ✓]: ~~I’m done, please outline the user onboarding flow.~~ **Overview**
- `steer` [exact ✗ · FRR ✓]: ~~I’m done, please outline the~~ **Outline of** user onboarding ~~flow.~~ **flow**


### fail — idx 4846

- **source**: Let me say first that the network configuration is outdated.
- **target**: ~~Let me say first that the~~ **The** network configuration is outdated.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Let me say first that the network configuration is outdated.~~ **The**
- `ef32` [exact ✗ · FRR ✗]: ~~Let me say first that the network configuration is outdated.~~ **The**
- `steer` [exact ✗ · FRR ✗]: ~~Let me say first that the network configuration is outdated.~~ **```**

## universal_quantifiers

### near — idx 4860

- **source**: All the water evaporated.
- **target**: ~~All the~~ **The** water **completely** evaporated.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~All the~~ **The** water evaporated.
- `ef32` [exact ✗ · FRR ✓]: ~~All the water~~ **The completely water, completely, completely completely, completely completely, completely completely, completely completely, completely, completely completely completely, completely completely completely** evaporated.
- `steer` [exact ✗ · FRR ✓]: ~~All the~~ **The** water evaporated.


### near — idx 4884

- **source**: All cables were untangled.
- **target**: ~~All~~ **The** cables were **fully** untangled.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~All~~ **The full** cables were untangled.
- `ef32` [exact ✗ · FRR ✓]: ~~All~~ **The full** cables were untangled.
- `steer` [exact ✗ · FRR ✓]: ~~All~~ **The** cables were untangled.


### fail — idx 4853

- **source**: All employees received a bonus.
- **target**: ~~All~~ **The** employees received ~~a~~ **the entire** bonus.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~All employees~~ **Every employee** received a bonus.
- `ef32` [exact ✗ · FRR ✓]: ~~All employees~~ **The entire entire entire entire entire amount entire entire amount amount of of the of the entire of the the the entire entire amount of of** received ~~a~~ **received** bonus.
- `steer` [exact ✗ · FRR ✗]: ~~All employees~~ **Every employee** received a bonus.

## verbal_suffix

### success — idx 4949

- **source**: The team has worked hard to solidify their position in the market.
- **target**: The team has worked hard to ~~solidify~~ **solid** their position in the market.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The team has worked hard to ~~solidify~~ **solid** their position in the market.
- `ef32` [exact ✓ · FRR ✓]: The team has worked hard to ~~solidify~~ **solid** their position in the market.
- `steer` [exact ✗ · FRR ✗]: The team has worked hard to solidify their ~~position in the market.~~ **market position.**


### near — idx 4934

- **source**: The manager will facilitate the meeting.
- **target**: The manager will ~~facilitate~~ **facility** the meeting.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The manager will ~~facilitate~~ **facilities** the meeting.
- `ef32` [exact ✗ · FRR ✓]: The manager will ~~facilitate~~ **facilities** the meeting.
- `steer` [exact ✗ · FRR ✗]: The manager will ~~facilitate~~ **be facilitating** the meeting.


### near — idx 4916

- **source**: The artist tried to beautify the garden.
- **target**: The artist tried to ~~beautify~~ **beauty** the garden.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The artist tried to beautify **and** the garden.
- `ef32` [exact ✗ · FRR ✓]: The artist tried to beautify **and** the garden.
- `steer` [exact ✗ · FRR ✗]: The artist ~~tried~~ **sought** to beautify the garden.


### fail — idx 4943

- **source**: She wanted to simplify the instructions.
- **target**: She wanted to ~~simplify~~ **simple** the instructions.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: She wanted to ~~simplify~~ **make** the ~~instructions.~~ **instructions simpler.**


### fail — idx 4929

- **source**: He was able to stabilize the situation.
- **target**: He was able to ~~stabilize~~ **stable** the situation.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: He ~~was able to stabilize~~ **successfully stabilized** the situation.

