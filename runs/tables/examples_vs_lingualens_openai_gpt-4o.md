# LinguaLens transformation examples — focus: ours


**Coverage**: success 60, near 88, fail 42 over 97 features.

Systems: `ours` (routed), `ef32` (k32), `lingualens` (clamp10), `steer` (steer0.5). Pairs where every system has an output: 499.

Categories (= the residual-frontier decomposition):

- **success** — focus system exact-matched the target
- **near** — exact miss, but the judge saw the feature realized in the commanded direction — directionally realizable, not exactly editable
- **fail** — exact miss and not realized — the unreachable end

Outputs are word-diffed against the SOURCE: **added/substituted**, ~~removed~~. `exact` is against the target; `FRR` is the judge's realized verdict (— = not judged for that system).

## active_verbs

### near — idx 15

- **source**: We fixed the leaky faucet yesterday.
- **target**: ~~We fixed the~~ **The** leaky faucet **was fixed by us** yesterday.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~We fixed the leaky faucet~~ **It was repaired** yesterday.
- `ef32` [exact ✗ · FRR ✓]: ~~We~~ **were not by by by a by us a by by by us by by, by by a by by by by us, by by** fixed ~~the leaky~~ **by us** faucet **by** yesterday.
- `lingualens` [exact ✗ · FRR —]: We fixed the leaky ~~faucet yesterday.~~ **faucet.**
- `steer` [exact ✗ · FRR ✓]: ~~We fixed the leaky faucet~~ **It was repaired** yesterday.


### fail — idx 42

- **source**: They fixed the error in the software code.
- **target**: ~~They fixed the~~ **The** error in the software ~~code.~~ **code was fixed by them.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~They~~ fixed ~~the error in the software~~ **by by by by by by by a by a by a by by a a by by a by a a by a by by by a by a a by a by a** code.
- `lingualens` [exact ✗ · FRR —]: ~~They fixed the error in the software code.~~ **```**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## adjectival_suffix

### success — idx 88

- **source**: The student was obedient.
- **target**: The student was ~~obedient.~~ **obey.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The student was ~~obedient.~~ **obey.**
- `ef32` [exact ✓ · FRR ✓]: The student was ~~obedient.~~ **obey.**
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: The student was ~~obedient.~~ **obediently.**


### near — idx 71

- **source**: The golden crown sparkled.
- **target**: The ~~golden~~ **gold** crown sparkled.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~golden crown~~ **gold crowns** sparkled.
- `ef32` [exact ✗ · FRR ✓]: The ~~golden crown~~ **gold crowns** sparkled.
- `lingualens` [exact ✗ · FRR —]: The ~~golden~~ crown ~~sparkled.~~ **is gilded.**
- `steer` [exact ✗ · FRR ✓]: The ~~golden~~ **gleaming** crown sparkled.


### fail — idx 61

- **source**: They explored cultural differences.
- **target**: They explored ~~cultural~~ **culture** differences.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: They explored ~~cultural differences.~~ **the differences between cultures.**
- `steer` [exact ✗ · FRR ✗]: They ~~explored~~ **examined** cultural differences.

## adverbial_suffix

### near — idx 146

- **source**: She explained thoughtfully.
- **target**: She explained ~~thoughtfully.~~ **thoughtful.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~explained thoughtfully.~~ **thoughtful.**
- `ef32` [exact ✗ · FRR ✓]: She ~~explained thoughtfully.~~ **thoughtful.**
- `lingualens` [exact ✗ · FRR —]: ~~She explained thoughtfully.~~ **He was thoughtful.**
- `steer` [exact ✗ · FRR ✗]: She ~~explained thoughtfully.~~ **thoughtfully explained.**


### fail — idx 114

- **source**: She sings beautifully.
- **target**: She sings ~~beautifully.~~ **beautiful.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: She ~~sings beautifully.~~ **sings.**
- `steer` [exact ✗ · FRR ✓]: She sings ~~beautifully.~~ **beautifully songs.**

## agentive_suffix

### near — idx 184

- **source**: The operator fixed the machine.
- **target**: The ~~operator~~ **operate** fixed the machine.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~operator~~ **function** fixed the machine.
- `ef32` [exact ✗ · FRR ✓]: The ~~operator~~ **function** fixed the machine.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: The operator ~~fixed~~ **repaired** the machine.


### fail — idx 165

- **source**: She worked with a famous consultant.
- **target**: She worked with a famous ~~consultant.~~ **consult.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: She worked with a famous ~~consultant.~~ ****consultant**-**in**<strong>g</strong>**
- `steer` [exact ✗ · FRR ✗]: She worked with a ~~famous~~ **famouse** consultant.

## anaphor

### success — idx 202

- **source**: The boys played their game happily.
- **target**: The boys played ~~their~~ **a** game happily.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The boys played ~~their~~ **a** game happily.
- `ef32` [exact ✓ · FRR ✓]: The boys played ~~their~~ **a** game happily.
- `lingualens` [exact ✗ · FRR —]: The boys played their game ~~happily.~~ **happily.versy**
- `steer` [exact ✓ · FRR ✓]: The boys played ~~their~~ **a** game happily.

## appositives

### near — idx 268

- **source**: Their house, a historic landmark, attracts visitors.
- **target**: Their house, **which is** a historic landmark, attracts visitors.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Their house, a historic landmark, attracts visitors.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✓]: Their ~~house,~~ **house,, a, is, is,, is** a **is a,, and is, a, is,, is, very** historic landmark, attracts visitors.
- `lingualens` [exact ✗ · FRR —]: ~~Their house, a historic landmark, attracts visitors.~~ **\n**
- `steer` [exact ✗ · FRR ✓]: ~~Their house, a historic landmark, attracts visitors.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 256

- **source**: The puppy, a golden retriever, chewed my slippers.
- **target**: The puppy, **which is** a golden retriever, chewed my slippers.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The puppy, a **very** golden retriever, chewed my slippers.
- `ef32` [exact ✗ · FRR ✗]: The puppy, a **very** golden retriever, chewed my slippers.
- `lingualens` [exact ✗ · FRR —]: The puppy, a golden retriever, ~~chewed my slippers.~~ **is a little bit of trouble**
- `steer` [exact ✗ · FRR ✓]: ~~The puppy, a golden retriever, chewed my slippers.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## clausal_subjects

### success — idx 337

- **source**: That you care means a lot to me.
- **target**: ~~That you care~~ **Your caring** means a lot to me.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~That you care~~ **Your caring** means a lot to me.
- `ef32` [exact ✗ · FRR ✓]: ~~That you~~ **Our beloved beloved** care means a lot ~~to~~ **our own own own our own own beloved beloved beloved own heart own own heart heart heart heart own heart heart own heart heart heart heart** me.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✓ · FRR ✓]: ~~That you care~~ **Your caring** means a lot to me.


### near — idx 311

- **source**: What matters is honesty.
- **target**: ~~What matters is honesty.~~ **Honesty matters.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~What matters is~~ **Honionionionededed by him his voice voice own voice own him-ion-rere-re him, his own voice** honesty.
- `ef32` [exact ✗ · FRR ✓]: ~~What matters is~~ **Honionionionededed by him his voice voice own voice own him-ion-rere-re him, his own voice** honesty.
- `lingualens` [exact ✗ · FRR —]: ~~What matters is honesty.~~ **```python**
- `steer` [exact ✓ · FRR ✓]: ~~What matters is honesty.~~ **Honesty matters.**

## cleft_sentences

### success — idx 351

- **source**: It was in June that we married.
- **target**: ~~It was~~ **We married** in ~~June that we married.~~ **June.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~It was~~ **We married** in ~~June that we married.~~ **June.**
- `ef32` [exact ✗ · FRR ✓]: ~~It was~~ **Marriage** in June ~~that we married.~~
- `lingualens` [exact ✗ · FRR —]: ~~It was~~ **We were married** in ~~June that we married.~~ **June.**
- `steer` [exact ✓ · FRR ✓]: ~~It was~~ **We married** in ~~June that we married.~~ **June.**


### near — idx 376

- **source**: What scared the cat was the vacuum cleaner.
- **target**: ~~What~~ **The vacuum cleaner** scared the ~~cat was the vacuum cleaner.~~ **cat.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~What scared the~~ **The** cat was **scared by** the vacuum cleaner.
- `ef32` [exact ✗ · FRR ✓]: ~~What scared the cat was the~~ **The C.CC.C 121221.22 term. is "-Cs-Cs cat'sss** vacuum cleaner.
- `lingualens` [exact ✗ · FRR —]: ~~What scared the cat was the vacuum cleaner.~~ **```**
- `steer` [exact ✗ · FRR ✓]: ~~What scared the~~ **The** cat was **scared by** the vacuum cleaner.


### fail — idx 388

- **source**: What worries me is the deadline.
- **target**: ~~What~~ **The deadline** worries ~~me is the deadline.~~ **me.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~What worries me~~ **The National deadline** is ~~the deadline.~~ **for submission is for submission of for submission of of of for submission of of of of for of submission of of of for submission submission of March .**
- `lingualens` [exact ✗ · FRR —]: ~~What~~ **The deadline is what** worries ~~me is the deadline.~~ **me.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## commisive

### success — idx 418

- **source**: I swear the alarm will sound at dawn.
- **target**: ~~I swear the~~ **The** alarm ~~will sound~~ **sounds** at dawn.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~I swear the~~ **The** alarm ~~will sound~~ **sounds** at dawn.
- `ef32` [exact ✗ · FRR ✓]: ~~I swear the alarm will~~ **The** sound ~~at dawn.~~
- `lingualens` [exact ✗ · FRR —]: ~~I swear the alarm will sound at dawn.~~ **```**
- `steer` [exact ✓ · FRR ✓]: ~~I swear the~~ **The** alarm ~~will sound~~ **sounds** at dawn.


### near — idx 401

- **source**: We will have dinner ready by 7.
- **target**: ~~We will have dinner~~ **Dinner is** ready ~~by~~ **at** 7.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~We will have dinner ready by 7.~~
- `ef32` [exact ✗ · FRR ✓]: ~~We will have dinner ready by 7.~~
- `lingualens` [exact ✗ · FRR —]: We will have dinner ready ~~by~~ **at** 7.
- `steer` [exact ✗ · FRR ✓]: ~~We~~ **Dinner** will ~~have dinner~~ ready by 7.

## comparative

### success — idx 473

- **source**: That sofa is softer than mine.
- **target**: That sofa is ~~softer~~ **soft** than mine.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: That sofa is ~~softer~~ **soft** than mine.
- `ef32` [exact ✓ · FRR ✓]: That sofa is ~~softer~~ **soft** than mine.
- `lingualens` [exact ✗ · FRR —]: That sofa is ~~softer than mine.~~ **soft.**
- `steer` [exact ✗ · FRR ✗]: That sofa is softer than ~~mine.~~ **my sofa.**


### near — idx 483

- **source**: She is cleverer than people think.
- **target**: She is ~~cleverer~~ **clever** than people think.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She is ~~cleverer~~ than people think.
- `ef32` [exact ✗ · FRR ✓]: She is ~~cleverer~~ than people think.
- `lingualens` [exact ✗ · FRR —]: She is ~~cleverer than people think.~~ **cleverer.**
- `steer` [exact ✗ · FRR ✓]: ~~She is cleverer than people think.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### fail — idx 492

- **source**: That theory is more logical than the last.
- **target**: That theory is ~~more~~ logical than the last.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## coordination

### success — idx 543

- **source**: The team overlooked minor errors not to mention critical flaws.
- **target**: The team overlooked minor ~~errors not to mention critical flaws.~~ **errors.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The team overlooked minor ~~errors not to mention critical flaws.~~ **errors.**
- `ef32` [exact ✓ · FRR ✓]: The team overlooked minor ~~errors not to mention critical flaws.~~ **errors.**
- `lingualens` [exact ✗ · FRR —]: The team overlooked minor errors ~~not to mention~~ **and** critical flaws.
- `steer` [exact ✗ · FRR ✓]: ~~The team overlooked minor errors not to mention critical flaws.~~ **rowspan rowspan rowspan**

## copular_be

### success — idx 558

- **source**: The exam is tomorrow.
- **target**: The exam ~~is~~ **takes place** tomorrow.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The exam ~~is~~ **takes place** tomorrow.
- `ef32` [exact ✓ · FRR ✓]: The exam ~~is~~ **takes place** tomorrow.
- `lingualens` [exact ✓ · FRR —]: The exam ~~is~~ **takes place** tomorrow.
- `steer` [exact ✗ · FRR ✗]: The exam is ~~tomorrow.~~ **the following day.**


### near — idx 551

- **source**: John is a pirate.
- **target**: John ~~is~~ **works as** a pirate.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: John ~~is~~ **as as as as as as as as as as** a pirate.
- `ef32` [exact ✗ · FRR ✓]: John ~~is~~ **as as as as as as as as as as** a pirate.
- `lingualens` [exact ✗ · FRR —]: John is a ~~pirate.~~ **software engineer.**
- `steer` [exact ✗ · FRR ✗]: John is a ~~pirate.~~ **buccaneer.**

## count_nouns

### near — idx 639

- **source**: I need three forks and spoons.
- **target**: I need ~~three forks and spoons.~~ **cutlery items.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I need ~~three forks and spoons.~~ **to items, items.**
- `ef32` [exact ✗ · FRR ✓]: I need ~~three forks and spoons.~~ **to items, items.**
- `lingualens` [exact ✗ · FRR —]: I need ~~three forks and spoons.~~ **three.**
- `steer` [exact ✗ · FRR ✓]: I need three forks ~~and spoons.~~ **and.**

## declaration

### success — idx 692

- **source**: A statute legitimizes a quota system remains in place.
- **target**: A ~~statute legitimizes a~~ quota system remains in place.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: A ~~statute legitimizes a~~ quota system remains in place.
- `ef32` [exact ✓ · FRR ✓]: A ~~statute legitimizes a~~ quota system remains in place.
- `lingualens` [exact ✗ · FRR —]: A statute **remains in place that** legitimizes a quota ~~system remains in place.~~ **system.**
- `steer` [exact ✗ · FRR ✗]: ~~A statute legitimizes a quota system remains in place.~~ **rowspan rowspan**


### fail — idx 675

- **source**: The jury acquits defendants are found innocent.
- **target**: ~~The jury acquits defendants~~ **Defendants** are found innocent.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~The jury acquits defendants are~~ **Defendants** found ~~innocent.~~ **innocent by the jury.**
- `ef32` [exact ✗ · FRR ✗]: ~~The jury acquits defendants~~ **Deits,,,,,,,, and, and and, and de defendants, and de de de, and de de, and** are ~~found~~ **not,,** innocent.
- `lingualens` [exact ✗ · FRR —]: ~~The jury acquits defendants are found innocent.~~ **```python**
- `steer` [exact ✗ · FRR ✗]: ~~The jury acquits defendants are~~ **Defendants** found ~~innocent.~~ **innocent by the jury.**

## degree_prefix

### success — idx 726

- **source**: This camera captures ultrawide images.
- **target**: This camera captures ~~ultrawide~~ **wide** images.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: This camera captures ~~ultrawide~~ **wide** images.
- `ef32` [exact ✓ · FRR ✓]: This camera captures ~~ultrawide~~ **wide** images.
- `lingualens` [exact ✗ · FRR —]: This camera captures ~~ultrawide images.~~ **a wide image.**
- `steer` [exact ✗ · FRR ✓]: This camera captures ~~ultrawide~~ **wide-angle** images.


### near — idx 729

- **source**: She felt underappreciated.
- **target**: She felt ~~underappreciated.~~ **appreciated.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~felt underappreciated.~~ **feltappreciated.**
- `ef32` [exact ✗ · FRR ✓]: She ~~felt underappreciated.~~ **feltappreciated.**
- `lingualens` [exact ✗ · FRR —]: She ~~felt underappreciated.~~ **felt.**
- `steer` [exact ✗ · FRR ✓]: She felt ~~underappreciated.~~ **valued.**


### fail — idx 741

- **source**: The film depicts an archetypal hero.
- **target**: The film depicts ~~an archetypal~~ **a typical** hero.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The film depicts an ~~archetypal hero.~~ **ordinary life of a,, of a normal person,,,, a.**
- `ef32` [exact ✗ · FRR ✗]: The film depicts an ~~archetypal hero.~~ **ordinary life of a,, of a normal person,,,, a.**
- `lingualens` [exact ✗ · FRR —]: The film depicts an ~~archetypal~~ **arche** hero.
- `steer` [exact ✗ · FRR ✗]: The film ~~depicts an archetypal hero.~~ **features a hero archetype.**

## deixis

### success — idx 757

- **source**: You did an excellent job on the report.
- **target**: ~~You~~ **Frank** did an excellent job on the report.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~You~~ **Frank** did an excellent job on the report.
- `ef32` [exact ✓ · FRR ✓]: ~~You~~ **Frank** did an excellent job on the report.
- `lingualens` [exact ✓ · FRR —]: ~~You~~ **Frank** did an excellent job on the report.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 774

- **source**: We celebrated our anniversary together.
- **target**: ~~We~~ **David and Emma** celebrated ~~our~~ **the** anniversary together.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~We celebrated our anniversary together.~~ **The**
- `ef32` [exact ✗ · FRR ✓]: ~~We~~ **and** celebrated ~~our~~ **the and David David Emma and and David and Emma Emma the David and and Emma David and David David David and and Emma** anniversary **and, and David David** together.
- `lingualens` [exact ✗ · FRR —]: ~~We celebrated our anniversary together.~~ **The sentence is a bit more formal than necessary. Let's make it a little more concise.**
- `steer` [exact ✗ · FRR ✓]: ~~We celebrated our anniversary together.~~ **The**

## deontic

### success — idx 813

- **source**: She might visit her grandparents.
- **target**: She ~~might visit~~ **visits** her grandparents.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~might visit~~ **visits** her grandparents.
- `ef32` [exact ✓ · FRR ✓]: She ~~might visit~~ **visits** her grandparents.
- `lingualens` [exact ✗ · FRR —]: She might visit ~~her~~ **his** grandparents.
- `steer` [exact ✗ · FRR ✗]: She ~~might visit~~ **possibly visits** her grandparents.


### near — idx 833

- **source**: She can take the bus.
- **target**: She ~~can take~~ **takes** the bus.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~can take~~ **gives** the bus.
- `ef32` [exact ✗ · FRR ✓]: She ~~can take~~ **gives** the bus.
- `lingualens` [exact ✓ · FRR —]: She ~~can take~~ **takes** the bus.
- `steer` [exact ✓ · FRR ✓]: She ~~can take~~ **takes** the bus.

## direct_object

### near — idx 914

- **source**: She found the keys.
- **target**: ~~She~~ **The keys were** found ~~the keys.~~ **by her.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~found the~~ **was was her was a her a her her her her her her her own own keyboard hands own keyboard own keyboard hands keyboard her hand by** keys.
- `ef32` [exact ✗ · FRR ✓]: She ~~found the~~ **was was her was a her a her her her her her her her own own keyboard hands own keyboard own keyboard hands keyboard her hand by** keys.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: She ~~found~~ **discovered** the keys.

## directive

### success — idx 854

- **source**: I command students to memorize this formula.
- **target**: ~~I command students to~~ **Students** memorize this formula.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~I command students to~~ **Students** memorize this formula.
- `ef32` [exact ✓ · FRR ✓]: ~~I command students to~~ **Students** memorize this formula.
- `lingualens` [exact ✗ · FRR —]: ~~I command students to memorize this formula.~~ **Student**
- `steer` [exact ✗ · FRR ✓]: ~~I command students to memorize this formula.~~ **Students:**


### near — idx 860

- **source**: I insist you finish the soup.
- **target**: ~~I insist you~~ **You** finish the soup.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~I~~ **You** insist you finish the soup.
- `ef32` [exact ✗ · FRR ✓]: ~~I~~ **You** insist you finish the soup.
- `lingualens` [exact ✗ · FRR —]: ~~I insist you~~ **You must** finish the soup.
- `steer` [exact ✓ · FRR ✓]: ~~I insist you~~ **You** finish the soup.

## discourse_markers

### success — idx 962

- **source**: However, it’s still worth a try.
- **target**: ~~However, it’s~~ **It’s** still worth a try.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~However,~~ it’s still worth a try.
- `ef32` [exact ✓ · FRR ✓]: ~~However,~~ it’s still worth a try.
- `lingualens` [exact ✗ · FRR —]: ~~However, it’s~~ **It's** still worth a try.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 955

- **source**: Seriously, this is the best option.
- **target**: ~~Seriously, this~~ **This** is the best option.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Seriously, this is the best option.~~ **This**
- `ef32` [exact ✗ · FRR ✓]: ~~Seriously, this is the best option.~~ **This**
- `lingualens` [exact ✓ · FRR —]: ~~Seriously, this~~ **This** is the best option.
- `steer` [exact ✓ · FRR ✓]: ~~Seriously, this~~ **This** is the best option.

## echo_questions

### near — idx 1029

- **source**: "What did he say?" What did he say? “Goodbye.”
- **target**: "What did he say?" ~~What did he say?~~ **He said** “Goodbye.”

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~"What did he say?" What did he say? “Goodbye.”~~ **He said goodbye**
- `ef32` [exact ✗ · FRR ✓]: "What ~~did~~ he ~~say?" What did~~ **said,.?" said,, and** he ~~say? “Goodbye.”~~ **said, " "I'm, "IGoodbye,**
- `lingualens` [exact ✗ · FRR —]: ~~"What did he say?" What did he say? “Goodbye.”~~ **```**
- `steer` [exact ✗ · FRR ✓]: ~~"What did he say?" What did he say? “Goodbye.”~~ **He said goodbye**

## elliptical_sentences

### near — idx 1078

- **source**: I prefer tea, John coffee.
- **target**: I prefer tea, **and** John **prefers** coffee.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I prefer tea, **and I I do I do do I do do I do do I do do I, do I do, do not** John coffee.
- `ef32` [exact ✗ · FRR ✓]: I prefer tea, **and I I do I do do I do do I do do I do do I, do I do, do not** John coffee.
- `lingualens` [exact ✗ · FRR —]: I prefer tea, ~~John~~ **and** coffee.
- `steer` [exact ✗ · FRR ✓]: ~~I prefer tea, John coffee.~~ **" " " " " " " " " " " " " " and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and**


### fail — idx 1065

- **source**: She plays more than I do.
- **target**: She plays more than I ~~do.~~ **play.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: She plays more than ~~I do.~~ **you.**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## emphatic_structure

### success — idx 1138

- **source**: You do speak French.
- **target**: You ~~do~~ speak French.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: You ~~do~~ speak French.
- `ef32` [exact ✓ · FRR ✓]: You ~~do~~ speak French.
- `lingualens` [exact ✗ · FRR —]: You ~~do~~ **might** speak French.
- `steer` [exact ✗ · FRR ✓]: ~~You do speak French.~~ **rowspan rowspan rowspan**


### near — idx 1128

- **source**: She did arrive early.
- **target**: She ~~did arrive~~ **arrived** early.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~did arrive~~ **joined** early.
- `ef32` [exact ✗ · FRR ✓]: She ~~did arrive~~ **joined** early.
- `lingualens` [exact ✗ · FRR —]: ~~She did arrive~~ **He arrived** early.
- `steer` [exact ✓ · FRR ✓]: She ~~did arrive~~ **arrived** early.

## epistemic

### success — idx 1166

- **source**: They must arrive at 5 PM.
- **target**: They ~~must~~ **will** arrive at 5 PM.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They ~~must~~ **will** arrive at 5 PM.
- `ef32` [exact ✓ · FRR ✓]: They ~~must~~ **will** arrive at 5 PM.
- `lingualens` [exact ✗ · FRR —]: ~~They must arrive at 5 PM.~~ **```**
- `steer` [exact ✓ · FRR ✓]: They ~~must~~ **will** arrive at 5 PM.


### near — idx 1187

- **source**: It is probable that the dog is barking loudly.
- **target**: ~~It is probable that the~~ **The** dog is barking loudly.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~It is probable that the dog is barking loudly.~~ **The**
- `ef32` [exact ✗ · FRR ✓]: ~~It is probable that the dog is barking loudly.~~ **The**
- `lingualens` [exact ✓ · FRR —]: ~~It is probable that the~~ **The** dog is barking loudly.
- `steer` [exact ✗ · FRR ✓]: ~~It is probable that the dog is barking loudly.~~ **```**

## euphemism

### near — idx 1206

- **source**: He crossed over.
- **target**: He ~~crossed over.~~ **died.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~crossed over.~~ **died in.**
- `ef32` [exact ✗ · FRR ✓]: He ~~crossed over.~~ **died in.**
- `lingualens` [exact ✗ · FRR —]: He ~~crossed over.~~ **crossed.**
- `steer` [exact ✓ · FRR ✓]: He ~~crossed over.~~ **died.**

## existential

### success — idx 1259

- **source**: There exists a solution for this equation.
- **target**: ~~There~~ **A solution** exists ~~a solution~~ for this equation.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~There~~ **A solution** exists ~~a solution~~ for this equation.
- `ef32` [exact ✗ · FRR ✓]: ~~There exists a solution for this equation.~~ **A water water is is a-waterwater which is is.**
- `lingualens` [exact ✗ · FRR —]: ~~There exists~~ **The solution to this equation is contained in** a ~~solution for this equation.~~ **set of parentheses.**
- `steer` [exact ✓ · FRR ✓]: ~~There~~ **A solution** exists ~~a solution~~ for this equation.


### near — idx 1275

- **source**: There occurs a chemical reaction spontaneously.
- **target**: ~~There occurs a~~ **A** chemical reaction **occurs** spontaneously.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~There occurs~~ **Spontaneously,** a chemical reaction ~~spontaneously.~~ **occurs.**
- `ef32` [exact ✗ · FRR ✓]: ~~There occurs a chemical reaction spontaneously.~~ **A person who has been taken spontaneously from.**
- `lingualens` [exact ✗ · FRR —]: ~~There occurs a~~ **The** chemical reaction ~~spontaneously.~~ **is spontaneous.**
- `steer` [exact ✗ · FRR ✓]: ~~There occurs~~ **Spontaneously,** a chemical reaction ~~spontaneously.~~ **occurs.**

## existential_quantifiers

### success — idx 1335

- **source**: I felt some relief.
- **target**: I felt ~~some~~ relief.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I felt ~~some~~ relief.
- `ef32` [exact ✓ · FRR ✓]: I felt ~~some~~ relief.
- `lingualens` [exact ✗ · FRR —]: I felt ~~some relief.~~ **it.**
- `steer` [exact ✗ · FRR ✓]: ~~I felt some relief.~~ **rowspan rowspan rowspan**

## expletive

### near — idx 1375

- **source**: It looks like rain.
- **target**: ~~It~~ **Rain** looks ~~like rain.~~ **imminent.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~It looks like~~ **storm imminent danger, danger, and,, and danger, and and immediately,,, and rain, and rain, and danger, and and** rain.
- `ef32` [exact ✗ · FRR ✓]: ~~It looks like~~ **storm imminent danger, danger, and,, and danger, and and immediately,,, and rain, and rain, and danger, and and** rain.
- `lingualens` [exact ✗ · FRR —]: It ~~looks like rain.~~ **seems.**
- `steer` [exact ✗ · FRR ✗]: It ~~looks like rain.~~ **appears to be raining.**

## expressive

### success — idx 1407

- **source**: We apologize that the train departs late.
- **target**: ~~We apologize that the~~ **The** train departs late.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~We apologize that the~~ **The** train departs late.
- `ef32` [exact ✓ · FRR ✓]: ~~We apologize that the~~ **The** train departs late.
- `lingualens` [exact ✗ · FRR —]: ~~We apologize that~~ **The sentence was posted at the end of** the train ~~departs~~ **is** late.
- `steer` [exact ✗ · FRR ✗]: We ~~apologize that~~ **apologize,** the train departs late.


### near — idx 1449

- **source**: We regret the road getting icy.
- **target**: ~~We regret the~~ **The** road ~~getting~~ **gets** icy.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~We regret the~~ **The** road getting icy.
- `ef32` [exact ✗ · FRR ✓]: ~~We regret the~~ **The** road getting icy.
- `lingualens` [exact ✗ · FRR —]: We ~~regret~~ **gets** the road ~~getting~~ icy.
- `steer` [exact ✗ · FRR ✓]: ~~We regret the~~ **The** road ~~getting~~ **got** icy.


### fail — idx 1433

- **source**: We apologize for the microphone feedbacking.
- **target**: ~~We apologize for the~~ **The** microphone ~~feedbacking.~~ **feedbacks.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~We apologize for the microphone feedbacking.~~ **The sentences:**
- `ef32` [exact ✗ · FRR ✗]: ~~We apologize for the microphone feedbacking.~~ **The radio stations feedback signals.**
- `lingualens` [exact ✗ · FRR —]: We apologize for the microphone ~~feedbacking.~~ **feedback.**
- `steer` [exact ✗ · FRR ✗]: ~~We apologize for the microphone feedbacking.~~ **The sentences:**

## extraposition

### near — idx 1461

- **source**: It was expected that profits would grow.
- **target**: ~~It was expected that~~ **That** profits would ~~grow.~~ **grow was expected.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: It ~~was~~ **is** expected ~~that profits would grow.~~ **to grow profits.**
- `ef32` [exact ✗ · FRR ✓]: ~~It was expected~~ **What is is meant by** that **to by definition to by revenue increase in in revenue definition of of revenue of revenue to increase by revenue revenue revenue revenue** profits ~~would grow.~~ **for.**
- `lingualens` [exact ✗ · FRR —]: It ~~was expected that profits would grow.~~ **isthe **erwägen** **von** **be** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü** **ü**
- `steer` [exact ✗ · FRR ✓]: It ~~was~~ **is** expected ~~that profits would grow.~~ **to grow profits.**


### fail — idx 1458

- **source**: It is crucial that everyone participates.
- **target**: ~~It~~ **That everyone participates** is ~~crucial that everyone participates.~~ **crucial.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: It is crucial ~~that~~ **for** everyone ~~participates.~~ **to participate.**
- `ef32` [exact ✗ · FRR ✓]: ~~It~~ **That** is ~~crucial~~ **is that,, which is is, and is is** that ~~everyone~~ **is is being used used is being is being which is being is being used** participates.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: It is crucial ~~that~~ **for** everyone ~~participates.~~ **to participate.**

## factives

### near — idx 1532

- **source**: The realization that money was missing panicked him.
- **target**: ~~The realization that money was~~ **Money** missing panicked him.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~The realization that money was missing~~ panicked him.
- `ef32` [exact ✗ · FRR ✓]: ~~The realization that money was missing~~ panicked him.
- `lingualens` [exact ✗ · FRR —]: The ~~realization that~~ money **he** was missing ~~panicked him.~~ **was a consequence of his own actions.**
- `steer` [exact ✗ · FRR ✓]: The ~~realization that money was missing panicked him.~~ **Money Missing Panic Him.**

## first_conditional

### near — idx 1580

- **source**: If they're playing outside, close the window.
- **target**: ~~If they're~~ **They’re** playing outside, **so** close the window.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~If they're~~ **So they’re** playing outside, close the window.
- `ef32` [exact ✗ · FRR ✓]: ~~If they're playing outside,~~ **So,,'tt’ss,, so so so so playing, so so so that and so on, so so so that so that that of,** close the window.
- `lingualens` [exact ✗ · FRR —]: ~~If they're~~ **If’re** playing outside, close ~~the window.~~ **it.**
- `steer` [exact ✗ · FRR ✓]: ~~If they're~~ **So they’re** playing outside, close the window.


### fail — idx 1576

- **source**: If you are driving fast, you will get a ticket.
- **target**: ~~If you~~ **You** are driving fast, **and** you ~~will~~ get a ticket.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~If~~ **You're drivingre and driving driving you're and you** you are **and** driving ~~fast,~~ **driving** you ~~will get~~ **are and driving you're driving and driving, and you driving you** a ticket.
- `lingualens` [exact ✗ · FRR —]: If ~~you are driving fast, you will get a ticket.~~ **आप गाँहलूँ**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## futurates

### success — idx 1608

- **source**: She is finishing her report by Friday.
- **target**: She ~~is finishing~~ **will finish** her report by Friday.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~is finishing~~ **will finish** her report by Friday.
- `ef32` [exact ✓ · FRR ✓]: She ~~is finishing~~ **will finish** her report by Friday.
- `lingualens` [exact ✗ · FRR —]: She ~~is finishing her~~ **will finish his** report by Friday.
- `steer` [exact ✓ · FRR ✓]: She ~~is finishing~~ **will finish** her report by Friday.


### near — idx 1607

- **source**: I am visiting my grandparents during the holidays.
- **target**: I ~~am visiting~~ **will visit** my grandparents during the holidays.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I ~~am~~ **will go** visiting my grandparents during the holidays.
- `ef32` [exact ✗ · FRR ✓]: I ~~am~~ **will go** visiting my grandparents during the holidays.
- `lingualens` [exact ✓ · FRR —]: I ~~am visiting~~ **will visit** my grandparents during the holidays.
- `steer` [exact ✓ · FRR ✓]: I ~~am visiting~~ **will visit** my grandparents during the holidays.

## future

### success — idx 1653

- **source**: He will teach chemistry.
- **target**: He ~~will teach~~ **teaches** chemistry.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~will teach~~ **teaches** chemistry.
- `ef32` [exact ✓ · FRR ✓]: He ~~will teach~~ **teaches** chemistry.
- `lingualens` [exact ✗ · FRR —]: He will ~~teach~~ **the** chemistry.
- `steer` [exact ✓ · FRR ✓]: He ~~will teach~~ **teaches** chemistry.


### near — idx 1668

- **source**: He’ll take a taxi.
- **target**: ~~He’ll take~~ **He takes** a taxi.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~He’ll take a taxi.~~ **He has**
- `ef32` [exact ✗ · FRR ✓]: ~~He’ll take a taxi.~~ **He has**
- `lingualens` [exact ✓ · FRR —]: ~~He’ll take~~ **He takes** a taxi.
- `steer` [exact ✓ · FRR ✓]: ~~He’ll take~~ **He takes** a taxi.


### fail — idx 1685

- **source**: Are they announcing the winner?
- **target**: ~~Are~~ **Do** they ~~announcing~~ **announce** the winner?

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Are~~ **Do** they announcing the winner?
- `ef32` [exact ✗ · FRR ✗]: ~~Are~~ **Do** they announcing the winner?
- `lingualens` [exact ✗ · FRR —]: ~~Are they announcing the winner?~~ **```**
- `steer` [exact ✗ · FRR ✗]: ~~Are~~ **Will** they ~~announcing~~ **announce** the winner?

## future_perfect

### success — idx 1704

- **source**: They will have eaten dinner together.
- **target**: They ~~will have eaten~~ **eat** dinner together.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They ~~will have eaten~~ **eat** dinner together.
- `ef32` [exact ✓ · FRR ✓]: They ~~will have eaten~~ **eat** dinner together.
- `lingualens` [exact ✗ · FRR —]: They ~~will have eaten dinner~~ **eat** together.
- `steer` [exact ✓ · FRR ✓]: They ~~will have eaten~~ **eat** dinner together.


### near — idx 1729

- **source**: He will have watched TV in the evenings.
- **target**: He ~~will have watched~~ **watches** TV in the evenings.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~will have~~ watched TV in the evenings.
- `ef32` [exact ✗ · FRR ✓]: He ~~will have~~ watched TV in the evenings.
- `lingualens` [exact ✓ · FRR —]: He ~~will have watched~~ **watches** TV in the evenings.
- `steer` [exact ✓ · FRR ✓]: He ~~will have watched~~ **watches** TV in the evenings.

## future_progressive

### success — idx 1756

- **source**: You are going to be watching movies.
- **target**: You ~~are going to be watching~~ **watch** movies.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: You ~~are going to be watching~~ **watch** movies.
- `ef32` [exact ✓ · FRR ✓]: You ~~are going to be watching~~ **watch** movies.
- `lingualens` [exact ✗ · FRR —]: You are ~~going to be~~ watching movies.
- `steer` [exact ✗ · FRR ✓]: ~~You are going to be watching~~ **You's watch** movies.


### near — idx 1782

- **source**: Which movies will you be watching?
- **target**: Which movies ~~will~~ **do** you ~~be watching?~~ **watch?**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Which~~ **What** movies ~~will~~ **do** you ~~be watching?~~ **watch?**
- `ef32` [exact ✓ · FRR ✓]: Which movies ~~will~~ **do** you ~~be watching?~~ **watch?**
- `lingualens` [exact ✓ · FRR —]: Which movies ~~will~~ **do** you ~~be watching?~~ **watch?**
- `steer` [exact ✗ · FRR ✓]: ~~Which~~ **What** movies ~~will~~ **do** you ~~be watching?~~ **watch?**

## given_known

### success — idx 1835

- **source**: That tree fell during storm.
- **target**: ~~That~~ **A** tree fell during storm.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~That~~ **A** tree fell during storm.
- `ef32` [exact ✓ · FRR ✓]: ~~That~~ **A** tree fell during storm.
- `lingualens` [exact ✗ · FRR —]: ~~That~~ **A** tree fell during **the** storm.
- `steer` [exact ✗ · FRR ✗]: ~~That~~ **A** tree fell during **the** storm.

## hyperbole

### near — idx 1889

- **source**: I’m so angry I could explode.
- **target**: I’m ~~so angry I could explode.~~ **very angry.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I’m ~~so angry I could explode.~~ **very excited.**
- `ef32` [exact ✗ · FRR ✓]: I’m ~~so angry I could explode.~~ **very excited.**
- `lingualens` [exact ✗ · FRR —]: ~~I’m~~ **I'm** so ~~angry~~ **angry,** I could explode.
- `steer` [exact ✗ · FRR ✓]: ~~I’m so angry~~ I ~~could explode.~~ **am very angry.**

## imperative_sentence

### success — idx 1929

- **source**: Let's finish this by tomorrow.
- **target**: ~~Let's~~ **We should** finish this by tomorrow.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Let's~~ **We should** finish this by tomorrow.
- `ef32` [exact ✓ · FRR ✓]: ~~Let's~~ **We should** finish this by tomorrow.
- `lingualens` [exact ✗ · FRR —]: ~~Let's finish this by tomorrow.~~ **```python**
- `steer` [exact ✓ · FRR ✓]: ~~Let's~~ **We should** finish this by tomorrow.


### near — idx 1913

- **source**: Stand up.
- **target**: ~~Stand~~ **You are required to stand** up.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Stand up.~~ ****to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to**
- `ef32` [exact ✗ · FRR ✓]: ~~Stand up.~~ **is a not to to a 1 1121 to to the 1 122 01 up to to.**
- `lingualens` [exact ✗ · FRR —]: ~~Stand up.~~ **You are a precise text editor.**
- `steer` [exact ✗ · FRR ✓]: ~~Stand up.~~ ****to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to**

## indirect_speech

### near — idx 1992

- **source**: He announced he had been promoted.
- **target**: He ~~announced he had~~ **announced, “I have** been ~~promoted.~~ **promoted.”**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He announced he had been ~~promoted.~~ **promoted, " " "I "I''m "mmII'mm " "mmI'mI'm'mmm**
- `ef32` [exact ✗ · FRR ✓]: He announced he had been ~~promoted.~~ **promoted, " " "I "I''m "mmII'mm " "mmI'mI'm'mmm**
- `lingualens` [exact ✗ · FRR —]: ~~He announced he had~~ **I have** been promoted. **”**
- `steer` [exact ✗ · FRR ✓]: ~~He announced he had been promoted.~~ **I waspromoted.**


### fail — idx 1956

- **source**: He warned that it might rain later.
- **target**: He ~~warned that it~~ **warned, “It** might rain ~~later.~~ **later.”**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: He warned that it might rain ~~later.~~ **later,, ' ' ' ', ' ' ' ' ' ' ' ' ' ', ' ' ' ' ' ' ' ' '**
- `ef32` [exact ✗ · FRR ✗]: He warned that it might rain ~~later.~~ **later,, ' ' ' ', ' ' ' ' ' ' ' ' ' ', ' ' ' ' ' ' ' ' '**
- `lingualens` [exact ✗ · FRR —]: He ~~warned that it~~ might ~~rain later.~~ **just be a little bit of a troublemaker.”**
- `steer` [exact ✗ · FRR ✓]: He ~~warned that it~~ **cautioned, 'It** might rain ~~later.~~ **later.'**

## intensifiers

### near — idx 2035

- **source**: I don’t very much like spicy food, to be honest.
- **target**: I don’t ~~very much~~ like spicy food, to be honest.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I don’t ~~very~~ much like spicy food, to be honest.
- `ef32` [exact ✗ · FRR ✓]: I don’t ~~very~~ much like spicy food, to be honest.
- `lingualens` [exact ✗ · FRR —]: I ~~don’t~~ **don't** very much like spicy ~~food, to be honest.~~ **food.**
- `steer` [exact ✗ · FRR ✓]: ~~I don’t very much like spicy food, to be honest.~~ **rowspan rowspan rowspan**


### fail — idx 2012

- **source**: I really need to finish this work by noon.
- **target**: I ~~really~~ need to finish this work by noon.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: I really need to finish this ~~work by noon.~~ **work.**
- `steer` [exact ✗ · FRR ✓]: ~~I really need to finish this work by noon.~~ **rowspan rowspan rowspan**

## interrogative

### success — idx 2052

- **source**: Does the engine require regular maintenance?
- **target**: ~~Does the~~ **The** engine ~~require~~ **requires** regular ~~maintenance?~~ **maintenance.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Does the~~ **The** engine ~~require~~ **requires** regular ~~maintenance?~~ **maintenance.**
- `ef32` [exact ✗ · FRR ✓]: ~~Does the~~ **The** engine ~~require regular maintenance?~~ **provides.. The 111111,11,,,1, 11,,, responsible for the.**
- `lingualens` [exact ✗ · FRR —]: ~~Does the engine require regular maintenance?~~ **```**
- `steer` [exact ✓ · FRR ✓]: ~~Does the~~ **The** engine ~~require~~ **requires** regular ~~maintenance?~~ **maintenance.**


### near — idx 2091

- **source**: When was the novel written?
- **target**: ~~When~~ **The novel** was **written during** the ~~novel written?~~ **19th century.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~When was the novel written?~~ **The 119191 119199191119991.19 111919**
- `ef32` [exact ✗ · FRR ✓]: ~~When was the novel written?~~ **The 119191 119199191119991.19 111919**
- `lingualens` [exact ✗ · FRR —]: ~~When~~ **The sentence** was **written in** the ~~novel written?~~ **1990s.**
- `steer` [exact ✗ · FRR ✓]: ~~When was the novel written?~~ **19th century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century century**

## intransitive_verb

### near — idx 2103

- **source**: He resigned.
- **target**: He ~~resigned.~~ **quit his job.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~resigned.~~ **his job job job job his job job job job job job job job job job job job his job job job job job job job job job.**
- `ef32` [exact ✗ · FRR ✓]: He ~~resigned.~~ **his job job job job his job job job job job job job job job job job job his job job job job job job job job job.**
- `lingualens` [exact ✗ · FRR —]: He ~~resigned.~~ **left.**
- `steer` [exact ✗ · FRR ✓]: He ~~resigned.~~ **was laid off.**

## linking_verb

### near — idx 2165

- **source**: The child appears hungry.
- **target**: The child ~~appears hungry.~~ **asks for food.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The child ~~appears hungry.~~ **asked for food for.**
- `ef32` [exact ✗ · FRR ✓]: The child ~~appears hungry.~~ **asked for food for.**
- `lingualens` [exact ✓ · FRR —]: The child ~~appears hungry.~~ **asks for food.**
- `steer` [exact ✗ · FRR ✗]: The child ~~appears~~ **is** hungry.


### fail — idx 2189

- **source**: The coffee tastes bitter.
- **target**: ~~The coffee tastes bitter.~~ **Excessive brewing darkens the coffee.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The coffee ~~tastes bitter.~~ **is dark roast.**
- `ef32` [exact ✗ · FRR ✓]: ~~The~~ **Dark and the the crime and the crime crime the crime crime** coffee ~~tastes bitter.~~ **the crime crime crime the crime dark is the the the dark and the the.**
- `lingualens` [exact ✗ · FRR —]: The coffee ~~tastes~~ **is** bitter.
- `steer` [exact ✗ · FRR ✗]: The coffee ~~tastes bitter.~~ **is dark roast.**

## mass_noun

### success — idx 2240

- **source**: He has data to back up his argument.
- **target**: He has ~~data~~ **numbers** to back up his argument.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He has ~~data~~ **numbers** to back up his argument.
- `ef32` [exact ✓ · FRR ✓]: He has ~~data~~ **numbers** to back up his argument.
- `lingualens` [exact ✗ · FRR —]: He has ~~data~~ **the numbers** to back up his argument.
- `steer` [exact ✗ · FRR ✗]: He has ~~data~~ **evidence** to ~~back up~~ **support** his argument.


### near — idx 2226

- **source**: She enjoys creating art in her free time.
- **target**: She enjoys ~~creating art~~ **painting and drawing** in her free time.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She enjoys ~~creating art~~ **painting and and and and and and and and and and and and and and and and and painting and painting and painting and painting** in her free time.
- `ef32` [exact ✗ · FRR ✓]: She enjoys ~~creating art~~ **painting and and and and and and and and and and and and and and and and and painting and painting and painting and painting** in her free time.
- `lingualens` [exact ✗ · FRR —]: She ~~enjoys creating art in her free time.~~ **loves to draw.**
- `steer` [exact ✗ · FRR ✓]: She ~~enjoys creating art~~ **paints** in her free time.


### fail — idx 2202

- **source**: The report contains valuable information about trends.
- **target**: The report ~~contains~~ **lists** valuable ~~information~~ **details** about trends.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The report ~~contains~~ **details** valuable information about trends.
- `ef32` [exact ✗ · FRR ✗]: The report ~~contains~~ **details** valuable information about trends.
- `lingualens` [exact ✗ · FRR —]: The report ~~contains~~ **provides** valuable information about trends.
- `steer` [exact ✗ · FRR ✗]: The report ~~contains~~ **details** valuable information about trends.

## metaphor

### near — idx 2283

- **source**: Patience's hourglass emptied.
- **target**: ~~Patience's hourglass emptied.~~ **The audience grew restless.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Patience's hourglass emptied.~~ **The audience''''The rest of the the the audience''''' of the the rest of of the of the the audience'''' interest in**
- `ef32` [exact ✗ · FRR ✓]: ~~Patience's hourglass emptied.~~ **The audience''''The rest of the the the audience''''' of the the rest of of the of the the audience'''' interest in**
- `lingualens` [exact ✗ · FRR —]: ~~Patience's hourglass emptied.~~ **The silence grew.**
- `steer` [exact ✗ · FRR ✓]: ~~Patience's hourglass emptied.~~ **The crowd grew restless.**

## middle_verb

### success — idx 2328

- **source**: This wood carves smoothly with sharp tools.
- **target**: This wood ~~carves~~ **is carved** smoothly with sharp tools.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: This wood ~~carves~~ **is carved** smoothly with sharp tools.
- `ef32` [exact ✗ · FRR ✓]: This wood ~~carves smoothly with sharp tools.~~ **is is is is a is a is a very is a very a very a smoothly.**
- `lingualens` [exact ✗ · FRR —]: This wood carves ~~smoothly~~ with sharp tools.
- `steer` [exact ✓ · FRR ✓]: This wood ~~carves~~ **is carved** smoothly with sharp tools.


### near — idx 2304

- **source**: Fresh bread smells wonderful when baking.
- **target**: Fresh bread ~~smells~~ **is smelled as** wonderful when baking.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Fresh bread ~~smells~~ **is is as is a is as a as is a a as a is a as baked as as a as fresh as as a as as** wonderful when baking.
- `ef32` [exact ✗ · FRR ✓]: Fresh bread ~~smells~~ **is is as is a is as a as is a a as a is a as baked as as a as fresh as as a as as** wonderful when baking.
- `lingualens` [exact ✗ · FRR —]: ~~Fresh~~ **Freshly baked** bread smells ~~wonderful when baking.~~ **wonderful.**
- `steer` [exact ✗ · FRR ✓]: Fresh bread smells wonderful when **it is** baking.


### fail — idx 2331

- **source**: Leather jackets wear comfortably in cool weather.
- **target**: Leather jackets ~~wear~~ **are worn** comfortably in cool weather.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: Leather jackets ~~wear~~ **are are are are are are used** comfortably in cool weather.
- `ef32` [exact ✗ · FRR ✗]: Leather jackets ~~wear~~ **are are are are are are used** comfortably in cool weather.
- `lingualens` [exact ✗ · FRR —]: ~~Leather jackets wear comfortably in cool weather.~~ **```python**
- `steer` [exact ✗ · FRR ✓]: Leather jackets ~~wear comfortably~~ **are comfortable** in cool weather.

## negation_prefix

### success — idx 2356

- **source**: He is an inexperienced driver.
- **target**: He is an ~~inexperienced~~ **experienced** driver.

- `ours` via ef32 [exact ✓ · FRR —] ←focus: He is an ~~inexperienced~~ **experienced** driver.
- `ef32` [exact ✓ · FRR —]: He is an ~~inexperienced~~ **experienced** driver.
- `lingualens` [exact ✗ · FRR —]: He is ~~an inexperienced driver.~~ **a seasoned professional in the field of software development.**
- `steer` [exact ✗ · FRR —]: He is ~~an inexperienced~~ **a seasoned** driver.


### fail — idx 2381

- **source**: The team is disorganized.
- **target**: The team is ~~disorganized.~~ **organized.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The team is ~~disorganized.~~ **organized in by the a by team a in the by the the the a team order of by the the a of the the the to a**
- `ef32` [exact ✗ · FRR ✗]: The team is ~~disorganized.~~ **organized in by the a by team a in the by the the the a team order of by the the a of the the the to a**
- `lingualens` [exact ✗ · FRR —]: The team is ~~disorganized.~~ ****organizzate****
- `steer` [exact ✗ · FRR ✗]: The team is ~~disorganized.~~ **unorganized.**

## nominal_adverbials

### near — idx 2420

- **source**: She studied abroad last semester.
- **target**: She studied ~~abroad~~ **in a foreign country** last semester.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She studied ~~abroad~~ **a a school foreign country country a school country foreign country school or in a a school country school in a a school foreign country a country** last semester.
- `ef32` [exact ✗ · FRR ✓]: She studied ~~abroad~~ **a a school foreign country country a school country foreign country school or in a a school country school in a a school foreign country a country** last semester.
- `lingualens` [exact ✗ · FRR —]: She studied ~~abroad last semester.~~ **abroad.**
- `steer` [exact ✗ · FRR ✓]: She studied ~~abroad~~ **foreign country** last semester.


### fail — idx 2413

- **source**: He slept late on Saturday morning.
- **target**: He slept **until** late on Saturday morning.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~He slept late on Saturday morning.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✗]: He ~~slept late~~ **slept, until the very late,, he, and,, to, ( to he has, has not, and and, and has** on Saturday ~~morning.~~ **morning,.**
- `lingualens` [exact ✗ · FRR —]: He slept ~~late on Saturday morning.~~ **until Saturday.**
- `steer` [exact ✗ · FRR ✗]: ~~He slept late on Saturday morning.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## nominal_suffix

### success — idx 2462

- **source**: She showed great kindness to strangers.
- **target**: She showed great ~~kindness~~ **kind** to strangers.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She showed great ~~kindness~~ **kind** to strangers.
- `ef32` [exact ✓ · FRR ✓]: She showed great ~~kindness~~ **kind** to strangers.
- `lingualens` [exact ✗ · FRR —]: She showed great kindness to ~~strangers.~~ **all.**
- `steer` [exact ✗ · FRR ✗]: She ~~showed~~ **exhibited** great kindness to strangers.


### near — idx 2456

- **source**: The assistant handed me the documents.
- **target**: The ~~assistant~~ **assist** handed me the documents.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~assistant~~ **pass** handed me the documents.
- `ef32` [exact ✗ · FRR ✓]: The ~~assistant~~ **pass** handed me the documents.
- `lingualens` [exact ✗ · FRR —]: The ~~assistant handed~~ **assistan passed** me the documents.
- `steer` [exact ✗ · FRR ✗]: The ~~assistant~~ **documents were** handed **to** me **by** the ~~documents.~~ **assistant.**


### fail — idx 2495

- **source**: She majors in linguistics.
- **target**: She majors in ~~linguistics.~~ **lingual.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: She ~~majors in linguistics.~~ **is a linguist.**
- `steer` [exact ✗ · FRR ✗]: She ~~majors~~ **is majoring** in linguistics.

## non_defining_relative_clauses

### near — idx 2509

- **source**: The beach, where we met, holds special memories.
- **target**: The ~~beach, where we met,~~ **beach** holds special memories. **We met there.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~The beach, where we met, holds special memories.~~ **The.. They'rere here.........re.. There.. There...'tt..'ts**
- `ef32` [exact ✗ · FRR ✓]: ~~The beach, where we met, holds special memories.~~ **The.. They'rere here.........re.. There.. There...'tt..'ts**
- `lingualens` [exact ✗ · FRR —]: ~~The beach, where we met, holds special memories.~~ **We met there.**
- `steer` [exact ✗ · FRR ✓]: The ~~beach,~~ **beach** where we ~~met, holds special memories.~~ **met.**

## non_synecdoche_metonymy

### success — idx 2562

- **source**: The teacher graded forty minds.
- **target**: The teacher graded forty ~~minds.~~ **papers.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The teacher graded forty ~~minds.~~ **papers.**
- `ef32` [exact ✓ · FRR ✓]: The teacher graded forty ~~minds.~~ **papers.**
- `lingualens` [exact ✗ · FRR —]: The teacher graded ~~forty~~ **forty-two** minds.
- `steer` [exact ✗ · FRR ✓]: The teacher graded forty ~~minds.~~ **student papers.**

## noun_clauses

### success — idx 2645

- **source**: I doubt that he will apologize.
- **target**: I doubt ~~that he will apologize.~~ **his apology.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: I doubt ~~that he will apologize.~~ **his apology.**
- `ef32` [exact ✓ · FRR ✓]: I doubt ~~that he will apologize.~~ **his apology.**
- `lingualens` [exact ✗ · FRR —]: I doubt that he will ~~apologize.~~ **be sorry.**
- `steer` [exact ✗ · FRR ✓]: I doubt ~~that he~~ **his** will **to** apologize.


### near — idx 2636

- **source**: They discussed what strategies to adopt.
- **target**: They discussed ~~what strategies to adopt.~~ **possible strategies.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They discussed ~~what strategies to adopt.~~ **possible possibilities.**
- `ef32` [exact ✗ · FRR ✓]: They discussed ~~what strategies to adopt.~~ **possible possibilities.**
- `lingualens` [exact ✗ · FRR —]: They discussed ~~what strategies to adopt.~~ **the possibilities.**
- `steer` [exact ✗ · FRR ✓]: They discussed ~~what strategies to adopt.~~ **potential strategies.**

## noun_plural

### success — idx 2686

- **source**: The wishes were granted.
- **target**: The ~~wishes~~ **wish** were granted.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The ~~wishes~~ **wish** were granted.
- `ef32` [exact ✓ · FRR ✓]: The ~~wishes~~ **wish** were granted.
- `lingualens` [exact ✗ · FRR —]: The ~~wishes were~~ **wish was** granted.
- `steer` [exact ✗ · FRR ✗]: The ~~wishes~~ **desires** were granted.


### near — idx 2675

- **source**: The songs are popular.
- **target**: The ~~songs~~ **song** are popular.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~songs are~~ **song is** popular.
- `ef32` [exact ✗ · FRR ✓]: The ~~songs are~~ **song is** popular.
- `lingualens` [exact ✗ · FRR —]: ~~The songs are popular.~~ **```python**
- `steer` [exact ✗ · FRR ✓]: The ~~songs are~~ **song is** popular.

## object_expletives

### near — idx 2720

- **source**: They made it easy to understand the instructions.
- **target**: They ~~made it easy to understand~~ **simplified** the ~~instructions.~~ **instructions for understanding.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~They made it easy to understand~~ **For the purposes of this simplify the the instruction of of for the the of this the the interpretation of of** the instructions.
- `ef32` [exact ✗ · FRR ✓]: ~~They made it easy to understand~~ **For the purposes of this simplify the the instruction of of for the the of this the the interpretation of of** the instructions.
- `lingualens` [exact ✗ · FRR —]: They made it ~~easy~~ **easier** to ~~understand~~ the instructions.
- `steer` [exact ✗ · FRR ✓]: They ~~made it easy to understand~~ **simplified** the instructions.

## of_genitive

### success — idx 2756

- **source**: He lost the key of his.
- **target**: He lost ~~the key of his.~~ **his key.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: He lost ~~the key of his.~~ **his key.**
- `ef32` [exact ✗ · FRR ✓]: He lost ~~the key of his.~~ **his eye.**
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✓ · FRR ✓]: He lost ~~the key of his.~~ **his key.**


### near — idx 2765

- **source**: I discovered the secret of theirs.
- **target**: I discovered ~~the secret of theirs.~~ **their secret.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I discovered ~~the secret of theirs.~~ **their place.**
- `ef32` [exact ✗ · FRR ✓]: I discovered ~~the secret of theirs.~~ **their place.**
- `lingualens` [exact ✓ · FRR —]: I discovered ~~the secret of theirs.~~ **their secret.**
- `steer` [exact ✗ · FRR ✓]: ~~I~~ **They** discovered ~~the secret of theirs.~~ **their secret.**

## optative

### success — idx 2822

- **source**: Would that fire spread quickly!
- **target**: ~~Would that fire spread quickly!~~ **Fire spreads quickly.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Would that fire spread quickly!~~ **Fire spreads quickly.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would that~~ fire ~~spread quickly!~~ **spread,, and is fire and and is,,,, and and fire and and,, and fire and and, and is in!.**
- `lingualens` [exact ✗ · FRR —]: ~~Would that fire spread quickly!~~ **```**
- `steer` [exact ✓ · FRR ✓]: ~~Would that fire spread quickly!~~ **Fire spreads quickly.**

## passive_voice

### success — idx 2872

- **source**: The news is being broadcast by the radio station.
- **target**: The ~~news~~ **radio station** is ~~being broadcast by~~ **broadcasting** the ~~radio station.~~ **news.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: The ~~news~~ **radio station** is ~~being broadcast by~~ **broadcasting** the ~~radio station.~~ **news.**
- `ef32` [exact ✗ · FRR ✓]: The news is ~~being broadcast by~~ **also also available on online at on the website online on at on the** the radio ~~station.~~ **the website Internet website station on the the the Internet,, which broadcasts broadcasts operates.**
- `lingualens` [exact ✗ · FRR —]: The news is being ~~broadcast~~ **broadcasted** by the radio station.
- `steer` [exact ✓ · FRR ✓]: The ~~news~~ **radio station** is ~~being broadcast by~~ **broadcasting** the ~~radio station.~~ **news.**


### near — idx 2867

- **source**: The software has been updated by the technician.
- **target**: The ~~software~~ **technician** has ~~been~~ updated ~~by~~ the ~~technician.~~ **software.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: The ~~software has been~~ **technician** updated ~~by~~ the ~~technician.~~ **software.**
- `ef32` [exact ✗ · FRR ✓]: The ~~software has been~~ **phone is phone phone** updated ~~by~~ **and and with and and-levellevel phone with phone and** the ~~technician.~~ **phone is phone and is phone phone really really is phone really with.**
- `lingualens` [exact ✗ · FRR —]: The ~~software has been~~ **technician** updated ~~by~~ the ~~technician.~~ **software.**
- `steer` [exact ✗ · FRR ✓]: The ~~software has been~~ **technician** updated ~~by~~ the ~~technician.~~ **software.**

## past

### success — idx 2922

- **source**: Lions hunted in groups.
- **target**: Lions ~~hunted~~ **hunt** in groups.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Lions ~~hunted~~ **hunt** in groups.
- `ef32` [exact ✓ · FRR ✓]: Lions ~~hunted~~ **hunt** in groups.
- `lingualens` [exact ✗ · FRR —]: ~~Lions hunted in groups.~~ **```**
- `steer` [exact ✓ · FRR ✓]: Lions ~~hunted~~ **hunt** in groups.


### near — idx 2945

- **source**: They rehearsed the play daily.
- **target**: They ~~rehearsed~~ **rehearse** the play daily.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They ~~rehearsed~~ **rehehearsed** the play daily.
- `ef32` [exact ✗ · FRR ✓]: They ~~rehearsed~~ **rehehearsed** the play daily.
- `lingualens` [exact ✓ · FRR —]: They ~~rehearsed~~ **rehearse** the play daily.
- `steer` [exact ✓ · FRR ✓]: They ~~rehearsed~~ **rehearse** the play daily.


### fail — idx 2915

- **source**: Rivers flowed to the sea.
- **target**: Rivers ~~flowed~~ **flow** to the sea.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: ~~Rivers flowed to the sea.~~ **\upharryDemografie**
- `steer` [exact ✗ · FRR ✓]: ~~Rivers flowed~~ **Water flows** to the sea.

## past_participle

### success — idx 2986

- **source**: He had brushed his teeth.
- **target**: He had ~~brushed~~ **brush** his teeth.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He had ~~brushed~~ **brush** his teeth.
- `ef32` [exact ✓ · FRR ✓]: He had ~~brushed~~ **brush** his teeth.
- `lingualens` [exact ✗ · FRR —]: He ~~had~~ brushed his teeth.
- `steer` [exact ✗ · FRR ✓]: He ~~had~~ brushed his teeth.


### near — idx 2994

- **source**: She had folded the clothes.
- **target**: She had ~~folded~~ **fold** the clothes.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~had folded~~ **has fold** the clothes.
- `ef32` [exact ✗ · FRR ✓]: She ~~had folded~~ **has fold** the clothes.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: She ~~had~~ folded the clothes.

## past_participle_irregular

### near — idx 3037

- **source**: The house has been built.
- **target**: The house has ~~been built.~~ **be build.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The ~~house has been~~ **houses be be** built.
- `ef32` [exact ✗ · FRR ✓]: The ~~house has been~~ **houses be be** built.
- `lingualens` [exact ✗ · FRR —]: The house ~~has been~~ **is** built.
- `steer` [exact ✗ · FRR ✗]: The house ~~has been~~ **was** built.


### fail — idx 3018

- **source**: The meeting has been held.
- **target**: The meeting has been ~~held.~~ **hold.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The ~~meeting has~~ **meetings have** been held.
- `ef32` [exact ✗ · FRR ✗]: The ~~meeting has~~ **meetings have** been held.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: The meeting has ~~been held.~~ **taken place.**

## past_perfect

### success — idx 3077

- **source**: It had flown south.
- **target**: It ~~had flown~~ **flies** south.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: It ~~had flown~~ **flies** south.
- `ef32` [exact ✓ · FRR ✓]: It ~~had flown~~ **flies** south.
- `lingualens` [exact ✗ · FRR —]: It ~~had flown~~ **flew** south.
- `steer` [exact ✓ · FRR ✓]: It ~~had flown~~ **flies** south.


### near — idx 3070

- **source**: I had opened the window.
- **target**: I ~~had opened~~ **open** the window.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: I ~~had opened~~ **start** the window.
- `ef32` [exact ✗ · FRR ✓]: I ~~had opened~~ **start** the window.
- `lingualens` [exact ✗ · FRR —]: I ~~had opened~~ **have open** the window.
- `steer` [exact ✓ · FRR ✓]: I ~~had opened~~ **open** the window.

## past_progressive

### near — idx 3120

- **source**: We were compressing files.
- **target**: We ~~were compressing~~ **compress** files.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~were compressing files.~~
- `ef32` [exact ✗ · FRR ✓]: We ~~were compressing files.~~
- `lingualens` [exact ✓ · FRR —]: We ~~were compressing~~ **compress** files.
- `steer` [exact ✗ · FRR ✓]: We ~~were compressing~~ **compressed** files.

## past_tense

### success — idx 3168

- **source**: We agreed with the decision.
- **target**: We ~~agreed~~ **agree** with the decision.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: We ~~agreed~~ **agree** with the decision.
- `ef32` [exact ✓ · FRR ✓]: We ~~agreed~~ **agree** with the decision.
- `lingualens` [exact ✗ · FRR —]: We ~~agreed with the decision.~~ **agree.**
- `steer` [exact ✗ · FRR ✓]: We ~~agreed~~ **concur** with the decision.


### near — idx 3180

- **source**: We continued the discussion.
- **target**: We ~~continued~~ **continue** the discussion.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: We ~~continued~~ **will discuss** the discussion.
- `ef32` [exact ✗ · FRR ✓]: We ~~continued~~ **will discuss** the discussion.
- `lingualens` [exact ✓ · FRR —]: We ~~continued~~ **continue** the discussion.
- `steer` [exact ✗ · FRR ✗]: We ~~continued~~ **resumed** the discussion.

## past_tense_irregular

### success — idx 3219

- **source**: She swam in the pool all afternoon.
- **target**: She ~~swam~~ **swim** in the pool all afternoon.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~swam~~ **swim** in the pool all afternoon.
- `ef32` [exact ✓ · FRR ✓]: She ~~swam~~ **swim** in the pool all afternoon.
- `lingualens` [exact ✗ · FRR —]: She ~~swam~~ **swims** in the pool all afternoon.
- `steer` [exact ✗ · FRR ✗]: She swam in the pool all ~~afternoon.~~ **day.**


### near — idx 3204

- **source**: He took the book from the shelf.
- **target**: He ~~took~~ **take** the book from the shelf.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~took~~ **takes** the book from the shelf.
- `ef32` [exact ✗ · FRR ✓]: He ~~took~~ **takes** the book from the shelf.
- `lingualens` [exact ✗ · FRR —]: He ~~took~~ **takes** the book from the shelf.
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### fail — idx 3205

- **source**: The children built a sandcastle at the beach.
- **target**: The children ~~built~~ **build** a sandcastle at the beach.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✓ · FRR —]: The children ~~built~~ **build** a sandcastle at the beach.
- `steer` [exact ✗ · FRR ✓]: The children ~~built~~ **construct** a sandcastle at the beach.

## personification

### near — idx 3327

- **source**: Hope quietly stirred.
- **target**: ~~Hope quietly stirred.~~ **People felt hopeful.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Hope quietly stirred.~~ **people excited to people's feelingre to of people for people''s.**
- `ef32` [exact ✗ · FRR ✓]: ~~Hope quietly stirred.~~ **people excited to people's feelingre to of people for people''s.**
- `lingualens` [exact ✗ · FRR —]: ~~Hope quietly stirred.~~ **The world seemed to be a little bit more hopeful.**
- `steer` [exact ✗ · FRR ✓]: Hope ~~quietly stirred.~~ **felt hopeful.**

## politeness

### success — idx 3390

- **source**: Would you kindly stay a little longer?
- **target**: ~~Would you kindly stay~~ **Stay** a little ~~longer?~~ **longer.**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Would you kindly stay~~ **Stay** a little ~~longer?~~ **longer.**
- `ef32` [exact ✗ · FRR ✓]: ~~Would you kindly stay~~ **Stay and stop** a ~~little longer?~~ **little.**
- `lingualens` [exact ✗ · FRR —]: ~~Would you kindly stay a little longer?~~ **```**
- `steer` [exact ✓ · FRR ✓]: ~~Would you kindly stay~~ **Stay** a little ~~longer?~~ **longer.**


### near — idx 3355

- **source**: Could you please explain that again?
- **target**: ~~Could you please explain~~ **Explain** that ~~again?~~ **again.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Could~~ **Tell** you ~~please explain~~ that ~~again?~~ **again.**
- `ef32` [exact ✗ · FRR ✓]: ~~Could you please explain~~ **What means,, to you, me is** that ~~again?~~ **that to me.**
- `lingualens` [exact ✗ · FRR —]: ~~Could you please explain that again?~~ **```**
- `steer` [exact ✗ · FRR ✓]: ~~Could~~ **Tell** you ~~please explain~~ that ~~again?~~ **again.**

## possessive_form

### success — idx 3428

- **source**: They studied the philosopher's writings.
- **target**: They studied the ~~philosopher's~~ **philosopher** writings.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: They studied the ~~philosopher's~~ **philosopher** writings.
- `ef32` [exact ✓ · FRR ✓]: They studied the ~~philosopher's~~ **philosopher** writings.
- `lingualens` [exact ✗ · FRR —]: They studied the ~~philosopher's writings.~~ **writings of the philosopher.**
- `steer` [exact ✗ · FRR ✓]: ~~They studied the philosopher's writings.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 3413

- **source**: He repaired the car's broken mirror.
- **target**: He repaired the ~~car's~~ **car** broken mirror.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He repaired the ~~car's broken~~ **car** mirror.
- `ef32` [exact ✗ · FRR ✓]: He repaired the ~~car's broken~~ **car** mirror.
- `lingualens` [exact ✗ · FRR —]: He repaired the ~~car's broken mirror.~~ **mirror of the car.**
- `steer` [exact ✗ · FRR ✓]: ~~He repaired the car's broken mirror.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## present_participle

### near — idx 3483

- **source**: He is practicing the guitar.
- **target**: He ~~is practicing~~ **practices** the guitar.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: He ~~is practicing~~ **plays** the guitar.
- `ef32` [exact ✗ · FRR ✓]: He ~~is practicing~~ **plays** the guitar.
- `lingualens` [exact ✗ · FRR —]: He is ~~practicing the guitar.~~ **a precise text editor.**
- `steer` [exact ✓ · FRR ✓]: He ~~is practicing~~ **practices** the guitar.

## present_perfect

### success — idx 3542

- **source**: Have you fed the stray cats?
- **target**: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?
- `ef32` [exact ✓ · FRR ✓]: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?
- `lingualens` [exact ✓ · FRR —]: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?
- `steer` [exact ✓ · FRR ✓]: ~~Have~~ **Do** you ~~fed~~ **feed** the stray cats?


### near — idx 3504

- **source**: The dog has barked at strangers.
- **target**: The dog ~~has barked~~ **barks** at strangers.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The dog ~~has barked~~ at strangers.
- `ef32` [exact ✗ · FRR ✓]: The dog ~~has barked~~ at strangers.
- `lingualens` [exact ✓ · FRR —]: The dog ~~has barked~~ **barks** at strangers.
- `steer` [exact ✓ · FRR ✓]: The dog ~~has barked~~ **barks** at strangers.

## present_progressive

### success — idx 3598

- **source**: Is magma solidifying underground?
- **target**: ~~Is~~ **Does** magma ~~solidifying~~ **solidify** underground?

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~Is~~ **Does** magma ~~solidifying~~ **solidify** underground?
- `ef32` [exact ✗ · FRR ✗]: ~~Is magma solidifying~~ **Does magmaize** underground?
- `lingualens` [exact ✗ · FRR —]: ~~Is magma solidifying underground?~~ **```**
- `steer` [exact ✓ · FRR ✓]: ~~Is~~ **Does** magma ~~solidifying~~ **solidify** underground?


### near — idx 3560

- **source**: Waves are crashing ashore.
- **target**: Waves ~~are crashing~~ **crash** ashore.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Waves are crashing~~ **wave close** ashore.
- `ef32` [exact ✗ · FRR ✓]: ~~Waves are crashing~~ **wave close** ashore.
- `lingualens` [exact ✗ · FRR —]: ~~Waves are crashing~~ **Wave crash** ashore.
- `steer` [exact ✗ · FRR ✓]: Waves ~~are crashing~~ **crashed** ashore.

## punctual_durative

### success — idx 3604

- **source**: He shivers nonstop.
- **target**: He ~~shivers nonstop.~~ **shivers.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He ~~shivers nonstop.~~ **shivers.**
- `ef32` [exact ✓ · FRR ✓]: He ~~shivers nonstop.~~ **shivers.**
- `lingualens` [exact ✗ · FRR —]: He ~~shivers nonstop.~~ ****shivers**.**
- `steer` [exact ✗ · FRR ✓]: ~~He shivers nonstop.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**


### near — idx 3637

- **source**: She sits.
- **target**: She ~~sits.~~ **sits down.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~She sits.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✓]: She ~~sits.~~ **has sits down.**
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: ~~She sits.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 3611

- **source**: A clock ticks without pause.
- **target**: A clock ~~ticks without pause.~~ **ticks.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: ~~A clock ticks without pause.~~ **```python**
- `steer` [exact ✗ · FRR ✓]: ~~A clock ticks without pause.~~ **rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan rowspan**

## quantifier

### near — idx 3696

- **source**: There is any evidence to support that.
- **target**: ~~There is any evidence to support~~ **Evidence supports** that.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~There is any evidence to support that.~~ **Evidence**
- `ef32` [exact ✗ · FRR ✓]: ~~There is any evidence to support that.~~ **Evidence**
- `lingualens` [exact ✗ · FRR —]: ~~There is any evidence to support that.~~ **```**
- `steer` [exact ✓ · FRR ✓]: ~~There is any evidence to support~~ **Evidence supports** that.


### fail — idx 3685

- **source**: Did you take many photos on vacation?
- **target**: Did you take ~~many~~ photos on vacation?

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: ~~Did you take many photos on vacation?~~ **```python**
- `steer` [exact ✗ · FRR ✓]: ~~Did you take many photos on vacation?~~ **rowspan rowspan**

## quantitative_prefix

### fail — idx 3740

- **source**: The factory produces kilowatt machines.
- **target**: The factory produces ~~kilowatt machines.~~ **machines rated at one thousand watts.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The factory produces ~~kilowatt~~ **kilowatt-horsepower** machines.
- `ef32` [exact ✗ · FRR ✗]: The factory produces **at one one one million hundred thousand million one million million a year at hundred million thousand thousand** kilowatt ~~machines.~~ **per of a a one million hour machines per at.**
- `lingualens` [exact ✗ · FRR —]: The factory produces kilowatt ~~machines.~~ **of power.**
- `steer` [exact ✗ · FRR ✗]: The factory produces ~~kilowatt~~ **kilowatt-horsepower** machines.

## referring

### success — idx 3762

- **source**: These poems touch hearts.
- **target**: ~~These poems~~ **Poems** touch hearts.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~These poems~~ **Poems** touch hearts.
- `ef32` [exact ✓ · FRR ✓]: ~~These poems~~ **Poems** touch hearts.
- `lingualens` [exact ✗ · FRR —]: ~~These poems touch hearts.~~ **```python**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*


### near — idx 3781

- **source**: His stories teach lessons.
- **target**: ~~His stories~~ **Stories** teach lessons.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~His stories teach~~ **News teaches** lessons.
- `ef32` [exact ✗ · FRR ✓]: ~~His stories teach~~ **News teaches** lessons.
- `lingualens` [exact ✗ · FRR —]: ~~His stories teach lessons.~~ **```**
- `steer` [exact ✗ · FRR ✓]: ~~His stories teach lessons.~~ **Stories by which lessons are taught.**

## relative_clauses

### success — idx 3848

- **source**: The teacher who inspired me retired last year.
- **target**: ~~The~~ **My inspiring** teacher ~~who inspired me~~ retired last year.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~The~~ **My inspiring** teacher ~~who inspired me~~ retired last year.
- `ef32` [exact ✗ · FRR ✓]: ~~The teacher who inspired me retired~~ **My own family life life, beautiful and beautiful beautiful beautiful beautiful beautiful inspiring beautiful family, my own family own family family teacher, my family own family family** last year.
- `lingualens` [exact ✗ · FRR —]: ~~The teacher who inspired me retired~~ **My teacher, who,** last ~~year.~~ **year, retired.**
- `steer` [exact ✓ · FRR ✓]: ~~The~~ **My inspiring** teacher ~~who inspired me~~ retired last year.


### near — idx 3802

- **source**: The man who called me is my uncle.
- **target**: The man ~~who called~~ **calling** me is my uncle.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The man ~~who called~~ **says** me is my uncle.
- `ef32` [exact ✗ · FRR ✓]: The man ~~who called~~ **says** me is my uncle.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✓ · FRR ✓]: The man ~~who called~~ **calling** me is my uncle.


### fail — idx 3832

- **source**: The garden which Sarah tends is beautiful.
- **target**: ~~The garden which~~ Sarah tends **the garden and it** is beautiful.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The garden ~~which~~ **that** Sarah tends is beautiful.
- `ef32` [exact ✗ · FRR ✓]: ~~The garden which Sarah tends~~ is ~~beautiful.~~ **a very very a very good a a good a very very a good very a a good and a a beautiful beautiful and a a a and a and a**
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: The garden ~~which~~ **that** Sarah tends is beautiful.

## representative

### near — idx 3889

- **source**: Researchers confirm otters use tools to eat.
- **target**: ~~Researchers confirm otters~~ **Otters** use tools to eat.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Researchers confirm otters use tools to eat.~~ **In**
- `ef32` [exact ✗ · FRR ✓]: ~~Researchers confirm otters use tools to eat.~~ **In**
- `lingualens` [exact ✗ · FRR —]: ~~Researchers confirm otters use tools to eat.~~ **```python**
- `steer` [exact ✗ · FRR ✓]: Researchers confirm otters ~~use tools to eat.~~ **используют инструменты для ЕATING.**

## resultative

### near — idx 3935

- **source**: He cut the rope free.
- **target**: He cut the **rope. The** rope **was** free.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~He cut the rope free.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**
- `ef32` [exact ✗ · FRR ✓]: He **has** cut the **very** rope free.
- `lingualens` [exact ✗ · FRR —]: He cut the ~~rope free.~~ **rope.**
- `steer` [exact ✗ · FRR ✓]: ~~He cut the rope free.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**


### fail — idx 3916

- **source**: She tied the knot tight.
- **target**: She tied the **knot. The** knot **was** tight.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: She tied the ~~knot tight.~~ **knot.**
- `steer` [exact ✗ · FRR ✓]: ~~She tied the knot tight.~~ **" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "**

## s_genitive

### success — idx 4407

- **source**: Laura’s presentation impressed everyone.
- **target**: ~~Laura’s~~ **Her** presentation impressed everyone.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Laura’s~~ **Her** presentation impressed everyone.
- `ef32` [exact ✓ · FRR ✓]: ~~Laura’s~~ **Her** presentation impressed everyone.
- `lingualens` [exact ✓ · FRR —]: ~~Laura’s~~ **Her** presentation impressed everyone.
- `steer` [exact ✗ · FRR ✓]: ~~Laura’s~~ **Her** presentation impressed everyone. **.**


### near — idx 4416

- **source**: Harry’s birthday is tomorrow.
- **target**: ~~Harry’s~~ **His** birthday is tomorrow.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~Harry’s~~ **His** birthday is tomorrow. **. . . . .**
- `ef32` [exact ✗ · FRR ✓]: ~~Harry’s birthday~~ **His own father's** is tomorrow.
- `lingualens` [exact ✓ · FRR —]: ~~Harry’s~~ **His** birthday is tomorrow.
- `steer` [exact ✗ · FRR ✓]: ~~Harry’s~~ **His** birthday is tomorrow. **. . . . .**

## spatial_or_directional_prefix

### success — idx 4041

- **source**: The event was an extra-ordinary success.
- **target**: The event was an ~~extra-ordinary~~ **exceptional** success.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The event was an ~~extra-ordinary~~ **exceptional** success.
- `ef32` [exact ✓ · FRR ✓]: The event was an ~~extra-ordinary~~ **exceptional** success.
- `lingualens` [exact ✗ · FRR —]: The event was ~~an extra-ordinary~~ **a** success.
- `steer` [exact ✗ · FRR ✗]: The event was an ~~extra-ordinary~~ **extraordinary** success.


### near — idx 4038

- **source**: He demonstrated subatomic particle movement.
- **target**: He demonstrated ~~subatomic particle movement.~~ **movement of particles smaller than atoms.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: He demonstrated ~~subatomic particle movement.~~ **particles smaller than atoms.**
- `ef32` [exact ✗ · FRR ✓]: He demonstrated ~~subatomic particle movement.~~ **the small the of small than than size of of of smaller than than than than the than of the the than of of the the the small than movement of of of.**
- `lingualens` [exact ✗ · FRR —]: He demonstrated ~~subatomic particle movement.~~ **the movement of atoms.**
- `steer` [exact ✗ · FRR ✓]: He demonstrated ~~subatomic particle movement.~~ **particles smaller than atoms.**


### fail — idx 4040

- **source**: The scientist examined intrauterine growth.
- **target**: The scientist examined ~~intrauterine growth.~~ **growth inside the uterus.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The scientist examined intrauterine ~~growth.~~ **growth the the the the the the the the the the the the the the the the the the the the the the the the the the the the the.**
- `ef32` [exact ✗ · FRR ✗]: The scientist examined intrauterine ~~growth.~~ **growth the the the the the the the the the the the the the the the the the the the the the the the the the the the the the.**
- `lingualens` [exact ✗ · FRR —]: The scientist examined ~~intrauterine growth.~~ **the uterus.**
- `steer` [exact ✗ · FRR ✓]: The scientist ~~examined intrauterine~~ **studied fetal** growth.

## split_infinitives

### success — idx 4086

- **source**: He had to suddenly adjust the settings to fix the machine.
- **target**: He had to ~~suddenly~~ adjust the settings **suddenly** to fix the machine.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: He had to ~~suddenly~~ adjust the settings **suddenly** to fix the machine.
- `ef32` [exact ✗ · FRR ✓]: He had to ~~suddenly~~ adjust the settings **automatically** to fix the machine.
- `lingualens` [exact ✗ · FRR —]: He had to ~~suddenly~~ adjust the settings to fix the machine.
- `steer` [exact ✓ · FRR ✓]: He had to ~~suddenly~~ adjust the settings **suddenly** to fix the machine.


### near — idx 4079

- **source**: They plan to more effectively communicate with their team members.
- **target**: They plan to ~~more effectively~~ communicate with their team ~~members.~~ **members more effectively.**

- `ours` via steer [exact ✗ · FRR ✓] ←focus: They plan to **communicate** more effectively ~~communicate~~ with their team members.
- `ef32` [exact ✗ · FRR ✓]: They plan to ~~more effectively~~ communicate with their team ~~members.~~ **members regularly and and and with.**
- `lingualens` [exact ✗ · FRR —]: They plan to **communicate** more effectively ~~communicate~~ with their team members.
- `steer` [exact ✗ · FRR ✓]: They plan to **communicate** more effectively ~~communicate~~ with their team members.


### fail — idx 4070

- **source**: He hopes to easily pass the exam with regular study.
- **target**: He hopes to ~~easily~~ pass the exam **easily** with regular study.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: He hopes to easily pass the exam ~~with~~ **through** regular study.
- `ef32` [exact ✗ · FRR ✓]: He hopes to ~~easily~~ pass the exam **only** with regular study.
- `lingualens` [exact ✗ · FRR —]: He hopes to ~~easily~~ pass the exam with regular study.
- `steer` [exact ✗ · FRR ✗]: He hopes to easily pass the exam ~~with~~ **through** regular study.

## static_dynamic

### success — idx 4125

- **source**: A yogi maintains postures.
- **target**: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.
- `ef32` [exact ✓ · FRR ✓]: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.
- `lingualens` [exact ✗ · FRR —]: ~~A yogi maintains postures.~~ **The sentence is already in the form of a statement, not a question.**
- `steer` [exact ✓ · FRR ✓]: ~~A~~ **The** yogi ~~maintains~~ **makes** postures.


### near — idx 4145

- **source**: Ice becomes solid.
- **target**: ~~Ice becomes~~ **The ice remains** solid.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Ice becomes~~ **The 22 22'2''s 2222's'ss remains''s's** solid.
- `ef32` [exact ✗ · FRR ✓]: ~~Ice becomes~~ **The 22 22'2''s 2222's'ss remains''s's** solid.
- `lingualens` [exact ✗ · FRR —]: ~~Ice becomes solid.~~ **The minimal change is to the the word "minimal" is added to the end of the sentence.**
- `steer` [exact ✗ · FRR ✗]: Ice ~~becomes~~ **turns** solid.


### fail — idx 4124

- **source**: An acrobat maintains equilibrium midair.
- **target**: ~~An~~ **The** acrobat ~~maintains equilibrium midair.~~ **balances on beams.**

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~An acrobat maintains equilibrium midair.~~ **The acrobats balance on the air.**
- `ef32` [exact ✗ · FRR ✗]: ~~An~~ **The** acrobat ~~maintains equilibrium midair.~~ **on the the on the on the on on the the the on the on the the on mida the the the on the the on the the on the.**
- `lingualens` [exact ✗ · FRR —]: An acrobat maintains ~~equilibrium midair.~~ **its equilibrium.**
- `steer` [exact ✗ · FRR ✗]: ~~An acrobat maintains equilibrium midair.~~ **The acrobats balance on the air.**

## subject_auxiliary_inversion

### near — idx 4193

- **source**: May you stay healthy and happy!
- **target**: ~~May~~ **I hope** you stay healthy and ~~happy!~~ **happy.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~May you stay~~ **I'''m'mm'mmm' Immm' I Immm' Immmm** healthy and happy!
- `ef32` [exact ✗ · FRR ✓]: ~~May you stay~~ **I'''m'mm'mmm' Immm' I Immm' Immmm** healthy and happy!
- `lingualens` [exact ✗ · FRR —]: ~~May you stay~~ **I hopestay** healthy and happy!
- `steer` [exact ✓ · FRR ✓]: ~~May~~ **I hope** you stay healthy and ~~happy!~~ **happy.**

## subject_verb_inversion

### near — idx 4212

- **source**: There goes his comment.
- **target**: ~~There~~ **His comment** goes ~~his comment.~~ **there.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~There goes~~ **is is also also also his is also also also here, known as as as the as the here,,, is, and** his comment.
- `ef32` [exact ✗ · FRR ✓]: ~~There goes~~ **is is also also also his is also also also here, known as as as the as the here,,, is, and** his comment.
- `lingualens` [exact ✗ · FRR —]: ~~There goes his comment.~~ **It was a minimal change, to make it clear:**
- `steer` [exact ✗ · FRR ✓]: ~~There goes~~ **That was** his comment.


### fail — idx 4233

- **source**: Here comes the mailman with today’s letters.
- **target**: ~~Here~~ **The mailman** comes ~~the mailman~~ **here** with today’s letters.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: Here comes the mailman with ~~today’s~~ **today's** letters.
- `ef32` [exact ✗ · FRR ✓]: ~~Here comes the mailman~~ **The house, and** with **him which they was was, as which they was was here, and came, as as was came here to me with him** today’s letters.
- `lingualens` [exact ✗ · FRR —]: Here ~~comes~~ the mailman with today’s letters.
- `steer` [exact ✗ · FRR ✗]: Here comes the mailman with ~~today’s~~ **today's** letters.

## subjunctive_mood

### success — idx 4287

- **source**: I wish I could paint pictures.
- **target**: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.
- `ef32` [exact ✓ · FRR ✓]: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.
- `lingualens` [exact ✗ · FRR —]: I ~~wish I could paint pictures.~~ **hope que te pueda pintar.**
- `steer` [exact ✓ · FRR ✓]: I ~~wish~~ **hope** I ~~could~~ **can** paint pictures.


### near — idx 4283

- **source**: I wish I had more money.
- **target**: I ~~wish~~ **hope** I ~~had~~ **can get** more money.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: I ~~wish I had~~ **hope to get** more money.
- `ef32` [exact ✗ · FRR ✓]: I ~~wish~~ **hope get** I ~~had~~ **can get get** more money.
- `lingualens` [exact ✗ · FRR —]: **I'm not sure what** I ~~wish I had more money.~~ **can do to make this work.**
- `steer` [exact ✗ · FRR ✓]: I ~~wish I had~~ **hope to get** more money.

## superlative

### success — idx 4337

- **source**: He took the safest route.
- **target**: He took the ~~safest~~ **safe** route.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: He took the ~~safest~~ **safe** route.
- `ef32` [exact ✓ · FRR ✓]: He took the ~~safest~~ **safe** route.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✓]: He took the ~~safest~~ **secure** route.


### near — idx 4323

- **source**: That was the silliest mistake ever.
- **target**: That was the ~~silliest~~ **silly** mistake ever.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: That was the ~~silliest~~ mistake ever.
- `ef32` [exact ✗ · FRR ✓]: That was the ~~silliest~~ mistake ever.
- `lingualens` [exact ✗ · FRR —]: That was the ~~silliest mistake ever.~~ **silly mistake.**
- `steer` [exact ✓ · FRR ✓]: That was the ~~silliest~~ **silly** mistake ever.


### fail — idx 4316

- **source**: He is the most respectful student here.
- **target**: He is the ~~most~~ respectful student here.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: He is the most ~~respectful~~ **esteemed** student here.
- `steer` [exact ✗ · FRR ✓]: ~~He is the most respectful student here.~~ **rowspan rowspan rowspan**

## synecdoche

### near — idx 4362

- **source**: The museum acquired a Renaissance brushstroke.
- **target**: The museum acquired a Renaissance ~~brushstroke.~~ **painting.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The museum acquired a ~~Renaissance brushstroke.~~ **collection painting of the by painting the Museum of painting in of the the of painting of the**
- `ef32` [exact ✗ · FRR ✓]: The museum acquired a ~~Renaissance brushstroke.~~ **collection painting of the by painting the Museum of painting in of the the of painting of the**
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✓ · FRR ✓]: The museum acquired a Renaissance ~~brushstroke.~~ **painting.**


### fail — idx 4382

- **source**: Pilots checked the skies before takeoff.
- **target**: Pilots checked ~~the skies~~ **weather reports** before takeoff.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✓]: ~~Pilots checked the skies~~ **Pilots,, weather,,, checked,, weather and, and weather weather, weather weather and weather weather** before ~~takeoff.~~ **and and and and and takeoff,,.**
- `lingualens` [exact ✗ · FRR —]: ~~Pilots checked the skies before takeoff.~~ **```**
- `steer` [exact ✗ · FRR ✗]: *(copy — no edit)*

## tag_questions

### success — idx 4482

- **source**: You were sleeping, weren’t you?
- **target**: ~~You were sleeping, weren’t you?~~ **Were you sleeping?**

- `ours` via steer [exact ✓ · FRR ✓] ←focus: ~~You were sleeping, weren’t you?~~ **Were you sleeping?**
- `ef32` [exact ✗ · FRR ✓]: ~~You~~ **Did your house have own house or** were ~~sleeping, weren’t you?~~ **a own house?**
- `lingualens` [exact ✗ · FRR —]: You ~~were sleeping, weren’t~~ **were, weren't** you?
- `steer` [exact ✓ · FRR ✓]: ~~You were sleeping, weren’t you?~~ **Were you sleeping?**


### near — idx 4466

- **source**: Julia is singing, isn’t she?
- **target**: **Is** Julia ~~is singing, isn’t she?~~ **singing?**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Julia is singing, isn’t she?~~ **Is Java's name India's?**
- `ef32` [exact ✗ · FRR ✓]: ~~Julia is singing, isn’t she?~~ **Is Java's name India's?**
- `lingualens` [exact ✗ · FRR —]: ~~Julia is singing, isn’t she?~~ **Is he singing?**
- `steer` [exact ✓ · FRR ✓]: **Is** Julia ~~is singing, isn’t she?~~ **singing?**

## telic_atelic

### success — idx 4510

- **source**: Students memorized a formula.
- **target**: Students memorized ~~a formula.~~ **formulas.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: Students memorized ~~a formula.~~ **formulas.**
- `ef32` [exact ✓ · FRR ✓]: Students memorized ~~a formula.~~ **formulas.**
- `lingualens` [exact ✗ · FRR —]: ~~Students~~ **Student** memorized a ~~formula.~~ **formulas.**
- `steer` [exact ✗ · FRR ✗]: Students memorized ~~a~~ **the** formula.


### near — idx 4513

- **source**: Engineers designed a bridge.
- **target**: Engineers designed ~~a bridge.~~ **bridges.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Engineers designed ~~a~~ bridge.
- `ef32` [exact ✗ · FRR ✓]: Engineers designed ~~a~~ bridge.
- `lingualens` [exact ✗ · FRR —]: ~~Engineers designed a bridge.~~ **```python**
- `steer` [exact ✗ · FRR ✓]: Engineers designed ~~a bridge.~~ **the bridges.**


### fail — idx 4533

- **source**: Ice was cracking.
- **target**: Ice ~~was cracking.~~ **cracked.**

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Ice was cracking.~~ **He cracked.**
- `ef32` [exact ✗ · FRR ✗]: ~~Ice was cracking.~~ **He cracked.**
- `lingualens` [exact ✗ · FRR —]: ~~Ice was cracking.~~ **```**
- `steer` [exact ✗ · FRR ✗]: Ice ~~was cracking.~~ **shattered.**

## temporal

### near — idx 4559

- **source**: They will arrive that day.
- **target**: They will arrive ~~that day.~~ **on Monday.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They will arrive ~~that day.~~ **on Monday Monday on March.**
- `ef32` [exact ✗ · FRR ✓]: They will arrive ~~that day.~~ **on Monday Monday on March.**
- `lingualens` [exact ✗ · FRR —]: They will ~~arrive that day.~~ **arrive.**
- `steer` [exact ✓ · FRR ✓]: They will arrive ~~that day.~~ **on Monday.**

## temporal_prefix

### near — idx 4604

- **source**: They held a post-match press conference.
- **target**: They held a ~~post-match~~ press ~~conference.~~ **conference after the match.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: They held ~~a post-match press~~ **the same the same same the same the same the the same the the the same the same the the the the same the same the the same the the same the same last** conference.
- `ef32` [exact ✗ · FRR ✓]: They held ~~a post-match press~~ **the same the same same the same the same the the same the the the same the same the the the the same the same the the same the the same the same last** conference.
- `lingualens` [exact ✗ · FRR —]: They held a ~~post-match press conference.~~ **match.**
- `steer` [exact ✗ · FRR ✓]: They held a ~~post-match~~ **post- match** press conference.


### fail — idx 4648

- **source**: The post-interview reflections were positive.
- **target**: The ~~post-interview~~ reflections **after the interview** were positive.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: The ~~post-interview reflections were positive.~~
- `ef32` [exact ✗ · FRR ✗]: The ~~post-interview reflections were positive.~~
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: The ~~post-interview~~ **post- interview** reflections were positive.

## third_person_singular

### success — idx 4668

- **source**: The river flows gently.
- **target**: The river ~~flows~~ **flow** gently.

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: The river ~~flows~~ **flow** gently.
- `ef32` [exact ✓ · FRR ✓]: The river ~~flows~~ **flow** gently.
- `lingualens` [exact ✗ · FRR —]: The river ~~flows gently.~~ **flow is gentle.**
- `steer` [exact ✗ · FRR ✓]: The river ~~flows~~ **flowed** gently.


### near — idx 4669

- **source**: She enjoys reading novels.
- **target**: She ~~enjoys~~ **enjoy** reading novels.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~enjoys reading~~ **read** novels.
- `ef32` [exact ✗ · FRR ✓]: She ~~enjoys reading~~ **read** novels.
- `lingualens` [exact ✗ · FRR —]: She enjoys ~~reading novels.~~ **reading.**
- `steer` [exact ✗ · FRR ✗]: She ~~enjoys~~ **likes** reading novels.


### fail — idx 4665

- **source**: He plays guitar beautifully.
- **target**: He ~~plays~~ **play** guitar beautifully.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: *(copy — no edit)*
- `ef32` [exact ✗ · FRR ✗]: *(copy — no edit)*
- `lingualens` [exact ✗ · FRR —]: He plays **the** guitar beautifully.
- `steer` [exact ✗ · FRR ✗]: He plays **the** guitar beautifully.

## transitional

### success — idx 4709

- **source**: He apologized sincerely; nonetheless, she refused forgiveness.
- **target**: He apologized ~~sincerely; nonetheless, she~~ **sincerely. She** refused forgiveness.

- `ours` via steer [exact ✓ · FRR ✓] ←focus: He apologized ~~sincerely; nonetheless, she~~ **sincerely. She** refused forgiveness.
- `ef32` [exact ✗ · FRR ✓]: He ~~apologized sincerely; nonetheless, she refused forgiveness.~~ **apologized. She said that**
- `lingualens` [exact ✗ · FRR —]: He ~~apologized sincerely; nonetheless,~~ **apologized, and yet again,** she ~~refused forgiveness.~~ **refused.**
- `steer` [exact ✓ · FRR ✓]: He apologized ~~sincerely; nonetheless, she~~ **sincerely. She** refused forgiveness.


### near — idx 4718

- **source**: Children laughed joyfully while parents looked exhausted.
- **target**: Children laughed ~~joyfully while parents~~ **joyfully. Parents** looked exhausted.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: Children laughed joyfully ~~while~~ **and** parents looked exhausted.
- `ef32` [exact ✗ · FRR ✓]: Children laughed joyfully ~~while~~ **and** parents looked exhausted.
- `lingualens` [exact ✗ · FRR —]: ~~Children laughed joyfully while parents looked exhausted.~~ **```**
- `steer` [exact ✓ · FRR ✓]: Children laughed ~~joyfully while parents~~ **joyfully. Parents** looked exhausted.

## transitive_verb

### success — idx 4767

- **source**: She lost consciousness.
- **target**: She ~~lost consciousness.~~ **fainted.**

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: She ~~lost consciousness.~~ **fainted.**
- `ef32` [exact ✓ · FRR ✓]: She ~~lost consciousness.~~ **fainted.**
- `lingualens` [exact ✓ · FRR —]: She ~~lost consciousness.~~ **fainted.**
- `steer` [exact ✓ · FRR ✓]: She ~~lost consciousness.~~ **fainted.**


### near — idx 4776

- **source**: She entered a state of sleep.
- **target**: She ~~entered a state of sleep.~~ **fell asleep.**

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: She ~~entered a state of sleep.~~ **was falling, falling,.**
- `ef32` [exact ✗ · FRR ✓]: She ~~entered a state of sleep.~~ **was falling, falling,.**
- `lingualens` [exact ✓ · FRR —]: She ~~entered a state of sleep.~~ **fell asleep.**
- `steer` [exact ✓ · FRR ✓]: She ~~entered a state of sleep.~~ **fell asleep.**

## turn_taking

### success — idx 4823

- **source**: Sorry to interrupt, who will chair the next meeting?
- **target**: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?

- `ours` via ef32 [exact ✓ · FRR ✓] ←focus: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?
- `ef32` [exact ✓ · FRR ✓]: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?
- `lingualens` [exact ✓ · FRR —]: ~~Sorry to interrupt, who~~ **Who** will chair the next ~~meeting?~~ **Meeting?**
- `steer` [exact ✓ · FRR ✓]: ~~Sorry to interrupt, who~~ **Who** will chair the next meeting?


### near — idx 4834

- **source**: Sorry, quick interruption: the staging environment is offline.
- **target**: ~~Sorry, quick interruption: the~~ **The** staging environment is offline.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: ~~Sorry, quick interruption: the staging~~ **The** environment is offline.
- `ef32` [exact ✗ · FRR ✓]: ~~Sorry, quick interruption: the staging~~ **The** environment is offline.
- `lingualens` [exact ✓ · FRR —]: ~~Sorry, quick interruption: the~~ **The** staging environment is offline.
- `steer` [exact ✗ · FRR ✓]: ~~Sorry, quick interruption: the staging environment is offline.~~ **```**


### fail — idx 4846

- **source**: Let me say first that the network configuration is outdated.
- **target**: ~~Let me say first that the~~ **The** network configuration is outdated.

- `ours` via ef32 [exact ✗ · FRR ✗] ←focus: ~~Let me say first that the network configuration is outdated.~~ **The**
- `ef32` [exact ✗ · FRR ✗]: ~~Let me say first that the network configuration is outdated.~~ **The**
- `lingualens` [exact ✗ · FRR —]: Let me ~~say first that~~ **state** the network configuration is outdated.
- `steer` [exact ✗ · FRR ✗]: ~~Let me say first that the network configuration is outdated.~~ **```**

## universal_quantifiers

### near — idx 4860

- **source**: All the water evaporated.
- **target**: ~~All the~~ **The** water **completely** evaporated.

- `ours` via steer [exact ✗ · FRR ✓] ←focus: ~~All the~~ **The** water evaporated.
- `ef32` [exact ✗ · FRR ✓]: ~~All the water~~ **The completely water, completely, completely completely, completely completely, completely completely, completely completely, completely, completely completely completely, completely completely completely** evaporated.
- `lingualens` [exact ✗ · FRR —]: ~~All the~~ **The** water evaporated.
- `steer` [exact ✗ · FRR ✓]: ~~All the~~ **The** water evaporated.


### fail — idx 4853

- **source**: All employees received a bonus.
- **target**: ~~All~~ **The** employees received ~~a~~ **the entire** bonus.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: ~~All employees~~ **Every employee** received a bonus.
- `ef32` [exact ✗ · FRR ✓]: ~~All employees~~ **The entire entire entire entire entire amount entire entire amount amount of of the of the entire of the the the entire entire amount of of** received ~~a~~ **received** bonus.
- `lingualens` [exact ✗ · FRR —]: ~~All~~ **The** employees received ~~a~~ **the** bonus.
- `steer` [exact ✗ · FRR ✗]: ~~All employees~~ **Every employee** received a bonus.

## verbal_suffix

### near — idx 4934

- **source**: The manager will facilitate the meeting.
- **target**: The manager will ~~facilitate~~ **facility** the meeting.

- `ours` via ef32 [exact ✗ · FRR ✓] ←focus: The manager will ~~facilitate~~ **facilities** the meeting.
- `ef32` [exact ✗ · FRR ✓]: The manager will ~~facilitate~~ **facilities** the meeting.
- `lingualens` [exact ✗ · FRR —]: *(copy — no edit)*
- `steer` [exact ✗ · FRR ✗]: The manager will ~~facilitate~~ **be facilitating** the meeting.


### fail — idx 4932

- **source**: The system needs to integrate the new changes.
- **target**: The system needs to ~~integrate~~ **integral** the new changes.

- `ours` via steer [exact ✗ · FRR ✗] ←focus: The system needs to integrate the new ~~changes.~~ **modifications.**
- `ef32` [exact ✗ · FRR ✓]: The ~~system~~ **essential** needs to ~~integrate~~ **fully** the new changes.
- `lingualens` [exact ✗ · FRR —]: The system needs to ~~integrate the new changes.~~ **be updated.**
- `steer` [exact ✗ · FRR ✗]: The system needs to integrate the new ~~changes.~~ **modifications.**

