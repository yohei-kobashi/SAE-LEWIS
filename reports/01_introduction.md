# 01 — Introduction 執筆資料(⚫EF除外後の全面改訂版)

## 1. 論文の一文

> **SAE activationsの因果妥当性は、介入によるminimal-pair編集の実行で
> 評価されるべきである — 本研究はその評価枠組みを提案し、既存の同定手法
> (FRC / AUROC / 事例レベルdelta)を同じ物差しで測る。**

新しい抽出手法の提案ではない。編集という評価器が加わることで評価の幅が
広がる: (i) exactという厳密なテキスト接地基準、(ii) 介入本数の局在性
スペクトル、(iii) net-FRRの特異性、(iv) 現象別の到達層(判別木)。

## 2. 動機の流れ(推奨の段落構成)

### 段落1 — SAEと言語現象の同定
SAEは残差ストリームを解釈可能な特徴辞書に分解する(Cunningham 2023,
Bricken 2023; Gemma Scope)。LinguaLens(EMNLP 2025)は言語現象のminimal
pairデータとFRC(PS/PNの調和平均)で「現象に対応する活性」を同定し、
AxBenchは概念検出のAUROCで同定を評価した。

### 段落2 — しかし同定の検証が検出的に閉じている
既存の検証は「発火統計が現象を追跡するか」(検出)か「顕著さがjudgeで
動くか」(自由生成の定性評価)に留まる。**検出できることと、その活性が
現象を因果的に担うことは別**であり(probing批判の系譜)、その乖離は
測られていない。CausalGymは言語minimal pairで介入手法を因果評価したが、
成功基準は**次トークン挙動のフリップ**であり、SAE latentは対象外。
SAEBench/RAVELは介入ベースのSAE評価を持つが、評価器は属性予測であり
**テキスト編集の実行**ではない。

### 段落3 — 提案: 編集可能性を因果基準にする
本研究は、同定された活性への**介入**(steering / clamping)が、対象の
言語現象**だけ**を反転させた最小対変換を実行できるか — random仕様・
無介入・再構成の統制との分離込み — を、活性集合の因果妥当性の基準として
提案する。編集は文法性を保った最小対変換として評価可能であり(exact)、
介入本数を振ることで表現の局在性まで測れる。

### 段落4 — 発見の予告(3層分解 = フック)

> **検出できる ≠ 因果的に担う ≠ 介入で書ける**

- 検出は完璧: AUROC top-1 mean **0.939**(AxBench自身のSAE-A 0.917超え)
- その top-1 は因果的に何も動かさない(true/random比 **0.86×**)。
  FRC top-3 は微弱だが純粋(67/0、p=1e-20)
- 仕様を現象レベル(彼らのプロトコル)から事例レベルdeltaに替えると、
  同一の介入機構で exact が 0.016/0.070 → **0.17〜0.23** — **介入の
  編集力は仕様が決める**
- 71/98現象で介入編集が成立(因果証明)、21現象はWHERE陽性でも実行
  不可(効果器側)、残りが同定側/タスク側 — 判別木

## 3. 貢献リスト

(i) **編集ベース因果評価枠組み**: minimal-pair編集の実行を成功基準に、
統制(random/empty/raw/recon)と多軸指標(exact・FRR/net-FRR・彼ら自身の
指標)で活性集合を評価するプロトコル。CausalGymの因果評価をテキスト編集
まで拡張し、SAEBench系の介入評価に編集という評価器を加える。

(ii) **適用による発見**: (a) 仕様が介入の編集力を決める(C1' 2×2 —
両論文の完全プロトコルは無介入床以下〜同等、事例レベル仕様は同じ機構で
3〜15×)、(b) 検出の完璧さと因果的実在の乖離(P-J)、(c) 局在性
スペクトル(介入k掃引・最小介入集合S_minと安定核×FRC3)、(d) 現象別の
判別木(A71/C21/B2/D4)。

(iii) **再現アンカーと相互評価**: LinguaLens介入評価(FIC)とAxBench
steering(SAE/SAE-A)の忠実再現、および相互評価(彼らのプロトコルを
我々のベンチで、LinguaLens機構を彼らのベンチで)。

## 4. Introで引く数値(最小セット)

- 仕様2×2: 0.0160 / 0.0701 vs **0.2337**(raw床 0.0601)
- 検出≠因果: 0.939 vs 0.86×(同一パイプライン内)
- 因果床: P-I 393/31、p=5.6e-81
- 判別木: A 71/98

## 5. タイトル案(⚫後、仮)

1. *Edit to Verify: Evaluating the Causal Validity of SAE Activations by
   Minimal-Pair Editing*
2. *Detection Is Not Causation: An Editing-Based Causal Evaluation of
   SAE Feature Identification*
3. *Can You Edit With It? Interventional Minimal-Pair Editing as an
   Evaluation of SAE Activations*

**使用禁止(EF系/旧枠組み)**: SAE-LEWIS / Edit Flows を冠する題、
~~Commanding Edits with SAE Features~~(条件付け枠組みの題)、
~~Lifting SAE Interventions into Discrete Edit Operations~~。

## 6. この章の地雷

- 「編集器(モデル)を提案」と読める表現をしない — 効果器は既存機構
  (steering/clamp)のみ。
- R5の先行を必ず踏む: CausalGym(挙動フリップ)/SAEBench(介入評価)/
  Beyond Input Activations(因果的latent選択の発想)— 「初」を置く場所は
  **テキスト編集という評価器**のみ。
- 免許規則: 事例レベルkを表現幅と書かない(05参照)。
- promptingに勝つ主張はしない(B2はSAE不使用参照としてのみ)。
