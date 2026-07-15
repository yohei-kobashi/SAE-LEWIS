# Related Work — サーベイ確定分と、書いてはいけないこと

第1ラウンド(deep-research、2026-07-15): 6角度 → 26出典 → 130主張抽出 →
25主張を3票の敵対的検証 → **12確定 / 13棄却**。棄却分は「使えない」の記録
として §X に残す(同じ罠を二度踏まないため)。

**現状のカバレッジ**: 軸1(probing)・軸2(離散編集)・軸3(steering限界)
は確定。**軸4(SAE介入・SAE懐疑論)、反例(c)(judge自己一致率)、(d)
(ルーティング)は検証済み知見ゼロ = 未調査**。第2ラウンド実行中。
→ **「反例は見つからなかった」と書いてよいのは(a)(b)だけ**。(c)(d)は
「調べていない」が正しい状態で、「探したが無い」と混同してはならない。

---

## 0. 新規性主張の現状(最重要)

| 主張 | 状態 | 根拠 |
|---|---|---|
| (a) SAE特徴を条件とした離散編集 | **反例なし(検証済)** | LinguaLens/Edit Flows/LEWIS/LevT/LaserTagger 全てで条件付け信号が従来型 |
| (b) SAE特徴を介入ノブでなく編集の仕様として使用 | **反例なし(ただし脅威1件)** | 同上。制約付きLevT系譜(未検証)が最近接 |
| (c) exact一致ペアのFRR = judge自己一致率 | **未調査** | 第2ラウンド待ち |
| (d) 編集サイズでSAE介入と離散編集をルーティング | **未調査** | 第2ラウンド待ち |

### 🔴 新規性の置き場所 — 編集語彙に置くと即死する

**INS/DEL/SUB という編集操作語彙それ自体に新規性を主張してはならない。**
Levenshtein Transformer (Gu et al., NeurIPS 2019) と LaserTagger
(Malmi et al., EMNLP-IJCNLP 2019) が先行し、GECToR/Seq2Edits も replace系
操作を持つ。LEWIS 自身の新規性すら編集語彙ではなく (a) 複数スパン同時編集
(b) スタイル分類器attentionテンプレート+スタイル特化MLM穴埋めによる教師
なし並行ペア合成の側にあった。

**差分は条件付け信号の側にのみ厳密に置く**: 「編集操作の語彙は既存
(Gu 2019 / Malmi 2019)。我々の差分は、その操作を**モデル内部から回収した
解釈可能な特徴辞書**で条件付ける点である」。

---

## 軸1. LLM内部の言語学的特徴 / detecting ≠ using

### LinguaLens (arXiv:2502.20344, EMNLP 2025 Main) — 我々の評価基盤かつ対比対象

SAE特徴を**活性空間の介入ノブとしてのみ**使用: forward propagation中に
対象特徴の活性値を固定(ablation=0 / enhancement=10)、出力は free-form
generated text。逐語:

> "When we modify the values of SAE's activation during forward propagation,
> we expect that such targeted interventions will influence the model's
> behavior"
> "In the ablation experiment, we set the target feature's activation to 0,
> and in the enhancement experiment, we set it to 10."

論文が挙げるSAE特徴の全8用途(活性マッピング、FRCによるbase vector
ランキング、EALE事前フィルタ、GPT-4o検証、反事実解析、介入対象、言語間
比較、因果効果測定)は**すべて解析または活性空間介入**で、編集仕様として
の用途は皆無。公式リポジトリの `Intervener` クラスも「intervention indices
指定→該当層で介入→テキスト生成」で編集操作語彙を持たない。

**🔴 先回りすべき留保**: LinguaLensは編集を**データ構築側**で使っている:

> "A counterfactual sentence s− is produced through a minimal edit that
> deletes or substitutes the trigger while preserving semantic content"

反証にはならない((a)は "as model output" と限定されるため)が、差分を
**「編集をデータ構築の道具として使う vs. SAE特徴を条件に編集操作を生成
する」**と明示しないと査読者に突かれる。

### Elazar et al., "Amnesic Probing" (arXiv:2006.00995, TACL 2021 vol.9 pp.160-175)

P-B(FRC同定した現象特徴で編集すると約10倍崩れる)の**系譜**。逐語:

> "Our findings demonstrate that conventional probing performance is not
> correlated to task importance, and we call for increased scrutiny of
> claims that draw behavioral or causal conclusions from probing results."
> "we point out the inability to infer behavioral conclusions from probing
> results and offer an alternative method that focuses on how the
> information is being used, rather than on what information is encoded."

**🔴 必須のヘッジ3点**:
1. **優先権を過剰主張しない**: "the canonical" ではなく **"a canonical /
   most-cited"**。競合定式化を併記すること — Ravichander, Belinkov & Hovy,
   "Probing the Probing Paradigm: Does Probing Accuracy Entail Task
   Relevance?" (EACL 2021, https://aclanthology.org/2021.eacl-main.295/);
   Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks"
   (EMNLP 2019); Belinkov, "Probing Classifiers: Promises, Shortcomings,
   and Advances" (Computational Linguistics 2022)。
2. **技法(INLP)は係争中**: Kumar, Tan & Sharma, "Probing Classifiers are
   Unreliable for Concept Removal and Detection" (NeurIPS 2022,
   arXiv:2207.04153); "Improving Causal Interventions in Amnesic Probing
   with Mean Projection or LEACE" (Findings of ACL 2025, arXiv:2506.11673)。
   ただし両者とも**原理を強化する方向**で反証ではない。
3. **軸のずれを正直に書く**: amnesic probingの軸は「モデルが**内部で**その
   特性を使うか」、P-Bの軸は「**我々が**特徴を外部制御信号として使えるか」。
   detecting≠using → detecting≠commanding は厳密な一般化ではなく**軸を
   またぐアナロジー**。系譜として提示し、直接的な延長として断定しない。

---

## 軸2. 離散編集 / 非自己回帰編集

### 定義枠: Malmi et al., "Text Generation with Text-Editing Models" (NAACL 2022 Tutorial, arXiv:2206.07043)

逐語(PDFのリガチャ・ハイフン折返しで素朴なgrepは0ヒット、pypdf抽出で復元):

> "Text-editing models are sequence-transduction methods that produce the
> output text by predicting edit operations which are applied to the inputs.
> In contrast, the traditional seq2seq methods produce the output from
> scratch, token by token."

著者は LaserTagger/Felix/EdiT5/Seq2Edits の当事者(Google Research)。
Table 2 が軸2の対象集合を列挙(EditNTS, Felix, GECToR, HCT, LaserTagger,
LevT, LEWIS, Masker, PIE, Seq2Edits, SL)。
**留保**: 7ページのtutorial abstractで網羅的サーベイではない → **定義枠・
整理枠として引用**し、カバレッジの典拠にはしない。

### 系譜と操作語彙

| 手法 | 年/会場 | 操作語彙 | **条件付け信号** |
|---|---|---|---|
| Levenshtein Transformer (Gu et al.) | NeurIPS 2019, arXiv:1905.11006 | insertion + deletion のみ(**SUBなし**) | source sequence(cross-attention) |
| LaserTagger (Malmi et al.) | EMNLP-IJCNLP 2019, arXiv:1909.01187 | タグ付け(語彙の正確な内容は**要再確認**) | 入力文のみ(BERT符号化) |
| LEWIS (Reid & Zhong) | Findings of ACL-IJCNLP 2021, arXiv:2105.08206 | insert/keep/replace/delete | source text + **スタイル分類器attention** |
| Edit Flows (Havasi et al.) | arXiv:2506.09018 | (**要再確認** — 0-3で棄却) | prefix / 画像 / CFGスケールのみ |

LevT 逐語: "Unlike previous approaches, the atomic operations of our model
are insertion and deletion." デコーダの分類器は Deletion / Placeholder /
Token Classifier の3つのみ。**訂正**: 置換は「複数イテレーションにまたがる」
のではなく**1イテレーション内**で実現 — "We essentially decompose one
iteration of our sequence generator into three phases: 'delete tokens –
insert placeholders – replace placeholders with new tokens'"。

LEWIS 逐語: "We propose a coarse-to-fine editor for style transfer that
transforms text using Levenshtein edit operations (e.g. insert, replace,
delete)." / §2.1 "The set of coarse Levenshtein transition types are
insert, keep, replace, and delete."

### 反例探索(a)(b) 陰性 — 条件付け信号の全在庫

- **LevT**: Appendix A 逐語 "we always omit the source information x in
  conditional sequence generation tasks such as machine translation which is
  handled by the cross-attention with an encoder on x." 全文検索で
  `style`=0, `attribute`=0。
  **🔴 査読で起きやすい混同**: §4 "Collaborate with Oracle" の "LevT has
  better interpretability and controllability" は**操作語彙**の性質であって
  条件付け信号の性質ではない。oracleは ground-truth へのアラインメント由来
  のトークン位置レベルactionで、言語的特性の仕様ではなく、policyを条件付け
  るのではなくdecode時に置き換える。この2つの "interpretable" を明示的に
  区別すること。
- **LEWIS**: 全文で sparse autoencoder / autoencoder / dictionary learning
  / sparse coding / interpretable features / feature dictionary /
  monosemanticity が**全て0ヒット**。
- **LaserTagger**: BERT-baseが入力系列のみを受ける。control code / style
  label / 潜在ベクトル一切なし。Google公式ブログの "controllable" は条件
  付けではなく**出力語彙の制限**を指す("By controlling the output phrase
  vocabulary...LaserTagger is less susceptible to hallucination")ため反証
  にならない。
- **Edit Flows**: 28ページv3 PDFを全文抽出しカウント: `sparse autoencoder`=0,
  `steer`=0, `style transfer`=0, `activation`=0, `probe`=0,
  `monosemantic`=0, `sae`=0。`interpret`=2は無関係な用法("interpreted as
  an insertion")、`feature`=1はソフトウェア的用法。39箇所の condition* 文脈
  を全列挙した結果、条件付けの全在庫は (i) ランダムprefix (ii) 画像early
  fusion (iii) CFGスケール(1.5/0.5/1.0)のみ。

### 🔴 主張(b)への最大の脅威(**未検証** — 第2ラウンドで確認中)

**制約付きLevT系譜**: Susanto et al. (ACL 2020) / EDITOR (Xu & Carpuat,
TACL 2021)。これらは「**離散編集を仕様(lexical constraints)で条件付ける**」
ことをまさに行っている。SAE-LEWISの差分は「**仕様がモデル内部から回収された
解釈可能な特徴空間に存在する**(表層の語リストではない)」点に置く必要がある。
一次資料未読 — Related Work執筆前に必ず確認。

### 🔴 軸2の残る穴

GECToR / EdiT5 / FELIX / Seq2Edits / discrete flow matching・diffusion text
editing は**通過した検証が存在しない**(Seq2Edits関連2件はいずれも1-2で棄却)。
特に「三操作語彙は新規でない」という重要な留保の根拠(GECToR/Seq2Editsが
replace系を持つ)**自体が未検証**。

---

## 軸3. Steering の限界

### Tan et al., "Analysing the Generalisation and Reliability of Steering Vectors" (NeurIPS 2024 main, arXiv:2407.12404)

我々の per-feature net-FRR 負値(past_tense −0.13, expressive −0.50,
subject_verb_inversion −0.30)の**公表済み先行事例**。逐語:

> "we rigorously investigate these properties, and show that steering vectors
> have substantial limitations both in- and out-of-distribution.
> In-distribution, steerability is highly variable across different inputs."
> "For all behaviours evaluated, steerability takes on a large range of values
> across different inputs, including negative values, where SVs produce the
> opposite of the desired behaviour."
> "Many of these datasets have almost half of the inputs being anti-steerable,
> implying that the effect of steering is highly unreliable."

"anti-steerability" は同論文の造語。著者: Tan, Chanin, Lynch, Kanoulas,
Paige, Garriga-Alonso, Kirk (UCL / FAR AI)。公表された異議は皆無で、後続
研究(arXiv:2505.22637; arXiv:2505.24859 — steeringの影響は "inconsistent,
negligible, or even counterproductive on some samples")は同方向に収束。

**🔴 引用時に必ず述べる2つのミスマッチ**:
1. **手法**: 評価対象は Contrastive Activation Addition (difference-of-means)
   **のみ**、モデルは Llama-2-7b-Chat / Qwen-1.5-14b-Chat、40のMWE系
   データセット+TruthfulQA中心の**多肢選択型**。論文中に**SAEは一切登場
   しない** → 「activation steering一般がper-input不信頼」の典拠として引用し、
   「SAE特徴steeringの直接測定」として引用してはならない。
2. **粒度**: Tan et al.の anti-steerability は**データセット内のper-input
   比率**(データセット平均のsteerabilityは概ね正のまま)。SAE-LEWISの
   net-FRR負値は **per-phenomenon の集計平均**であり、集計平均が負というのは
   **厳密により強い主張**。この非対称性は我々に有利 — Tan et al.は現象の
   存在の先行事例、負のnet-FRRは**より鋭い結果であって言い直しではない**。

**引用書式**: arXivは米綴り "Analyzing the Generalization..."、NeurIPS
proceedingsは英綴り "Analysing the Generalisation..." → 会議を引くときは後者。
二次アグリゲータの一部が "ICML 2024" と誤記しているので **NeurIPS 2024**。

---

## 軸4. SAE介入技術 / SAE懐疑論 — **未調査(第2ラウンド実行中)**

確定claim 12件の内訳は軸1が3件、軸2が7件、軸3が2件で**軸4はゼロ**。棄却
claimにも軸4由来のものは含まれない。したがって以下は**未検証の調査対象**
であって既知の事実ではない。

**🔴 最大の脅威(検索では発見済・未検証)**: **AxBench**
(arXiv:2501.17148, "Steering LLMs? Even Simple Baselines Outperform Sparse
Autoencoders", Wu et al., ICML 2025 Spotlight)。**Gemma-2-2B/9B = 我々と
同一のベースモデル**で「steeringでは単純なpromptingが既存全手法を上回る」と
報告。中核主張への最強の反論。

**生死を分ける区別**: AxBenchが「**SAE steeringは弱い**」に留まるのか、
「**SAEは編集の条件付けにも使えない**」まで含意するのか。前者なら我々の
立場はむしろAxBenchと**整合的**になる(「SAE steeringは弱い、だから離散編集で
条件付ける」)。P-B知見(検出できることと指令として使えることは別)も同方向。

その他の未検証: Towards/Scaling Monosemanticity (Bricken et al. 2023 /
Templeton et al. 2024), Cunningham et al. (arXiv:2309.08600), Gemma Scope
(arXiv:2408.05147), JumpReLU (arXiv:2407.14435), TopK SAE, SAEBench,
OpenSAE。

---

## X. 棄却された主張 — **書いてはいけないこと**

3票の敵対的検証で 0-3 / 1-2 となり**使用禁止**。同じ罠を二度踏まないための記録。

| 棄却された主張 | 票 | なぜ危険か |
|---|---|---|
| 「LinguaLensはtop-3のFRC-ranked base vectorのみで条件付け、FRC=PSとPNの調和平均」 | 0-3 | **🔴 P-Bの対比対象そのもの**。「同定は1-3特徴で足りるが編集には約32のinstance-level特徴が要る」という論証がこの対比に依存する。LinguaLens本文で再確認必須(第2ラウンド) |
| 「LinguaLens自身がSAE介入の制御信頼性の低さを認めている」 | 0-3 | 「SAE陣営自身による自認」という**有利すぎる論拠は使えない** |
| 「Edit Flowsの操作語彙はちょうどINS/DEL/SUBでMOVEなし」 | 0-3 | Edit Flowsの語彙の正確な内容は再確認が必要 |
| 「LaserTaggerの語彙はKEEP/DELETE/ADDちょうど3種でSUBは非プリミティブ」 | 0-3 | 同上 |
| Seq2Edits関連2件 | 1-2 | 軸2のSeq2Edits記述は**典拠なし** |
| 「steering vectorはOODでプロンプト変更に脆い」 | 0-3 | **Tan et al.のOOD側の主張は引用しない**。in-distribution側のみ3-0で通過 |
| 「LEWISはSAE interpretabilityの系譜全体に先行する」 | 事実誤り | Yun, Chen, Olshausen & LeCun, "Transformer visualization via dictionary learning" (arXiv:2103.15949, DeeLIO@NAACL 2021) がLEWISの**2ヶ月前**、SPINE (Subramanian et al., AAAI 2018) はさらに前。正しくは「LEWISは **Gemma Scope (2024) と LLM-SAEの系譜(Cunningham et al. 2023; Bricken et al. 2023)に**先行する」と狭く書く |
| 「LEWISには言語的特徴による条件付けが一切ない」 | 過剰 | スタイル分類器attentionという**学習された帰属信号**で条件付けており、「特徴を編集仕様に使う」の**最近接の隣人**。差別化は「**辞書としての解釈可能性**」に置く |

---

## Y. 執筆時の手続き規則

1. **逐語引用はすべてPDF全文抽出で再確認する。** このラウンドで二次要約器
   による**2件の捏造/矛盾**が検出された: Edit Flowsについてabstractと正面
   から矛盾する記述、LEWISについて存在しない大文字タグ(KEEP/REPLACE)を
   「逐語」と称した記述(実物は小文字と `<ins>`/`<repl>` マーカーのみ)。
2. **grepの0ヒットを信じない。** PDFのリガチャ("ﬁrst")とハイフン折返しで
   偽陰性が出る(NAACL tutorialで実際に発生)。
3. **「調べていない」と「探したが無い」を区別する。** 現状「反例なし」と
   書いてよいのは (a)(b) のみ。
4. **版番号を投稿時に再確認する。** LinguaLens v2 (EMNLP 2025 Main採択)、
   Tan et al. v6 (2025-05)、Edit Flows v3 が最新。
