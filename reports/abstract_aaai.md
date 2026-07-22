# AAAI提出用 Abstract(2026-07-22作成)

**タイトル**: SAE-EditFlow: Rendering SAE Interventions as Residual-Stream
Edits with a Learned Editor

**規則の確認**: 数値はすべてfeature-specプロトコル(03§3'、同定/評価
データ分離)の確定値。oracle-spec値は不使用(README規則0')。改善②③
(絞り込み・srcゲート・適応fine-tune)の結果確定後に数値を更新すること。

---

## Abstract(日本語)

スパース自己符号化器(SAE)は、大規模言語モデル(LLM)の内部状態を解釈
可能な活性の組に分解し、過去形や敬語といった言語学的特徴との対応づけを
可能にする。しかし、特定された活性が本当にその特徴を担うかを確かめる
因果検証の標準手段である活性への直接介入(定数への書き換えや単一方向の
加算)は、効果が弱く不安定であることが先行研究自身により報告されている。
さらに我々は、代表的枠組みであるLinguaLensの特徴同定が同定・検証に同一
データを用いており、データを半分にすると選ばれる活性の4割超が入れ替わる
という選択の不安定性を実測で示す。本研究は、介入を固定ルールではなく
学習された編集器に生成させる新手法SAE-EditFlowを提案する。編集器は
編集操作の流れを学ぶEdit Flows系の基盤から出発し、特徴ごとにオフラインで
同定したSAE活性の指定(スペック)と入力文を読み、編集ベクトルΔhを凍結
LLMの残差ストリームに注入する。テキストは凍結LLM自身が生成するため、
最小対の編集成功が活性同定の因果的証拠となる。スペックは評価と分離した
同定プール上の符号つき平均として構築され、選択の不安定性を回避しつつ
(分割半分間の類似度0.83)、評価文やその正解を一切参照しない。学習は
一般コーパスから合成した破損復元データのみで行い、評価データセットに
対してゼロショットである。英語99特徴の最小対編集による評価では、提案
手法は厳密一致で全先行介入手法を上回り、誤ったスペックでは編集しない
という介入としての特異性(random統制で編集率ほぼ0)を唯一備える。
特徴の増減をLLM判定で測る因果効果指標FICでは、特徴を消す方向で全層
0.85–0.94と、先行のクランプ介入(全セルほぼ0)および指示文ベースライン
を大きく上回る。これらの結果は、「同定された活性の因果検証」が学習
された介入によって初めて実用的な精度で可能になることを示す。

---

## 英語版の骨子(本文執筆時の参照用)

1. Background: SAEs decompose LLM internals; linguistic-feature
   identification (LinguaLens lineage) needs causal verification.
2. Problem: direct activation interventions are weak/unstable (their own
   admission); identification itself is selection-unstable (our
   split-half: >40% top-1 turnover) and lacks data separation.
3. Method: a learned editor (Edit Flows backbone, LLM2Vec encoder) reads
   an offline per-feature spec (signed pool-mean over held-out
   identification pairs; stability 0.83) + the input, and emits Δh into
   the frozen LM's residual stream; text comes from the frozen LM.
4. Training: self-supervised corruption-restoration on a general corpus
   only (zero-shot w.r.t. the evaluation set).
5. Results: best exact among interventions with unique specificity
   (random ≈ 0); FIC (ablation) 0.85–0.94 across layers vs ≈0 for
   clamping; prompting wins raw exact but fails specificity.
6. Claim: causal verification of identified activations becomes
   practical only with a learned intervention generator.

## 提出メモ

- 数値アンカー(2026-07-22確定、L12ほか): exact sup 0.128(random
  0.000)vs 較正steer 0.086 / AxBench準拠 0.070 / LinguaLens準拠 0.016;
  prompting 0.180はrandom 0.088で特異性なし(net 0.092)。FIC sup
  L4/L12/L20 = 0.85/0.94/0.85。
- 改善①②(specゲート・絞り込み)/③(適応fine-tune)の確定後、
  「全先行介入手法を上回り」の水準と数値アンカーを更新。
- amp方向(足す)はpromptingが最良(FIC 0.49)である点は本文で正直に
  記載(abstractでは方向を「特徴を消す方向で」と限定済み)。
