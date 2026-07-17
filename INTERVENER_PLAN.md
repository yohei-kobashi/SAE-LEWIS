# 学習型介入生成器(Intervener)— 提案モデル復活の設計(2026-07-17)

## 決定(ユーザー、2026-07-17)

- **判定基準は出力インターフェース**: 編集器がトークンを直接出力する(EF/
  旧LEWIS実装)なら条件付けで因果証明にならない。**編集結果をembedding
  (Δh)としてresidual streamに返し、テキストが凍結LM自身の生成として
  出る**なら、LinguaLens/steeringと同じ do(residual) の介入であり因果的に
  正当。アーキテクチャ(EFの双方向エンコーダ等)は流用してよい。
- これにより「学習型の提案モデル」が論文に復帰する。⚫(トークン出力の
  EF/LEWISの除外)は維持 — 出力チャネルだけが変わる。
- レイヤー: **L12で学習 → 勝てばL20**(AxBenchアンカー層)で再学習し層比較。

## アーキテクチャ

```
[仕様 z_amp,z_sup] + [src文]
  → 双方向エンコーダ(LLM2Vec gemma-2-2b + LoRA、EFの資産を流用可)
  → 位置依存の介入場 Δh_i(srcトークン位置ごと)+ 全域ベクトル Δh_dec
  → 凍結 gemma-2-2b-it の層ℓ(=12)に注入:
     prefill: rewriteプロンプト中のsrcスパン位置iに Δh_i を加算
     decode : 各生成ステップに Δh_dec を加算(steer scope=allの学習版)
  → 凍結LMが書き換え文を生成(greedy)
```

- 現行 steer は特殊ケース(Δh_i ≡ Δh_dec ≡ α·(za−zs)@W_dec、固定描画)。
  提案 = **固定描画を、src文脈と仕様に条件付けられた学習介入場に置換**。
  P-IのWHAT問題(固定介入は内容を指定できない)への正面解。
- 出力ヘッド W_out は**ゼロ初期化**(恒等スタート = 無介入から学習)。
- **ノルム制約**(重要): Δh のノルム予算(steerのα·|dvec|と同桁)を正則化
  で課し、実測ノルムを必ず報告 — 「介入」が任意のsoft-prompt制御チャネルに
  退化しない(=介入らしさを保つ)ための制約。査読対策も兼ねる。

## 学習

- データ: 既存corruptionキャッシュ(x0=corrupted, x1=clean, spec=SAE diff
  top-k)。ゼロショットOOD(LinguaLens不使用)は不変。
- 損失: 凍結gemma-2-2b-itにrewriteプロンプト+src(注入あり)を与え、
  x1をassistant応答としてteacher-forcing、**NLL(x1)を凍結LM越しに最小化**
  (勾配は生成器のみ)+ Δhノルム正則化 + **empty仕様→Δh=0**(null教師、
  前提保護)+ **random/mismatched仕様→Δh抑制**(P5の教訓を最初から)。
- ReFT/LoReFT(AxBenchの学習型介入)が機構の先行 — 新規性は「**事例レベル
  SAE仕様に条件付けられた介入生成**」と「編集=因果評価枠組み内での使用」。
- 位置対応: src文はrewriteプロンプト内に埋まる。-itとLLM2Vecはgemma語彙を
  共有(prompt_maskで実証済み)なので、srcスパンのトークン位置はオフセット
  で厳密に対応付く。

## 評価(既存C1'枠組みへそのまま接続)

- probe500(499ペア): true/empty/random、exact/sim/copy。
  **バー = steer0.5 0.2337/0.2385**(固定描画に学習が勝つか)。
  統制: empty→無編集(copy~1)、random→床、raw床0.0601。
- FRR(gold cache共有)、S_min(effector=intervener)も同じ器で回る。
- 因果主張の書き方: do(Δh(z))で凍結LMの生成が編集を実現。内容は学習される
  が(ReFT同様)、テキストは凍結LMの計算が生む=検証器は外生的。仕様への
  特異性はrandom/empty統制とゼロショットOODが担う。ノルム報告で「介入の
  大きさ」を明示。

## 実装ファイル

- `intervener.py` — InterventionGenerator(エンコーダ流用: editflow.pyの
  条件付け機構 or model.pyのLLM2Vecロード; W_out zero-init; 署名は既存
  コードから逐語コピーすること)
- `train_intervener.py` — 上記損失、resume、~50k steps、bf16
- `run_intervener_l12.sh` — interact-g複数セッション、BUDGET+resume、
  完走後にprobe自動実行
- 評価: `eval_clamp_baseline.py` に `--intervention learned --intervener-ckpt`
  を追加(SteerHookのdvecを生成器出力に差し替え、prefillは位置場・decodeは
  Δh_dec)— records互換でcompare系がそのまま使える

## 落とし穴メモ

- 2モデル同載(生成器のLLM2Vec 2b + 凍結-it 2b)— GH200 96GBで可、bf16。
- 生成器エンコーダのLoRAはS3から温めるか新品か → **新品**(S3はトークン
  出力用に最適化されており、恒等スタート(W_out=0)なら温める利点が薄い。
  条件付け部(feature_token_embeds)のみ流用候補)。
- teacher-forcing時のassistant応答フォーマット(chat template)は
  eval_clamp_baseline の make_enc と同一にする(学習/評価の枠一致)。
- ノルム予算の初期値: steer0.5の実測 |α·dvec| の分布を最初のセッションで
  測ってから決める(桁を合わせる)。
