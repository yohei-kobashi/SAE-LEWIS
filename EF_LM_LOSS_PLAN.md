# EF版 through-LM loss 学習計画(最終版、2026-07-18 ユーザー承認)

本ファイルはユーザーとの目標統一で確定した学習計画の記録。ここに書かれた
決定を変更する場合は必ずユーザーに確認する。

## 0. 決定事項(すべてユーザー決定)

| 項目 | 決定 |
|---|---|
| 大前提 | SAE介入(仕様)を信号としてeditorに入力し、editorの出力embedding(Δh)を凍結LMのresidual streamに戻す。テキストは凍結LMの生成 |
| 系統 | EF版のみ(LEWIS版は不採用)。レート場+内容場、反復推論 |
| loss | 凍結LMの出力を使って計算(teacher-forced NLL、勾配は凍結LM越し) |
| フレーム | **プロンプトなし・テキストのみ(LinguaLens完全準拠)**。rewrite等の指示プロンプトは使わない |
| steer基底 | なし(純粋学習)。v2(基底あり・rewrite枠)は比較腕として保全 |
| レイヤー | **L4 / L12 / L20 の3層、それぞれ独立に学習**(層条件付き単一モデル案=工夫2は不採用) |
| データ準備 | **工夫1採用**: 1回のforwardから全対象層のzを一括生成(サイドカー) |
| 評価 | LinguaLens準拠 — 現象(feature)ごとに対応layer/SAEを特定して評価する枠組み。学習は現象非依存 |

## 1. モデル(各層で同一構成、独立に3本)

```
入力: [BOS] + x_t のトークン列、SAE仕様(z_amp, z_sup — 層LのSAE)
エンコーダ: LLM2Vec双方向Gemma-2-2B + LoRA r=32
            (SAEEditFlow流用、feature-tokens条件付け=層LのW_dec行+符号+強度、
             tトークンで中間状態の時刻を条件付け)
ヘッド:  λ_i = σ(rate_head(h_i))   … 位置iに未実行編集が残る確率(bias −2初期化)
         v_i = content_head(h_i)    … 内容方向(Linear(d,d)ゼロ初期化)
出力:    Δh_i = λ_i · v_i  (ゼロ初期化により恒等スタート)
注入:    凍結gemma-2-2b-itの層L出力、[BOS]+x_t スパンの各位置
         (エンコーダ入力とLM入力が同一トークン列 → 位置対応は恒等)
```

## 2. 学習手順

- **LM入力**: `[BOS] + x_t + x1`(チャットテンプレートなし・指示なし)。
  lossはx1スパンのNLLのみ。「x0の後に編集済み文を出す」挙動自体をΔhが
  誘導する — 編集の成立はdo(Δh)にのみ帰属可能
- **中間状態サンプリング**: P(t=0)=0.5(x_t=x0)、それ以外 t〜U(0,1)。
  x_tはトークン列difflibアラインメントの部分適用で構築(CPU)。仕様は常に
  ペアの完全なdiff spec。t→1で「残り編集なし→λ→0」を学習し、推論時の
  反復と自己停止(Σλ<閾値)を可能にする
- **null教師**: empty仕様 p=0.08、mismatched仕様(パートナーspec、P5)
  p=0.12 — いずれも**Δhノルム抑制のみ**(NLL項なし; bare枠ではコピー目標を
  強制できないため。抑制でΔh≈0=無介入と同値)
- **ノルム予算**: 位置ごと‖Δh_i‖ ≤ 0.5·‖dvec_L‖、超過ペナルティ w=0.05、
  null抑制 w=0.1。実測ノルム/予算比を必ず報告
- ハイパラ: batch 4 × accum 2、lr 3e-4(ヘッド)/1e-4(LoRA)、40kステップ、
  k-top 32、k-amp/k-sup log:1-32、dev監視2000ごと(best-dev採用)、
  ckpt 2000ごと、resume対応
- **fail-fast**: 10kステップ時点でジョブ内probe100(bare枠、true/empty)を
  実行し、崩壊・再現不能を早期検出

## 3. データ準備(サイドカージョブ、工夫1)

- 対象: `runs/prod_gemma_v4/corruption` 先頭〜30万レコード + `corruption_seldev` 全件
- 各レコードのx/x'トークン列をベースgemma-2-2bに**1回ずつforward**し、
  層{4, 12, 20}のhidden statesを同時抽出 → 各層SAEでエンコード →
  **編集局所max-pool**(トークン列difflibの差分位置; なければglobal)→
  **層別blocklist**マスク → top-64 → zフィールドを差し替えた新キャッシュdir
  `corruption_z_l{4,12,20}` を書き出し(v4キャッシュ構築と同一の意味論)
- **L12も同パスで再計算**(3層のz意味論を完全統一。既存キャッシュは不変)
- **層別blocklist**: `build_grammaticality_blocklist.py` をL4/L20のSAEで実行
  (L12は既存 `runs/blocklist/blocklist.npy`)。同ジョブ内で先に構築
- resumable(シャード単位)

## 4. 評価(新probe、LinguaLens準拠bare枠)

- 499ペア(seed 42、既存probeと同一標本)。仕様は評価時にその層のSAEで
  オンザフライ構築(編集局所プール+層別blocklist、k 64/64)
- **入力 `[BOS]+src`、greedy、max_new = len(src)+24、extract_sentence**
- 腕: **ef(学習editor)/ steer0.5(同一bare枠の固定描画)/ raw(無介入)**
- 条件: true / empty / random。指標: exact / sim_target / copy
- 診断: **λ-IoU**(λ場 vs gold編集位置、true/empty/random)、ノルム実測
- 反復ablation(L12のみ、probe再実行): 1パス vs 2-3パス(自己停止)
- **現象別レイヤー表**: 3層のrecordsから現象ごとの最良層を集計
  (LinguaLensの全層カバレッジに対応する3点版)
- 参考併記: rewrite枠steer0.5=0.2385、v2結果(枠が異なることを明記)

## 5. ジョブとスケジュール(miyabi停止 7/29 9:00 絶対期限)

| 日 | 作業 | ジョブ |
|---|---|---|
| 7/18 | 実装(model/trainer/probe/sidecar/runner)+CPUテスト → **サイドカージョブ投入** | short-g 8h ×1 |
| 7/19 AM | サイドカー確認 → **L12学習ジョブ投入** | short-g 8h |
| 7/19 PM | L12のprobe100(10k時点、投入+3h頃)確認 → 生存なら **L4・L20投入(並行)** | short-g 8h ×2 |
| 7/20 | 3層probe回収(ジョブ内自動)、反復ablation、層プロファイル集計 | probe+オフライン |
| 7/21-22 | 予備(分岐A/B発動時の再学習1回分) | — |

## 6. ゲートと分岐

- **L12ゲート**: ef exact > steer0.5(同一bare枠)、empty/random=無介入同等、
  λ-IoUがtrue専有
- **分岐A(複製が立ち上がらない)**: bare枠ではΔhが「文の複製」まで誘導する
  必要があり、ノルム予算が律速し得る → norm-alpha 0.5→1.0 の腕を1本
  (予算超過は実測報告で透明化)。発動はユーザーに報告してから
- **分岐B(λが立たない)**: アラインメント既知の編集位置でλに弱い直接教師を追加
- L12通過後のL4/L20はレシピ変更なしの機械的展開

## 7. 論文成果物

1. 主表: 層{4,12,20} × 腕{ef, steer0.5, raw} × 条件{true, empty, random}
2. 深さプロファイル図(exact・λ-IoUの層依存)
3. 現象別レイヤー表(featureごとの対応層 — LinguaLens準拠評価の中核)
4. 反復推論ablation、ノルム実測/予算比
