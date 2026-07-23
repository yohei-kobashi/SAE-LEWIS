# 05 — Analysis / Discussion / Limitations 執筆資料
(2026-07-23 feature-specプロトコルへ全面改訂)

> 🔶 大前提: 提案=SAE spec→学習editor→Δh→residual stream(記述名:
> SAE-conditioned edit-flow intervention)。steer/clampはベースライン。
> 🚫 数値はfeature-specプロトコル確定値のみ(規則0')。旧枠組みの証拠系
> (P-J/P-I/C1'/判別木/net-FRR/S_min)は本文に使わない — 旧版の該当節は
> git履歴(6b0cbf3以前)参照。

## 1. Analysis の物語

### 1a. 相関的同定 ≠ 因果的編集(束ねる解釈枠)

同一の同定プール(4,451ペア)から作った介入なのに:

| 介入 | exact(L12 ablation) | 意味 |
|---|---|---|
| LinguaLens準拠(FRC-r3クランプ) | 0.016 | 検出的に選んだtop-3の二値クランプは編集を実行できない |
| AxBench準拠(AUROC-r1 steer) | 0.070 | 検出最良latentの単一方向加算も同様 |
| 較正steer(我々のspecで固定描画) | 0.086 | 同じspecでも固定ルールでは限界 |
| **学習editor(本手法)** | **0.128** | 同じ同定情報を学習介入が初めて編集に変換 |

- 「同定情報が悪い」のではなく「介入の書き方が情報を捨てている」ことの
  実証 — 差はすべて介入側(同定プールは共通)。
- 同型の乖離の先行と並置: **Hase et al.**(重み編集: localization≠editing)、
  **AxBench自身**(SAE-A: 検出+32%でもsteering−5%)、我々(テキスト編集)
  — 3ドメイン一致の乖離。系譜: amnesic probing(decodable≠used)の因果版。

### 1b. promptingとの正確な関係(3軸で書く)

1. **raw exact**: prompting 0.180 > 本手法 0.128 — 正直に書く。
2. **特異性**: prompting はrandom spec指定でも0.088編集(誤った指定に
   従ってしまう=因果検証の統制が効かない)。本手法はrandom≈0.000-0.054。
3. **統合FIC**: 本手法 0.463 > prompting 0.410(L12)。ablation成分は
   0.937 vs promptingの過剰編集による低下。
→ 「因果検証の道具」としては学習介入が優る、という主張の形にする。

### 1c. FIC成功の中身(健全性検証、04§9i)

- 本手法のablation成功195件 = exact 63 + 流暢な言い換え132、コピー0、
  壊れ文2% — 「壊して稼ぐ」ではない。
- steer L12の統合FIC 0.569は本手法0.463を上回るが**壊れ文19%**
  ("rowspan rowspan…")を含み割引が必要(壊れ文除外版FICは残タスク)。
- promptingは壊れ文1%だが過剰編集(random床の高さと同根)。

### 1d. enhancement/ablationの非対称(方向の物語)

- ablationが強く(0.128-0.148)、enhancementは挿入駆動が律速
  (copy率高)。改善ラウンド: ⑦文脈内spec(+15-16%両方向)、
  ③invstd(enhancement+32%)、T系学習側施策(v6: enh 0.130だが
  abl退行 — 切り分け中)。【最終構成確定後に数値を固定】
- enhancementのraw床の機序: 形態系の非文的対事実文をrewriterが文法修復
  してしまう(60件中50+が文法修復と一致、04§9d)— 方向難易度の実体。

## 2. Limitations(順序もこのまま推奨)

### 2a. 🔴 最重要 — specの出所
specは同定プールのminimal pairから構築する。「概念名だけからspecを作る」
(概念→latent引き当て)は行わない — それはAxBenchが0.695と測った検出
問題の世界。主張の射程 = 「**同定済みプールが与えられたときの因果的検証**」。
帰結: minimal pairプールが無いベンチ(AxBenchのConcept500等)には
本手法の腕を置けない(axbench_testdata.md §3)。
【T4採用時: train区画適応でzero-shot主張を手放したpool適応版を併記 —
2行構成】

### 2b. 同定/ハイパラ分離の残存注記
介入強度のdev選択は同定プール内の標本(spec構築への寄与<2.3%)で行って
おり、同定/ハイパラの完全分離ではない(漏れの兆候なし: dev選択3.5 vs
評価ピーク2.5の逆向きずれ)。**3分割移行(eval_split v2、§9n)で解消
予定** — 執筆時点の状態を正直に書く。

### 2c. editorの学習と汎化
- 学習はDolma合成破損ペアのみ(LinguaLens-Dataゼロショット)。ただし
  editorは学習時のper-pair specで訓練され、評価時のfeature平均specは
  分布ギャップを持つ(v6系の学習側施策はこのギャップの実験)。
- exactの絶対値はoracle上界(付録)の~26-47%(featureごとの残存) —
  「feature specが原理的に劣る」と「editor未適応」の合成で、T系切り分け
  が分離を試みる。

### 2d. 測定器の限界
- FIC judge(gpt-4o)はreliabilityのみ検証(validityは人手なしに主張
  しない)。judge再判定ノイズ床≈1.5pt。
- exactは0/1で、流暢な言い換え成功を数えない(FICが補完)— 2指標の
  役割分担を明示。
- greedy採用は復唱枠の前提(temp1.0でexact半減、04§9h)— デコード規約
  の議論は03§7'。

### 2e. その他(1文ずつ)
英語のみ・Gemma-2-2B(-it)+Gemma Scope 16k・単一SAE幅・LinguaLens-Data
99現象に限定。LinguaLensのFRCは論文(条件付き)とコード(周辺)が乖離 —
コード側に忠実。再現アンカーはパターンとレンジ(LinguaLensはモデルが
違う。AxBenchは同一スタック)。

## 3. 撤回・除外済みの主張(復活させない)

- 🚫 oracle-spec値の本文使用(0.3166/0.2485/0.1804等 — 付録の上界診断のみ)。
- 🔶 editorを使わない枠組み(素のsteer/clampを最終形とする構成)。
- ⚫ トークン出力EF系のすべて(routed/ef32/λ-IoU/M0/P-B)と、それに
  依存する旧証拠系(P-J/P-I/C1' 2×2/判別木A71-C21/net-FRR/S_min/
  FRR judge機構)— 結論はfeature-spec測定(04§9)が引き継ぐ。
- 「事例レベル仕様」を前提とした差別化(「集約しないのは我々だけ」)。
- (b)方向別構成(abl=⑦/enh=③)— 数値最良だが不自然としてユーザー不採用。
- 検索spec(入力適応)— 方向別に支配され不採用。⑦×③重ねがけ — 両方向で
  単独最良未満、不採用。v6(T1+T3同時)— 合算で届かず不採用推奨(切り分け中)。
- 「介入ベースSAE評価は初」等(README規則4)。
- 「judge品質/validityを測る」(reliabilityのみ)。
- 「事例k=32がfeatureの表現幅」等の旧免許規則の対象claim — feature-spec
  移行で前提ごと消滅(specはfeatureレベルk=64)。
