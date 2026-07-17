# 04 — Experiment 執筆資料(設定・ベースライン・全確定数値)

出典: `PAPER_OUTLINE.md` §4-6(6b〜6l)、`runs/tables/*`、
`runs/paper_metrics/report.md`。数値はすべて実測・確定(暫定は明記)。

## 1. 実験設定

- **評価データ**: LinguaLens-Data 英語 minimal pairs。
  - 499ペア(seed 42、全システム共通標本)。probe 200ペアは動作点F選択に
    使用 → **残り300ペアが汚染なしholdout**。
  - **997ペア = 499 + 未接触の確認ブロック498**(主要主張の事前登録確認用)。
- **指標**: exact(正規化一致)/ sim_target(埋め込み類似)/ copy率 / SARI /
  λ-IoU(WHERE)/ **FRR**(LLM judgeによる特徴実現率、LinguaLens整合)/
  net-FRR = FRR(true) − FRR(random)。統制: empty(無編集保存)・random
  (同数・同大きさ・ID違いの偽仕様)。
- **統計**: Δexact = 不一致ペアの符号検定 / 厳密McNemar、Δsim = paired 95% CI
  (`scripts/compare_ef_pipeline.py`、records.jsonl を idx 結合)。
- **FRR プロトコル**(`scripts/judge_feature_realization.py`): gold方向 =
  judge(src vs tgt)(equal除外・システム間共有キャッシュ)、システム判定 =
  judge(src vs out)(copyは自動equal)、FRR = P(判定方向==gold方向)。
  A/B提示順は **gold用とsystem用で独立のシード付きrng**(6e-1バグ修正済み)。
  主judge = GPT-4o、頑健性 = gemma-2-9b-it(ローカル)+ gpt-5.4-nano。
  exact は FRR の下界 → **FRR−exact = 「方向は正しいが不正確な編集」の量**。

## 2. ベースライン(ラベリングに注意)

| ID | 手法 | 一言 |
|---|---|---|
| B1 | **忠実クランプ**(LinguaLens公式repo+OpenSAE精読準拠): set介入・残差はSAE再構成で完全置換・全位置全ステップ、-itモデル+中立Rewriteプロンプト、クランプ値{5,10,20,Z}掃引 | 介入(活性書き換え) |
| B2 | **指示プロンプト**: 同じ仕様情報(特徴の解釈ラベル)を自然言語で与えて書き換え | 我々と同じ「条件付け」列 |
| B3 | **steering**: 指令デルタのW_dec描画 dvec を層12残差に h += α·dvec、α掃引 | 介入(commanded-delta最近接) |
| B4 | v6パイプライン(tagger→列挙→editor→ranker) | 多段カスケード |
| B5 | input-copy / empty / random 統制 | 前提保護 |

🔴 **B1/B3 の「我々の仕様」腕は info-matched 機構ベースライン**(LinguaLens/
AxBenchの性能ではない)。両論文の**完全プロトコル**は別腕(P-K の
ll_protocol / ax_protocol)として測定済み — 表では明確に区別する。

## 3. 主結果1 — EF単体 vs 多段パイプライン(C3、S4確定)

matched 499 /(holdout 300):

| system | exact | sim | 備考 |
|---|---|---|---|
| v6 pipeline | 0.1102 (0.1033) | 0.6681 (0.6744) | copy 0.627、SARI 40.99 |
| **S3 thr0.1** | **0.1904 (0.1833) = +73% (+77%)** | 0.6192 (0.6202) | sign test p<0.0001 / 0.003; sim −0.049 [−0.072,−0.025] 実在のコスト |
| S3 thr0.5 | 0.1683 (0.1633) | CI両標本で0を含むタイ | バランス王(p=0.002 / 0.020) |
| S3 det | 0.0905 | **0.6816 (0.6854)** | sim側もEFが点推定で上回る(ns) |

- バケット: 1-op **2.1×**(0.394-0.404 vs 0.192)、2-3 +40%、
  **4-8 2.4×(局所性の配当)**、9+ はEFのみ命中(1/13; holdout 0/7)。
- WHERE: **λ-IoU 0.7449** vs empty 0.1539 / random 0.3252。タガーの
  count-oracle 0.7472 とパリティ → 優位は「知識」でなく「決定経路」。
- 前提保護: **empty no_edit 1.0000**(全F・全ckpt・500ペア)。
  random no_edit 0.87(バー0.88をわずかに割る — 限界として記載)。

## 4. 主結果2 — C0: routed が全方式に勝つ(997 + 未接触498)

| system | exact@997 | SARI | sim | FRR(GPT-4o) |
|---|---|---|---|---|
| **routed**(count-rule T=1) | **0.2839** | **65.88** | **0.6554** | 0.8306 |
| ef32(EF単体, k=32) | 0.2237 | 63.20 | 0.5809 | **0.8735** |
| steer0.5(B3) | 0.2337 | 58.94 | 0.6016 | 0.7327 |
| oracle(2候補) | 0.3872 | — | — | — |

- **未接触確認ブロック498**: routed **0.2892** vs steer 0.2269(+80/−49、
  p≈0.007)、ef32 0.2369。規則は完全教師なし・事前登録・凍結。
- 499(全6系): routed 0.2786 / oracle 0.5210。
- **per-feature 構造(997、`_per_feature.csv` — 論文の中心図)**:
  - EF支配 = 形態・屈折の単一トークン現象: noun_plural 0.727 vs steer 0.000、
    past_tense 0.700 vs 0、anaphor 0.875 vs 0.125、superlative 0.667、
    adjectival_suffix 0.500 vs 0 …
  - steer支配 = 構造・多語現象: interrogative 0.700 vs EF 0、cleft 0.643、
    passive_voice 0.583 vs 0、future_perfect 0.917、SAI 0.500 …
  - **routed は概ね max(EF, steer) を回収** — λ場が「形態=1ハンク/構造=
    多ハンク」を教師なしで見分けている直接証拠。
  - 両者ゼロの残余: 比喩系(metaphor/personification/hyperbole)、省略・外置、
    談話、一部の項構造 — oracle 0.39 が 1.0 でない理由の現象リスト。

## 5. 主結果3 — C1': 介入の編集力は仕様が決める(P-K、499ペア)

**同一機構で仕様だけ差し替える2×2(exact)**:

| | 現象レベル仕様(彼らのプロトコル完全版) | 我々の事例レベル仕様 |
|---|---|---|
| clamp+再生成 | **0.0160**(LinguaLens: FRC top-3, clamp10/0) | 0.1743(11×) |
| steer+再生成 | **0.0701**(AxBench: AUROC top-1, steer)/ 0.0822(FRC×steer) | **0.2337** |

- **床と統制**: raw(無介入の書き換え役)**0.0601**、recon(encode→decode
  通過のみ)**0.0100** — 再構成ダメージだけで無情報rewriterが6×崩壊。
  LinguaLens完全版 0.0160 は**無介入を4倍下回る**(仕様の寄与 +0.006 ≪
  機構の損傷)。AxBench完全版の因果的寄与は +0.010(random 0.0521 は raw 以下)。
- 我々の仕様は同じ機構で raw +0.114(clamp)/ +0.174(steer)。
- α掃引: steer の頂点は鋭く 0.5(0.25/0.375/0.5/0.75 = 0.098/0.176/
  **0.2385**/0.18@499)。clamp の崖: 5/10/20 = 0.06/0.17/0.03。
  → 連続介入は「動作点が崖」(C1)。
- **主張文**: 「介入機構を固定したまま、仕様を各論文のプロトコル(FRC top-3 /
  単一AUROC latent)から我々の事例レベル編集局所deltaに替えると、exactは
  無介入床+0.01〜0.02 から 0.17〜0.23 へ上がる。」編集器を含まない介入のみの
  主張。promptingに負けるのは許容済み(研究目標上、介入とsteeringに勝つ
  ことが要件)。

## 6. 主結果4 — P-N: 両論文自身の指標でも再現(499、実装検証を兼ねる)

judge = 各論文の選択(LinguaLens: gpt-4o / AxBench: gpt-4o-mini)。
較正: P(absent|src) = 0.126(= judgeはデータの正例を87.4%認識)。

**LinguaLens指標(ablation成功、E_abl)**:

| run | P(Y=0\|targeted) | P(Y=0\|random) | E_abl |
|---|---|---|---|
| 我々の仕様×clamp | 0.613 | 0.226 | **+0.631** |
| LinguaLens完全版 | 0.220 | 0.200 | +0.091 |
| 我々の仕様×steer | 0.693 | 0.437 | **+0.370** |
| AxBench完全版 | 0.349 | 0.309 | +0.115 |

**AxBench指標(concept/instruct/fluency 0-2 の調和平均)**:

| run | true | random | concept | fluency(true) |
|---|---|---|---|---|
| 我々の仕様×clamp | **1.211** | 0.615 | 1.32 vs 0.63 | 1.80 |
| LinguaLens完全版 | 0.570 | 0.536 | 0.57 vs 0.53 | 1.98 |
| 我々の仕様×steer | **1.262** | 0.454 | 1.33 vs 0.55 | 1.69 |
| AxBench完全版 | 1.121 | 1.038 | 1.10 vs 1.02 | 1.97 |

- **raw参照**: P(absent|raw) = 0.303 / AxBench score(raw) = 1.081。
  → **LinguaLens完全版(0.220)は無介入より特徴を消せない**(copy 46%)。
  **AxBench完全版は無介入と区別できない**(+0.046 / +0.04)。
  我々の仕様は全指標で明確に超える(+0.31/+0.39、concept +0.27/+0.28)。
- 指標間の順位反転も記録: LinguaLens指標では clamp > steer(exactの逆)。
  P(absent)は文破壊でも上がる → fluencyを持つAxBench指標・exact・FRRの
  **多軸報告の必然性**の追加実例。
- 実装検証としての読み: clampはLinguaLensの定性的主張(targeted≫random)を
  強再現、steerはAxBench指標で concept +0.78 かつ fluency維持 — **両実装は
  両論文自身の指標で合格**(「実装が悪い」批判への先回り)。

## 7. 主結果5 — FRR(997、3 judge、rng修正後)

| system | gemma-2-9b-it | **GPT-4o(主)** | gpt-5.4-nano | exact |
|---|---|---|---|---|
| ef32 | **0.7894** | **0.8735** | **0.7679** | 0.2237 |
| routed | 0.7691 | 0.8306 | 0.7331 | **0.2839** |
| steer0.5 | 0.7406 | 0.7327 | 0.6922 | 0.2337 |
| steer net-FRR | 0.3234 | 0.4062 | 0.2221 | — |
| steer_rnd(偽陽性フロア) | 0.4172 | 0.3265 | 0.4701 | — |

- **judge不変の主張**: 3 judge全てで順位一致(ef32 > routed > steer)かつ
  全judgeでroutedがexact首位 → FRRとexactはシステムを異なる順に並べる =
  多軸報告の必然性がjudge非依存に成立。
- **厳密McNemar**(不一致ペア A/B、GPT-4o):
  ef32>steer +0.1408(205/67、p=1.9e-17、Bonferroni後も3 judge有意)、
  ef32>routed +0.0429(54/12、p=1.7e-07; nano 8.7e-04; gemmaのみ補正後n.s.
  — gemmaは識別力最低と独立に測定済みで整合)、routed>steer +0.0980
  (155/59、p=3.9e-11; gemma n.s.)。
- **トレードオフの生カウント**: routedはcount-ruleがsteerに切替えたペアでのみ
  ef32と異なる → GPT-4o判定で実現を54失い12得て正味−42、対価にexact+60 =
  **手放した実現1件あたりexact 1.43件**。FRRとexactは緊張関係にあり、
  動作点は用途が決める。
- 特異性: EF側 net-FRR 0.69 ≫ steer側 0.22-0.41。feature別net-FRRがsteerの
  見かけの実現を暴く: past_tense(steer net 0.200 vs ef32 1.000)、
  expressive −0.400、subject_verb_inversion −0.300、split_infinitives −0.308、
  static_dynamic −0.500(randomでも同等以上に「実現」= 偽陽性支配)。
- 残余フロンティア分解: 比喩系は「方向実現は可能・正確編集が不可能」
  (hyperbole FRR 1.00全系全judge)。extraposition はjudge分裂(gemma/nano
  0.25-0.50 vs GPT-4o 0.75-1.00)→ **単独judgeで書けない事実として報告**
  (旧「唯一の到達不能現象」記述は撤回済み)。

### judge信頼性(6e系 — 論文では「測定器の検証」節)

| judge | 自己一致率 [95%CI] | net-FRR | randomフロア |
|---|---|---|---|
| **GPT-4o** | **0.9860 [0.974,0.992]** | 0.4062 | 0.3265 |
| gemma-2-9b-it | 0.9717 [0.957,0.982] | 0.3234 | 0.4172 |
| gpt-5.4-nano | 0.8789 [0.853,0.901] | 0.2221 | 0.4701 |

- 原理: **exact一致ペアでは judge(src,out) と judge(src,tgt) が同一比較** →
  そこに限定したFRR = judgeの自己一致率(無償の自然発生重複)。
- degeneracy免疫: gold/system提示順は独立rng → always-A judge は 0.50。
  観測値は全て0.5から大きく上。
- **1つの測定が2役**: judge選定の根拠(GPT-4oが3指標すべて最良、judge間一致
  も上位2judgeが相互最一致: gemma↔GPT-4o 0.85-0.89 > nano対 0.77-0.83)+
  **McNemarの非差異性条件の検査**(per-system自己一致 0.9860/0.9926/0.9781、
  広がり1.45pt → 最悪ケースでも減衰保存: 観測+0.1399 vs 真値+0.1400)。
- 減衰の代数: p_A−p_B = (θ_A−θ_B)(q0+q1−1) — 対応差はスケールされるが
  シフトしない(符号保存)。前提=非差異的誤分類。Chen et al.
  (arXiv:2601.05420) の Rogan–Gladen を反転して導出。
- 🔴 書き方の禁止事項: 「judge品質/validity」「ノイズ床」(撤回済み — 有意性
  はMcNemarで判定)「reliabilityがシステム間分離を単調に予測」(gemmaで反証)。
- nanoを主判定にできない技術的理由: gpt-5.x系は temperature≠1 を拒否 →
  サンプリング判定しかできず、決定的条件でも19-21 flips(純粋なAPI非決定性)。

## 8. 統制・ablation群

- **M0(条件付け絞り込みの棄却)**: k8/16/32/64 = exact 0.060/0.126/0.210/
  0.190、FRR 0.7753/0.7895/~0.82/0.7874 — **kを絞るとexactもFRRも単調劣化**
  (低kはcopyでなく誤方向編集)。k=32が両指標同時の最適点。
- **B2統制崩壊**: prompt書き換えは exact 0.1242(EFの65%)だが
  **empty copy 0.4770 / random copy 0.2745**(EFは empty 1.00)。
  バケット逆転: 1-op 0.0909(EFの1/4.3)vs 9+ 0.154 — 自由再生成はテールに
  強く最小編集に弱い。
- **gold ΔSLOR = −0.50**: 正しい最小対編集は流暢さを下げる方向。steerの
  −0.10 は「指令された非流暢化の拒否」、EFの−0.98は過剰損傷0.47 —
  文法性はgold ΔSLORへの近さで測る(C4c)。
- **P-L(局所化steering)= null(ablation脚注)**: マスクは機能する(random行
  は変化)が、trueの抑制32特徴はプロンプトほぼ全域で発火 → マスクが実質全域
  で local ≡ all(4桁一致)。局在化の余地なし。

## 9. 因果分析(3層分解の実測 — Analysis節の核)

### P-I: 学習ゼロの介入readout(500ペア、layer-12、LMのhead teacher-forced)

- **WHERE陽性**: true仕様の発火(Δ_j = log p_int − log p_base が閾値超え)は
  randomの2〜3倍(delta_local: 3.57 vs 1.08)。**対応符号検定 393/31
  (tied 76)、p = 5.6e-81**。同定活性はLM自身の予測を因果的に動かす —
  SAE活性が言語現象を担うことの**学習フリーな因果証拠**。
- **WHAT不能(確定)**: argmax p_int も PMI(介入が最も昇格させたトークン)も
  exact 床(0.008-0.010 = 退化ペア4/500)、simはcopy基準0.6033未満。
  2 WHAT規則 × 2介入 × 2スコープ × 4閾値の全てで再現。
- **診断的逆転**: B3自由再生成 0.2337 ≫ readout 0.01 — 対応語には文脈内協調
  (右文脈・複数トークン)が必要で、teacher-forced単一位置置換では原理的に
  届かない。**EFの双方向エンコーダがまさにこれを解いた**。
- 現象別WHERE: 上位=発話行為・節タイプ(interrogative +4.12、expressive
  +3.30)、中位=形態・時制(EFの領地と一致)、底=比喩系
  (non_synecdoche_metonymy **−0.17** — 唯一の負)。
  → per-feature exact × FRR × WHERE の3面表で「検出/指令/介入のどの層まで
  到達するか」を現象ごとに示せる(論文図)。

### P-J: 選択方式の因果比較(target-free現象レベル仕様、readout WHERE)

| r | FRC(LinguaLens式)true/rnd | AUROC(AxBench式)true/rnd |
|---|---|---|
| 1 | — | **0.06 / 0.07 ← 信号ゼロ** |
| 3 | **0.17 / 0.00**(true専有) | 0.29 / 0.15 |
| 8 | 0.36 / 0.01 | 0.97 / 0.13 |
| 32 | 1.48 / 0.05(30×) | 2.98 / 0.76(3.9×) |
| 64 | 2.51 / 0.17 | 4.12 / 1.08 |
| 事例delta(参照) | 5.39 / 1.77 | 同左 |

- 対応検定: auroc_r1 27/11(p=0.014、**平均比0.86× = trueが下**)、
  frc_r3 **67/0(p=1.4e-20)**、frc_r32 277/3、auroc_r32 364/29。
- **sanity gate 通過**: top-1 AUROC mean **0.939** / median 0.979 / 74/99現象
  ≥0.9 — AxBench自身のSAE-A平均0.917超え。→ r=1のゼロ信号は選択失敗ではなく
  **単一latentの限界**。逸話: universal_quantifiers のAUROC=1.000 latentの
  Neuronpediaラベルは「proper names and titles」(交絡の実例、1文だけ)。
- **論文の1文**: 「AUROC選択器は検出をほぼ完璧にこなす(0.939)が、その単一
  latentは因果的に何も動かさない(0.86×)。FRCの3本は微弱でも純粋な因果信号
  を持つ(67/0)— 検出の完璧さと因果的実在は、同一パイプライン内で乖離する。」

### P-B: FRC同定特徴での条件付け(現象レベル仕様 → 学習済み編集器)

| | FRC-true | FRC-random | 事例delta(参照) |
|---|---|---|---|
| exact | **0.0140** | 0.0000 | 0.210 |
| λ-IoU | **0.5037** | 0.3440 | 0.7449 |

- true > random 明確(7ペア vs 0、IoU +0.16)→ **FRCの中身は実在の信号を
  運ぶが、事例レベル仕様の~7%**。true/randomは同等にミスマッチなので
  **train/testミスマッチ批判に免疫**。
- 情報漏洩の修正を明記(holdoutレシピ不一致で452/500混入 → 修正済み。
  P-Bは否定的結果なので漏洩はFRC有利にのみ働き、10×崩壊は保守的)。
- **2つの独立経路(学習済み編集器 P-B・無学習readout P-J)が同じ結論**:
  現象レベル同定特徴はWHEREを部分的に運び、WHATを運ばない。

## 10. 未完了(執筆時にプレースホルダ)

- **S6**(v7+P5学習)進行中 — GOならef32/routedの数値更新は**新システム
  扱いで再確認が必要**(凍結値0.2839/0.2892はS3+steer0.5の事前登録値)。
- **P-O**(事例レベル最小条件集合 |S_min|、union/Jaccard分析)— S6決着後。
  仮説:「|S_min|は小さいが中身が事例ごとに違う」= 現象レベル失敗族の機構的
  説明の完成。高Jaccard現象は因果検証済み辞書の種。
- BLEU/chrF(sacrebleu導入で埋まる)、定性例(`run_examples.sh` 未実行)、
  P-D k-curve、readout/P-JセルのFRR。
