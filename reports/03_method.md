# 03 — Method 執筆資料(🔶2026-07-18 editor前提へ改訂)

> 🔶 大前提: SAE介入(仕様)を信号としてeditorに入力し、editorの出力
> embedding(Δh)を凍結LMのresidual streamに戻す。Methodの中心は
> **提案editor(Intervener)**と、それを因果的に評価する枠組みの両方。

## 1. 全体像

対象: ある同定手法が出した活性集合 S(現象レベルでも事例レベルでもよい)。
問い: **S を信号として与えられたeditorの介入で、対象の言語現象だけを
反転させた最小対変換が出力されるか。**

```
同定手法(FRC / AUROC / 事例delta) → 活性集合S(仕様 z_amp/z_sup)
  → editor(Intervener: 仕様+src → Δh場)
     ベースライン: 固定描画(steer)/ LinguaLens式clamp
  → Δh を凍結LM層ℓのresidual streamに注入 + 凍結LMが再生成
  → 統制(random-S / empty / raw / recon)との分離
  → 指標(exact / FRR / 彼ら自身の指標)
  → 局在性(|S|掃引・最小集合S_min)・判別木
```

## 1b. 提案editor — Intervener(INTERVENER_PLAN.mdが正)

- 入力: 仕様(z_amp, z_sup)+ src文。双方向エンコーダ(LLM2Vec
  gemma-2-2b + LoRA、特徴トークン条件付け)。
- 出力: **embedding(Δh)のみ、トークンは出力しない** —
  prefill側は位置依存の介入場Δh_i(rewriteプロンプト中のsrcスパン)、
  decode側は全域ベクトルΔh_dec(各生成ステップ)。凍結gemma-2-2b-itの
  層ℓ=12(→L20比較)に注入し、凍結LMが書き換え文を生成(greedy)。
- **残差パラメータ化(v2)**: Δh = α·(za−zs)@W_dec(固定描画基底、α=0.5)
  + 学習補正(ゼロ初期化)。学習開始時点=steerベースラインを厳密再現。
- **ノルム予算**: 補正のノルムをsteer描画ノルムと同桁に正則化+実測報告 —
  介入が自由なsoft-promptチャネルに退化しないための制約。
- null教師: empty仕様→無介入(copy)、mismatched仕様→抑制(前提保護)。
- 学習: corruptionキャッシュ(x0,x1,SAE diff spec)でNLL(x1)を凍結LM
  越しに最小化(勾配は生成器のみ)+編集トークンCE重み。LinguaLens
  データは学習に不使用(ゼロショットOOD)。
- 因果主張の書き方: do(Δh(z))で凍結LMの生成が編集を実現。内容は学習
  されるが(ReFT同様)テキストは凍結LMの計算が生む=検証器は外生的。
  仕様への特異性はrandom/empty統制とゼロショットOODが担う。

## 2. データと仕様の構築

- 評価データ: LinguaLens-Data 英語 minimal pairs(499ペア seed42、
  確認ブロック込み997)。editorの学習はcorruptionキャッシュのみで行い、
  LinguaLensデータは学習に一切使わない(ゼロショットOOD)。
- **事例レベル仕様**(informed上限): `delta = z_tgt − z_src` の top-k
  符号分解(k_amp/k_sup、編集スパン局所プール、blocklist)。**SAE抽出
  のみに依存**(gemma-2-2b + Gemma Scope layer-12/16k JumpReLU)。
- **現象レベル仕様**(target-free、彼らのプロトコル): FRC top-r
  (LinguaLens公式 r=3; 実装は公式repoのコード側=周辺比率に忠実、
  リーク修正済み)/ AUROC top-r(AxBench公式 r=1; Mann-Whitney平均
  ランク、選択器のsanity gate = top-1 mean 0.939)。
- 運用時にtargetは無いので事例レベル仕様は「仕様が与えられたときの実現」
  の測定 — 仕様の出所は主張の射程外(Limitations筆頭)。

## 3. ベースライン効果器(既存機構の忠実実装 — editorの比較対象)

- **steer**(B3): dvec = za@W_dec − zs@W_dec を層12残差に h += α·dvec、
  全位置。α=0.5(掃引で確定した鋭い頂点)。-itモデル+中立Rewrite
  プロンプト、greedy。**editorの自明な特殊ケース**(Δh_i ≡ Δh_dec ≡
  α·dvec の固定描画)であり、Intervener v2の初期化基底。
- **clamp**(B1 = LinguaLens忠実): OpenSAE準拠のset介入(非活性は強制
  挿入)、**残差はSAE再構成で完全置換**、全位置・全ステップ、クランプ値
  {5,10,20,Z}。
- **統制**: random仕様(同数・同大きさ・ID違い)/ empty / raw(無介入の
  書き換え役)/ recon(multiply×1 再構成パススルー — 機構損傷の分離)。

## 4. 指標

- **exact**(正規化一致)= 最小対変換の厳密基準。sim_target・copy併記。
- **FRR**(LLM judge、LinguaLens整合): gold方向 = judge(src,tgt)、
  システム = judge(src,out)、提示順は gold/system 独立rng。
  **net-FRR = FRR(true) − FRR(random)** が特異性。3 judge
  (GPT-4o主・gemma-2-9b-it・gpt-5.4-nano)、厳密McNemar。
- **judge信頼性(限界コスト貢献)**: exact一致ペア上のFRR = judgeの
  自己一致率(無償の自然発生重複)。per-system自己一致がMcNemarの
  非差異性条件の検査を兼ねる(減衰の代数は6e-4/05参照)。
- **彼ら自身の指標**(P-N): LinguaLensのE_abl(judge gpt-4o)と
  AxBenchの3ルーブリック調和平均(gpt-4o-mini、公式テンプレ逐語)。

## 5. 局在性の測定器(局在×安定の2軸)

- **介入k掃引**(B-1): 事例レベル仕様を k=1,2,4,8,16,32,64 に切り詰めた
  steerのexact曲線 = **操作インターフェース幅**(免許規則: 表現幅とは
  書かない)。
- **最小介入集合S_min**(B-2): steerがexactを出したペアで、|delta|降順
  プレフィクスの二分探索 → 後退消去 → 最終検証(貪欲上界と明記)。
- **安定核×FRC3**(compare_smin_frc.py): 同一現象のS_min群で出現率≥50%
  の共通部分=現象側、入れ替わる部分=事例内容側、という操作的分解。
  安定核をFRC top-3と本数・中身で直接比較(免許規則・形2)。
- **現象レベルr掃引**(P-J、形1): target-freeなので内容混入が構成上ない。
  因果効果は学習ゼロreadout(下記)で測る。

## 6. 学習ゼロの因果readout(P-I)

凍結gemma-2-2b層12に介入し、**LM自身のheadの予測変化**(teacher-forced、
Δ_j = log p_int − log p_base の閾値超え=発火)を数える。読み手を介さない
因果の床。WHERE(位置)は測れるがWHAT(内容)は測れない — その分解自体が
結果(04参照)。

## 7. 判別木(Limitations用)

現象ごとに3種の証拠で分類: (1) do-介入編集(ベースラインsteer/clamp;
Intervener完成後はeditor行を追加)、(2) P-I WHERE、
(3) B2(SAE不使用のタスク実行可能性フロア)。
A = 介入編集成立 / C = WHERE陽性・介入不可(効果器側) /
B = WHERE無し・B2可(同定/SAE側の示唆) / D = WHERE無し・B2不可(不定)。
トークン出力の編集器(条件付け)は因果証拠にならないため判別に使わない。

## 8. 再現アンカー(実装忠実性の防御)

- **LinguaLens介入評価の再現**: 公式repoのIntervener逐語(set 0/10+recon
  完全置換、素プロンプト、temp1.0、100tok)、FRC top-3を1本ずつ50試行+
  random対照、E_abl/E_enh/ペナルティ付きFIC(w=0.5 — 式実装はTable2の
  FIC値を成功率から再現して検算済み)。
- **AxBench steeringの再現**: 公式repo逐語(AlpacaEval 10命令/概念の
  決定的サンプル、14 factor格子、addition steering、公式judgeテンプレ、
  factor選択half/holdout half)。同一スタックなので強いアンカー
  (詳細: axbench_testdata.md)。

## 9. この章の地雷

- steer/clampを「我々の提案」と書かない(既存機構の忠実実装=ベース
  ライン)。提案はIntervener(Δh出力のeditor)。
- **editorの出力インターフェースがΔhであることを冒頭で明示** —
  トークン出力の編集器(旧EF/LEWIS実装)は条件付けであり載せない(⚫)。
  hazard等トークン出力専用の機構記述は載せないが、流用している条件付け
  機構(双方向エンコーダ・特徴トークン)はIntervenerの記述として書く。
- FRCの論文/コード乖離、AUROCのsanity gate、リーク修正は必ず1文ずつ
  書く(実装批判への先回り)。
