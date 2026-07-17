# 04 — Experiment 執筆資料(🔶2026-07-18 editor前提へ改訂)

> 🔶 大前提: 提案=Intervener(SAE仕様→editor→Δh→residual stream)。
> steer/clampの数値はすべて**ベースライン**。Intervenerの主結果欄は
> v2学習完了待ち(v1=0.0200は失敗、残差基底のv2が学習中)。

数値はすべて実測・確定(進行中は明記)。トークン出力EF系の数値は
載せない(⚫)。旧版にあった routed/ef32/λ-IoU/M0/P-B は論文から除外 —
結論の引き継ぎ先: P-B → P-J、M0 → B-1 介入k掃引、内容混入の証拠 →
S_min組成分析。

## 0. 主結果0 — Intervener(提案editor)probe500 【学習完了待ち】

- v1(恒等初期化、40k steps): exact **0.0200** — コピーアトラクタ崩壊
  (true≈random 0.018、copy 0.65、|Δh|≈budget/4)。empty→copy 0.972で
  null防御は機能。**失敗分析として書ける**: x1≈x0のNLLはコピーが支配 →
  恒等スタートの学習介入は「無介入」に退化する。
- v2(残差基底 0.5·dvec + 学習補正、編集トークンCE重み4×): 学習中。
  初期状態=steerベースライン(0.2385@499)を厳密再現するため、
  下限はベースライン同等が設計上保証される。
- 報告軸: exact(バー=steer0.5 0.2385)、統制(empty→copy、random→
  ベースライン挙動)、**実測ノルム/予算比**(介入サイズの明示)。

## 1. 主結果1 — C1'ベースライン: 介入の編集力は仕様が決める(499/997ペア)

**同一の介入機構で仕様だけ差し替える2×2(exact)**:

| | 現象レベル仕様(彼らの完全プロトコル) | 事例レベル仕様(informed上限) |
|---|---|---|
| clamp+再生成 | **0.0160**(LinguaLens: FRC top-3, set10/0) | 0.1743 |
| steer+再生成 | **0.0701**(AxBench: AUROC top-1)/ 0.0822(FRC×steer) | **0.2337**@997(0.2385@499) |

- **床と統制**: raw(無介入の書き換え役)**0.0601**、recon **0.0100**
  (再構成通過だけで6×崩壊 — LinguaLens完全版0.0160は無介入を4倍下回り、
  仕様の寄与+0.006 ≪ 機構の損傷)、random 0.0521。
- 動作点の崖: steer α=0.25/0.375/0.5/0.75 → 0.098/0.176/**0.2385**/0.18
  (鋭い頂点)。clamp 5/10/20 → 0.06/0.17/0.03。
- **主張文**: 「介入機構を固定したまま、仕様を両論文のプロトコルから
  事例レベル編集局所deltaに替えると、exactは無介入床+0.01〜0.02から
  0.17〜0.23へ上がる。介入の編集力は仕様が決める。」

## 2. 主結果2 — P-N: 両論文自身の指標でも再現(実装検証を兼ねる)

LinguaLens指標(E_abl、judge gpt-4o; 較正 P(absent|src)=0.126):
我々の仕様×clamp **+0.631** / ×steer **+0.370** vs LinguaLens完全版
+0.091・AxBench完全版 +0.115。raw参照 P(absent|raw)=0.303 —
**LinguaLens完全版(0.220)は無介入より特徴を消せない**。
AxBench指標(調和平均、gpt-4o-mini): 我々 1.211/1.262 vs 完全版
0.570/1.121(raw 1.081)— **AxBench完全版は無介入と区別できない**。
指標間の順位反転(LinguaLens指標ではclamp>steer)= 多軸報告の必然性。

## 3. 主結果3 — P-J: 検出の完璧さと因果的実在の乖離(形1のr掃引)

学習ゼロreadout(P-I機構)での現象レベル(target-free)仕様の因果発火:

| r | FRC式 true/rnd | AUROC式 true/rnd |
|---|---|---|
| 1 | — | **0.06 / 0.07(ゼロ、比0.86×、27/11 p=0.014)** |
| 3 | **0.17 / 0.00(67/0、p=1.4e-20)** | 0.29 / 0.15 |
| 8 | 0.36 / 0.01 | 0.97 / 0.13 |
| 32 | 1.48 / 0.05(30×) | 2.98 / 0.76(3.9×) |
| 64 | 2.51 / 0.17 | 4.12 / 1.08 |
| 事例delta(参照) | 5.39 / 1.77 | 同左 |

- **sanity gate**: AUROC top-1 mean **0.939** / median 0.979(AxBench自身の
  SAE-A 0.917超え)→ r=1のゼロは選択失敗ではなく単一latentの限界。
  逸話: universal_quantifiers の AUROC=1.000 latent = "proper names"。
- **形1の主張**: rと単調・64で非飽和 → 「因果的に有効な現象表現は
  top-3より広い」(「対応」の定義を因果的に取り替えたと明示)。

## 4. 主結果4 — P-I: 因果床とWHERE/WHAT分解(500ペア、学習ゼロ)

- WHERE陽性: true発火 3.57 vs random 1.08(3.3×)、対応符号検定
  **393/31(tied 76)、p=5.6e-81**。
- WHAT不能: 2 WHAT規則×2介入×2スコープ×4閾値の全てで exact 床
  (0.008-0.010)。→ 「介入はWHEREを確立するが、正確な編集には
  再生成(steer+rewrite)が要る」。
- 現象別WHERE: interrogative +4.12、expressive +3.30(上位=発話行為・
  節タイプ)/ 比喩系が底(non_synecdoche_metonymy −0.17、唯一の負)。

## 5. 主結果5 — FRR(steer行)とjudge信頼性

- steer0.5: FRR 0.7327 / **net-FRR 0.4062** / random床 0.3265(GPT-4o、
  997ペア)。3 judgeで方向一致。**🔴 EF行を落とした表の再集計が残作業**
  (判定キャッシュはあるのでオフライン)。
- judge信頼性(限界コスト貢献): exact一致ペア上のFRR = 自己一致率。
  steer 0.9781(GPT-4o)。gold/system独立rngにより always-A judge は
  0.50に落ちる(degeneracy免疫)。per-system自己一致の広がりが
  非差異性(McNemarの妥当性条件)の検査を兼ねる。
- feature別net-FRRがsteerの見かけの実現を暴く: past_tense net 0.200、
  expressive −0.400、subject_verb_inversion −0.300 等(偽陽性支配の現象)。

## 6. 判別木(98現象、paper版)

**A 介入で編集可能 71 / C 効果器側(WHERE有・介入不可)21 /
B SAE側の示唆(WHERE無・B2可)2 / D 不定(WHERE無・B2不可)4**。
- C類は形態・接辞系+WHERE強陽性の談話系(noun_plural、possessive_form、
  echo_questions +9.00 等)— WHAT問題の現象名リスト。
- B類(given_known: B2は1.000なのにWHERE 0)= 交絡latentの具体名。
- 証拠3種のみで判別(do-介入編集 / P-I WHERE / B2フロア)。

## 7. 再現アンカー

- **LinguaLens再現**(完了): FIC 4/4で正 — past_tense **12.0**(Table2
  8.3)、metaphor **3.0**(6.9)がアンカー; 強い行(politeness 46.9→13.6、
  linking_verb 22.9→8.6)は別スタック(Llama+OpenSAE→gemma+GemmaScope)
  に転移せず。E_abl 4/4正。ctr絶対値は判定設計が違うため比較しない。
- **AxBench再現**(実行待ち): 2Bアンカー SAE 0.177/0.151、SAE-A
  0.166/0.132(L10/L20)。同一スタックなので強い照合。+ ll_set10
  (LinguaLens機構を彼らのベンチで)= 相互評価セル。

## 8. B2(SAE不使用参照)

prompt書き換え 0.1242、empty copy 0.4770(統制崩壊)— 判別木のフロアと
Limitationsでのみ使用。

## 9. 主結果6 — B-1 介入k掃引(完走、499ペア、局在性=操作幅)

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| true exact | 0.0681 | 0.0802 | 0.1082 | 0.1503 | 0.1944 | **0.2365** | 0.2385 |
| random | 0.0521 | 0.0641 | 0.0481 | 0.0481 | 0.0501 | 0.0601 | 0.0501 |

- 単調成長・**k=32で膝、32→64飽和**。random床は全kで平坦(特異性)。
- **k=1はtargetを見た最良の1本でも床**(0.0681 ≈ raw 0.0601 ≈ AxBench
  完全版0.0701)— 単一latentの限界は選択の問題ではない(informed でも
  1本では編集を指令できない。P-J sanity gate の介入編集版)。
- 免許規則の読み: 操作インターフェース幅の曲線(「~32本で飽和」)。
  表現幅は形1(P-J)・形2(S_min安定核)で主張。

## 10. 主結果7 — B-2 S_min + 安定核×FRC3(完走、134ペア、形2)

- **|S_min| median 5 / mean 7.9 / min 1 / max 43** vs 全仕様 98.7 —
  事例あたりの因果ハンドルは仕様の~5%。S_min=1の実例(of_genitive、
  passive_voice)。
- B-1との整合: 事例ごとの正しいハンドルは少数だが**事例ごとに違う**
  (Jaccard 0.0-0.4)→ magnitude順のtop-kは k=32 まで必要(magnitude
  順位≠因果順位)。
- **安定核(22現象≥3ペア)**: 非空 **20/22**、サイズ median 1.5 /
  mean 3.2。核ラベルは現象整合(past_perfect "had" 4/4、future "will"、
  interrogative "questions" 5/5)= 因果基準が意味的に正しい辞書項目を
  掘り当てる。
- **核∩FRC3 = 8/22、FRC3出現率 0.0-0.6** → 因果的対応集合はFRCの検出的
  同定と**大部分は別物**(一部は「含んで広い」)。**AUROC top-1 は核に
  頻出**(出現率1.00の現象あり)だが単独では編集不能(B-1 k=1床)。
- 空核 = 比喩系(synecdoche等、Jaccard 0.00)— 判別木と一貫。
- 出典: `runs/tables/smin_vs_frc.md`、`runs/prod_gemma_v6/prune_spec_steer/report.md`

## 11. 進行中・空欄
- **Intervener v2 学習+probe500(主結果0の本体)** → 勝てばL20再学習
- AxBench再現 L20(生成済み・judge実行中)→ L10
- FRR表再集計(EF行なし)+ intervener行の追加、clamp腕FRR(任意)、
  BLEU/chrF、定性例
