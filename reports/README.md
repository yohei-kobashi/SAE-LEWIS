# reports/ — 論文執筆用資料(日本語執筆の下敷き)

作成 2026-07-17、同日 🔵目的再固定・⚫EF完全除外を反映して全面改訂。
論文 = **編集ベース因果評価枠組み**(SAE活性の因果妥当性を、介入による
minimal-pair編集の実行で評価する)。効果器は **steer / clamp のみ**。
EF系(S系列・ef32・routed・λ-IoU・M0・P-B)はコード/runsに履歴として
残るが**論文には一切載せない**。

## ファイル構成

| ファイル | 対応する章 | 内容 |
|---|---|---|
| `01_introduction.md` | Introduction | 動機、主張、貢献リスト、冒頭数値、タイトル案 |
| `02_previous_works.md` | Related Work | 冒頭の⚫改訂ガイドに従って読む(D.3/D.6等は退役) |
| `03_method.md` | Method(=評価枠組み) | 仕様構築、効果器、統制、指標、局在性測定器、判別木、再現 |
| `04_experiment.md` | Experiments | 確定数値の表(EFなし)、統計、進行中の空欄 |
| `05_discussion_limitations.md` | Analysis / Limitations | 3層分解、免許規則、attenuation、限界 |
| `06_pipeline_and_theory.md` | 理論とパイプライン状態 | 条件付けvs介入の理論(⚫の根拠)、実験の状態表 |
| `axbench_testdata.md` | 付録/実験 | AxBenchテストデータの正体と再現・相互評価設計 |

数値の一次出典: `PAPER_OUTLINE.md`(🔵/⚫節と§6x台帳)、`runs/tables/`、
`runs/paper_metrics/report.md`、`runs/ll_repro/report.md`、`RELATED_WORK.md`。

## 🔴 全章共通の規則(違反すると主張が壊れる)

1. **⚫ EF系の数値・記述を論文に使わない**(routed 0.2839、ef32 0.2237、
   λ-IoU 0.74、M0のk掃引、P-BのFRC条件付け10×崩壊などすべて)。
   これらの結論はEF非依存の実験が引き継ぐ: P-B → **P-J**、M0 → **B-1
   介入k掃引**、内容混入の証拠 → **S_min組成分析**。
2. **免許規則**: 「事例レベルkで編集成功 → featureの表現はk本」と書かない
   (操作ハンドルには事例内容が混入)。「対応はtop-3より広い/別物」を
   書けるのは (形1) 現象レベルr掃引(P-J、因果的定義への取替を明示)
   (形2) S_min事例横断安定核×FRC3比較、の2形のみ。
3. judge評価は「**reliability**を測る」(品質/validityは禁止 — Norman et al.)。
4. R5の禁止事項: 「介入ベースのSAE評価は初」(SAEBench/RAVEL先行)、
   「因果基準でlatent選択は初」(Beyond Input Activations先行)、
   「言語minimal pairの因果介入評価は初」(CausalGym先行 — 差分は評価器
   [挙動→テキスト編集]と対象[学習特徴化→SAE latent])。
5. 単位は「言語現象」、編集は「最小対変換(minimal-pair transformation)」。
6. LinguaLensのFRCは論文とコードで定義が違う(条件付き vs 周辺)—
   我々はコード側に忠実、と1文明記。

## 数値クイックシート(論文に載る確定値のみ)

| 主張 | 数値 | 状態 |
|---|---|---|
| **C1' 仕様2×2(499)** | 我々の仕様×steer **0.2337**(0.2385@499)/×clamp 0.1743 vs LinguaLens完全版 **0.0160**・AxBench完全版 **0.0701**(FRC×steer 0.0822) | 確定・主claim |
| 床と統制 | raw 0.0601 / recon 0.0100 / random 0.0521 | 確定 |
| 動作点の崖 | steer α 0.25/0.375/0.5/0.75 = 0.098/0.176/**0.2385**/0.18; clamp 5/10/20 = 0.06/0.17/0.03 | 確定 |
| P-N(彼らの指標) | E_abl 我々 +0.631/+0.370 vs 両プロトコル +0.091/+0.115; AxBench score 1.211/1.262 vs 0.570/1.121(raw 1.081) | 確定 |
| P-I WHERE(因果床) | true発火 3.57 vs random 1.08、393/31、p=5.6e-81; WHAT不能 | 確定 |
| P-J r掃引(形1) | FRC r3 0.17/0.00(67/0, p=1.4e-20)→ r64 2.51 単調非飽和; AUROC r1 **ゼロ**(0.86×)なのに検出 mean **0.939** | 確定 |
| FRR(steer行) | FRR 0.7327 / net-FRR 0.4062 / random床 0.3265(GPT-4o); steer自己一致 0.9781 | 確定(EFなし再集計は残) |
| B2(SAE不使用参照) | 0.1242、empty copy 0.4770 | 確定 |
| LinguaLens再現 | FIC 12.0/8.6/13.6/3.0 vs Table2 8.3/22.9/46.9/6.9(弱い行がアンカー) | 確定 |
| 判別木(paper版) | **A 71 / C 21 / B 2 / D 4**(98現象) | 確定 |
| B-1 介入k掃引(操作幅) | k=1..64: 0.068→0.108→0.194→**0.237**→0.239(膝=32、k=1は床)、random平坦~0.05 | 確定 |
| B-2 S_min+安定核(形2) | \|S_min\| median **5**(vs 仕様98.7、S_min=1例あり); 安定核 20/22 非空・核∩FRC3 **8/22**・AUROC1は核に頻出 | 確定 |
| AxBenchアンカー | 2B: SAE 0.177/0.151、SAE-A 0.166/0.132(L10/L20) | 再現実行待ち |

## 進行中・待機(執筆時は空欄/暫定)

- **AxBench再現 L20/L10** + ll_set10(相互評価セル)→ judge
- FRR表のEF行なし再集計(オフライン)、clamp腕のFRR判定(任意)
- random辞書対照(情報軸を復活させる場合のみ)、BLEU/chrF、定性例
