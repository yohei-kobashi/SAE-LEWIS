# AxBench のテスト用データと再現・相互評価計画(2026-07-17、公式repo精読)

出典: github.com/stanfordnlp/axbench(コード逐語確認)+ arXiv:2501.17148。
実装: `scripts/eval_axbench_repro_gen.py` / `scripts/judge_axbench_repro.py`。

## 1. AxBench の steering テストデータの正体

**0.157 の内訳**: Table 2 の SAE-A **4構成平均**(gemma-2-2b L10 **0.166** /
L20 **0.132**、gemma-2-9b L20 0.186 / L31 0.143)。SAE(vanilla)は
0.177/0.151/0.191/0.140 → 平均 0.165。**我々のアンカーは2B列**(同一モデル・
同一SAE系列で再現可能): SAE 0.177/0.151、SAE-A 0.166/0.132。

テストデータは3つの部品からなる:

1. **概念リスト**: `axbench/concept500/prod_2b_l{10,20}_v1/generate/metadata.jsonl`
   — 500概念。各行 = 概念テキスト(例 "references to rental services and
   associated equipment")+ Neuronpedia ref(**その概念の出所latent ID** —
   vanilla SAE 腕はこのlatentをそのまま使う。探索なし)。
2. **命令(テスト入力)**: **AlpacaEval**(tatsu-lab/alpaca_eval.json)から
   **概念ごとに10命令**を `df.sample(10, random_state=concept_id)` で
   決定的にサンプル(コード逐語 `utils/dataset.py`)。gemma-2-2b-it の
   chat template を適用(add_generation_prompt、BOSは剥がす — 生成時に
   tokenizerが再付与)。
3. **検出用ラベル付きデータ**: `concept500/*/inference/latent_eval_data.parquet`
   (概念ごと positive 36 / negative 36 + hard negative)— SAE-A の
   AUROC選択と、steering強度の max_act 集約に使う。

**生成条件**(`predict_steer` + `sweep/wuzhengx/2b/*/no_grad.yaml` 逐語):
temperature 1.0 サンプリング、max_new_tokens 128、介入は
`h + factor × max_act × W_dec[latent]` を **layers[L].output の全位置**
(プロンプト+デコード)に加算。factor格子 =
[0.2, 0.4, …, 2.0, 2.5, 3.0, 4.0, 5.0](14点)。

**評価プロトコル**(`evaluate.py`、winrate_split_ratio=0.5):
- 指標 = **gpt-4o-mini** judge の3ルーブリック(concept / instruct /
  fluency、各0-2、"Rating: [[x]]")の**調和平均**。プロンプトは
  `evaluators/prompt_templates.py` 逐語(我々のjudgeスクリプトに逐語移植済)。
- **命令10本を半分に割り、片方でfactor選択、残り半分(holdout)で採点**。
  概念ごとにbest factorを選ぶ(= 彼らに最大限有利な設定でこの数字)。

**max_act の出所**: 主経路は Neuronpedia API の maxValue(要APIキー、
0以下なら50に置換)。**repo公認の代替**(`disable_neuronpedia_max_act`、
`gemmascope_axbench_max_act.yaml` 変種)= AxBench自身のデータセットから
最大活性を計算 — 我々はこちらを使う(meta.jsonに明記)。

**SAE-A の選択**: 「真ラベル付きデータセットでAUROCを計算し最大のlatentを
選ぶ」(論文逐語)。ラベル付きデータはrepoに同梱(上記parquet)なので、
Mann-Whitney平均ランクAUROC(我々の `select_features_auroc.py` と同じ数学、
sanity gate 0.939 で検証済みの実装)で再計算する。

## 2. 実装した再現の腕(`runs/axbench_repro/`)

| 腕 | 介入 | latent | anchor(2B L10/L20) |
|---|---|---|---|
| `sae` | addition steering(彼らの機構) | 概念の出所latent(vanilla) | 0.177 / 0.151 |
| `sae_a` | 同上 | AUROC argmax(彼らのSAE-A) | 0.166 / 0.132 |
| `ll_set10` | **LinguaLens機構**: set-10 + SAE再構成完全置換・全位置 | vanilla latent | —(相互評価の新セル) |

- デフォルト規模: 先頭100概念(500でフルスケール; resumeで拡張可)。
  概念×10命令×14factor×腕 = 100概念で腕あたり14,000生成(ll_set10は
  factor無しで1,000)。
- アンカーの読み: **桁一致ではなくパターンとレンジ**(彼らはNeuronpedia
  maxValue・500概念、我々はデータセットmax_act・N概念)。ただし
  モデル・SAE・層・プロンプト・judge・分割は同一なので、LinguaLens再現
  (モデル自体が違う)よりはるかに強いアンカー。

## 3. 提案手法(SAE-LEWIS)の同テストデータ評価 — 設計(未実装)

- **腕 `ours_ef`**: 無介入応答(同一シードでsteeringなし生成)を、EF編集器で
  spec = amplify {latent: v = factor × max_act} を条件に**離散編集**して
  概念を組み込む。彼らの指標(concept/instruct/fluency)でそのまま採点。
- 位置づけ: 「介入せずに条件付き編集で同じタスクを解く」= 我々の2×2の
  空白セルを彼らのベンチで測る。P-N での b3_ours 1.262 ≫ ax_protocol 1.121
  と同じ主張構造の、彼らのテストデータ版。
- 実装は **S6決着後**: チェックポイント読み込み・decode_flow 呼び出しは
  `scripts/prune_spec.py` から**逐語コピー**する(署名の発明は過去に事故
  多数 — 手続き規則)。応答128トークンはEFの学習長よりOODである点は
  結果に明記する。
- 注意: EFの条件付けは事例レベルdelta(k=32)で学習しており、単一latent
  仕様は P-J/P-B で測った「現象レベル仕様」に近い難しさを持つ可能性が
  高い。これは弱点の隠蔽ではなく、**AxBenchの土俵では検出latent 1本しか
  仕様が与えられない**という制約の実測になる(C1'の裏面)。

## 4. LinguaLens の同テストデータ評価

`ll_set10` 腕(実装済み)がそれ: 同じ概念latent・同じAlpacaEval命令・
同じjudgeで、彼らの set-10 介入がAxBenchのsteeringとどう並ぶか。
逆方向(AxBench steeringをLinguaLensのminimal pairタスクで測る)は
P-K/P-N で実施済み(ax_protocol 0.0701 / E_abl +0.115 / AxBench score
1.121 ≈ raw 1.081)。**これで相互評価マトリクスが両方向とも埋まる**。

## 5. 実行順(ユーザー指示: S6優先)

1. S6 学習完走 + probe500 ゲート判定(interact-g、進行中)
2. LinguaLens再現の生成 `run_ll_repro.sh`(interact-g 1セッション)
   → prepost で `run_ll_repro_judge.sh`
3. AxBench再現の生成 `run_axbench_repro.sh`(interact-g、L20から。
   resumeで複数セッション)→ prepost で `run_axbench_repro_judge.sh`
4. S6勝者確定後: `ours_ef` 腕の実装+生成、L10構成、必要なら500概念へ拡張
