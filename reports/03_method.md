# 03 — Method 執筆資料

出典: `PAPER_OUTLINE.md` §3、`EDIT_FLOWS_ZERO.md`(S系列の設計と判定)、
`editflow.py` / `train_editflow.py` / `decode_flow`。

## 1. タスク定義

- 入力: ソース文 x0 と **特徴デルタ仕様** (z_amp, z_sup)。
  仕様は `diff_intervention(z_src, z_tgt, k_top=32)` = SAE(x0) と SAE(x1) の
  差分 `delta = z_tgt − z_src` の top-k を符号分解したもの
  (正 = amplify、負 = suppress)。
- 出力: 編集後の文 x1(離散編集操作 INS/DEL/SUB の系列を適用)。
- **保存則**: empty 仕様 → 無編集(no-edit)を構造的に要求(null-record教師)。
- 評価は**ゼロショットOOD**: 学習は corruption キャッシュのみ、評価は
  LinguaLens minimal pairs(学習で一切見ない)。

🔴 執筆注意: 仕様は「事例レベル」= 実際の (src, tgt) ペアから計算した特徴差分。
運用時に target は無いので「仕様を概念から得る」問題は残る — Limitations で
正面から書く(05参照)。主張の射程は「**特徴レベルの仕様が与えられたとき、
離散編集が steering より良くそれを実現する**」。

## 2. アーキテクチャ(すべて凍結ベース + LoRA)

| 部品 | 実体 | 役割 |
|---|---|---|
| バックボーン | **凍結 Gemma-2-2B** + LLM2Vec双方向化 + **LoRA r=32** | 編集器のエンコーダ(双方向attention — 右文脈を見る) |
| SAE | **Gemma Scope layer-12 / 16k(JumpReLU)**、凍結 | 特徴抽出(仕様の定義域)。blocklist 32特徴を除外 |
| 条件付け | **feature-tokens**(下記) | 仕様の注入 — 活性は書き換えない |
| rate head | 位置ごと λ^{ins,del,sub} | WHERE + 操作種(hazard分解、下記) |
| token head Q | **凍結 lm_head** + logit-lens バイアス(W_U·W_dec、λ_lens=1) | WHAT(挿入・置換の内容)。SAE幾何の事前分布として再解釈 |

### 条件付け経路(🔴 論文の枠組みの根拠 — 正確に書く)

指令特徴1つにつき1トークンを **層0(入力)の prefix** として与える:

```
prefix = [F(f1,±,v1), …, F(fj,±,vj)]
F(f,±,v) = RMS較正(W_dec[f]) + sign_emb(±) + mag_mlp(log(1+v))
```

- W_dec[f](SAEデコーダ行)を埋め込みスケールにRMS再正規化し、学習される
  cond_scale を掛けて入力トークン化。**凍結Gemmaの活性はどの層でも一度も
  変更されない** — クランプもベクトル加算もしない。因果的には
  `P(edit | Z=z)` = **条件付け(証拠)**であり `P(Y | do(Z=z))` ではない。
- attentionが**個々の特徴と個々の編集サイトを直接束縛**できる(プール型
  2ベクトル条件付けの feature-bag binding 問題の正面解 — S2で採用確定)。
- CFGドロップアウトはトークン集合単位(全ドロップ = empty教師)。

## 3. Edit Flow 機構

### 継承と差分(D.3と整合させる — 過剰主張しない)

- **継承**: Edit Flows(Havasi et al., arXiv:2506.09018)のCTMC離散フローと
  **rate×token の因子分解**(式13: `u_t(ins(x,i,a)|x) = λ_{t,i}(x)·Q_{t,i}(a|x)`)。
  我々の2ヘッドはこの積そのもの。opの適用は状態遷移であり学習対象ではない。
- **差分**: (1) スクラッチ生成 → **ソース係留の編集regime**(x0=入力文、
  x1=目標文; 論文はX₀=∅か一様ランダムのみで、編集regimeは本研究の外挿)。
  (2) ランダムアラインメント → **最小編集の決定的アラインメント**
  (経路長=編集距離のとき自然)+ corruption が生成した演算列そのものを
  教師にする真アラインメント。(3) 自由レート → **hazard解析形**(下記)。
  (4) 条件付け prefix/画像/CFG → **SAE特徴トークン**。

### hazard 分解(較正の構造化)

編集regimeでは κ(t)=t³ が既知で各opは一度だけ発火するため、目標レートは
厳密に w(t)·1[pending](w(t) = 3t²/(1−t³))。よって

```
λ = w(t) × sigmoid(head(h))
```

と分解し、hazard因子は解析的に与えて **確率因子 P(pending|x_t) だけを学習**
させる。強度追従は構成上厳密(較正問題が消滅)、p は確率なので発火閾値が
自己較正され、**thr{F} デコード = p ≥ F** が文字通りの動作点になる
(S2実測: mean p が t によらず 0.50/0.48/0.46 と安定)。入力非依存のレート
経路が存在しないため premise 安全(empty→no-edit)。

### 局所化(Localized CTMC、Edit Flows 付録C.1 の編集regimeへの翻訳)

- 訓練: 各opの自己発火時刻 t* = u^{1/3}(κ⁻¹)、発火済みソースから
  Pois(λ_prop·Δt) で左右近傍へ被覆伝播。損失重みは
  λ_eff = w(t) + λ_prop·(編集済み隣接数 adj)。hazardベースを
  b = w(t)+λ_prop·adj に拡張して λ = b·p を維持(adj はゼロ初期化埋め込みで
  エンコーダにも供給 — warm-start 厳密)。λ_prop=4。
- デコード: 編集隣接サイトの発火バーが p ≥ F·w/(w+λ_prop·adj) に低下 —
  **一箇所編集すると隣の発火バーが下がる** = 統語変形が要求する協調的
  多点編集の事前分布。編集痕跡は decode_flow が `edited` リストで追跡
  (apply_step_ops が同期更新)。
- 効果(S3→S4確定): 4-8編集バケット 2.4×、9+ で唯一の命中 — 局所性の配当。

### デコード

thr{F}(Fが唯一のノブ)+ greedy Q、48 steps。本番はS3単一ckpt:
**thr0.1 = exact最大、thr0.5 = バランス**(S4のペア統計で確定)。
CFG・温度・best-of-K は「較正済みλを前提とする道具」で、hazard以前の
パイロットでは全敗(S0)— 採らない。

## 4. 学習データ(corruption cache)

- dolma文に **25 family の依存構造述語ベース変換**(PARTICLE/DATIVE/ADVPLACE/
  PPFRONT/QUOTINV/CLEFT等の並べ替え系を含む)+ MLM語彙corruption を適用し、
  round-trip 検証と対称SLORフィルタで文法性を担保。SAE blocklist 適用。
- **null-record 教師**: empty仕様 → no-edit を構造的に教える(前提保護の源)。
- 真アラインメント教師: corruption が適用した演算列をそのままスロット列に
  符号化(difflib事後推定の誤アラインを排除)。
- **v7 top-up(進行中のS6で使用 — 執筆時は暫定と明記)**:
  P4 = SPLITINF family(分裂不定詞幾何 "to ADV V"↔"to V…ADV")、
  P5 = **mismatched-z null教師**(確率0.12でレコードを「x_t=x0、gold=no-edit、
  条件付け=他ペアの実delta」に振替 — 「一致しないzでは黙れ」を教える欠落
  コントラスト)、語彙多様化(MLM top-k 8→24、seed 777)。12,000レコード。
  S6がGOなら本文へ、NO-GOならS3の記述のまま。

## 5. routed システム(C0のヘッドライン)

**count-rule T=1(完全教師なし・事前登録・凍結)**:

> EF自身のλ場が発火させた編集ハンク数が **≤1 なら EF(ef32)の出力**、
> それ以外は **steer0.5(介入fallback)** の出力を採用。

- ルータの学習なし・ラベルなし・検証集合フィットなし。「このペアにどれだけ
  編集が要るか」の推定器が**編集器自身のλ場**(自己言及的)。
- λ場は「形態=1ハンク / 構造=多ハンク」を教師なしで見分けている
  (per-feature表で routed ≈ max(EF, steer) を回収 — 04参照)。
- 🔴 表現規則: routed は「**条件付けと介入の、教師なし規則による相補的結合**」。
  「介入の結果」と呼ばない。steer fallback の差し替えは別システム扱い
  (未接触標本での再確認が要る — gain-router 0.3012 を昇格させなかった規律)。

## 6. 実装の細部(付録向け)

- SAE抽出: `sae_z_with_offsets`(トークンオフセット付き活性)、
  local scope = 編集スパン内トークンのみで delta を取る。
- 学習: 50k〜100k steps、warm-start系列(S1 hazard → S2 feature-tokens+真ア
  ライン 100k r=32 → S3 localized 50k)。k_top=32、k_amp/k_sup サンプリング、
  empty_conditioning_prob(CFGドロップ)。
- 推論コスト: 単一モデル1パス(タガー・列挙・ランカー・refine なし)。
  対比: v6パイプラインは4段カスケード。
- 再現性: 全システムが records.jsonl 形式で出力 → `compare_ef_pipeline.py`
  が matched-pair 統計まで一括。

## 7. この章の地雷

- 「hazard分解・局所化は我々の発明」と書かない — 局所化は付録C.1の翻訳、
  因子分解は式13の継承。**我々のものは: 編集regimeへの外挿、hazardの解析形、
  真アラインメント教師、SAE特徴トークン条件付け、count-rule ルーティング**。
- 「編集モデルの提案」として書かない — 「SAE特徴を仕様として実現する」話
  (C2の土俵設定)。編集語彙の新規性主張は即死(02参照)。
- λ-IoU の優位(0.74 vs タガー count-oracle 0.7472 はパリティ)は「知識」では
  なく「決定経路」(単一の較正可能スカラー vs 多段カスケードの決定損失)。
