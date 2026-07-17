# 02 — Previous Works(Related Work)執筆資料

出典: `RELATED_WORK.md`(英語草稿v2 = §D、全主張が一次資料PDFで検証済み)。
日本語で書く際も**構成・主張・引用は §D をそのまま翻訳・要約**すればよい。
以下はその骨格と、日本語化にあたっての要点・地雷。

## 0. 章の背骨 — 2軸分類(D.0)

| 手法 | WHERE | WHAT | 条件付け信号 |
|---|---|---|---|
| LinguaLens (Jing+ 2025) | 選択なし(全位置一律・定数クランプ 0/10) | 連続 | SAE特徴(現象あたり**3本**) |
| AxBench SAE/SAE-A (Wu+ PMLR v267) | 選択なし | 連続 (h+αw) | SAE特徴(概念あたり**1本**) |
| ActAdd / CAA / RepE | 選択なし | 連続 | 対照ペア由来の方向 |
| Levenshtein Transformer (Gu+ 2019) | 編集目標から推論 | **離散** | source のみ |
| LEWIS (Reid & Zhong 2021) | 編集目標から推論 | **離散** | source + スタイル分類器attention |
| Susanto+ 2020 / EDITOR (Xu & Carpuat 2021) | 編集目標から推論 | **離散** | **表層の用語辞書/語彙選好** |
| **本研究** | 編集目標から推論(λ-IoU 0.74) | **離散** (INS/DEL/SUB) | **SAE特徴**(事例あたり k=32) |

空白は「離散 × SAE特徴条件付け」のセル。左下(離散編集)は条件付けが表層に
留まり、右上(SAE介入)は作用が連続に留まる。

## 0b. 3つの選択方式の対比表(D.0b — 実験節でも使う中心表)

| | 本研究 | LinguaLens | AxBench |
|---|---|---|---|
| 選択の単位 | **この1ペア(事例)** | この現象(コーパス集約) | この概念(コーパス集約) |
| 集約 | **なし** | 平均(EALE = (1/N)Σ τ_k) | 平均/AUROC |
| 活性の使い方 | 連続・符号付き delta `z_tgt−z_src` | **二値**(発火したか)— 大きさを捨てる | 連続 max-pooled |
| 測る場所 | **編集スパン内のみ**(local) | 文中どこでも | passage全体 |
| ラベル | **不要** | sentence1/sentence2 | SAE-A: 要 / vanilla: 不要 |
| 選択基準 | delta の大きさ top-k | \|EALE\|>75%ile → FRC = H(PS,PN) | vanilla: 概念出所latent / SAE-A: AUROC最大 |
| 介入に使う本数 | **32**(事例) | **3**(現象) | **1**(概念) |

**非対称は5つあり、どれもkの値とは無関係**: (1) 集約しないのは我々だけ —
LinguaLensの原始量 τ_k(s)=a(1)−a(0) は我々のdeltaそのものであり、彼らは平均し
我々は平均しない(→ P-Bは「集約するか否か」のablation)。(2) 大きさを捨てる
のはLinguaLensだけ。(3) 編集スパンに局在させるのは我々だけ。(4) ラベル不使用
は我々だけ。(5) AxBench vanillaは「探索」をしていない(概念の出所latentを
そのまま使う)— 0.695 は選択失敗ではなく「出所latentが自分の概念を検出でき
ない」という数字。

**🔴 LinguaLensは論文とコードが食い違う**(公式repoで確認): 論文のPS/PNは
条件付き確率、repoの実装は周辺比率。minimal pairでは独立性が成立しないため
両者は一致しない(例: 両文で発火する話題特徴が半数にあると論文定義FRC=0、
コード0.5)。**我々のP-B/P-Jはコード側に忠実**(人々が実際に走らせる成果物を
再現)。彼らのdo記法をそのまま引かず、「実装は周辺比率であり因果量ではない」
と1文書く。

## D.1 — SAE特徴: 検出は指令ではない

- 系譜: SAE辞書学習(Cunningham/Bricken/Templeton)→ Gemma Scope(本研究の
  重み)→ LinguaLens(言語現象がこの空間で回収可能; FRC=PS/PNの調和平均、
  EALE 75%ile事前フィルタ、GPT-4oでtop-10検証)。
- **決定的な指摘**: LinguaLensはその特徴を**活性空間のノブとしてのみ**使う
  (ablation=0 / enhancement=10 のクランプ、出力は自由生成)。検証済み逐語:
  "In the ablation experiment, we set the target feature's activation to 0,
  and in the enhancement experiment, we set it to 10."
- LinguaLensの同定は**構成上無競争**: 検出ベースライン無し、AUROC/AUC/F1は
  論文中0回、PS/PNはminimal pair上(最も易しい識別)。AxBenchは反証可能に
  設計されている(12手法・対応t検定)からこそ vanilla SAE 0.695(12中11位)
  と「教師あり選択の価格」(0.695→0.917)を出せる。→ 分野には言語SAE特徴の
  **存在主張**(LinguaLens)と単一latent検出の**比較主張**(AxBench)がある
  が、「同定された特徴が**行動の根拠に足るか**」の評価が無い。そこがP-B。
- LinguaLensにおける「編集」はデータ構築側のみ("produced through a minimal
  edit that deletes or substitutes the trigger")。
- **反例なし(確定)**: "No prior work, to our knowledge, uses SAE features
  as the conditioning signal for a model that emits edit operations."
- Amnesic probing(Elazar et al., TACL 2021)= P-Bの系譜。"probing
  performance is not correlated to task importance"。**ヘッジ3点**:
  (1) "the canonical" でなく "a canonical" と書く(先行にHewitt & Liang 2019、
  Ravichander 2021、Belinkov 2022)。(2) 彼らは「モデル自身が使うか」、我々は
  「我々が外部指令として使えるか」— 隣接する問いで、精神の拡張であって
  文字通りの適用ではない("extends the spirit rather than the letter")。
  (3) P-Bの10×はexactでの測定であり、amnesic probingのtask lossとは別の量。

## D.2 — 懐疑的証拠と、それが我々に届かない理由

- AxBenchの否定的結論は **steering軸に閉じている**: "even at SAE scale,
  representation steering is still far behind simple prompting and
  finetuning baselines"。手法集合に編集手法は無い。
- **AxBench自身が detect-then-intervene の脱連関を実証**(SAE-A): 検出最良
  latentをAUROCで選び("select the highest-scoring feature by this metric")、
  同じlatentのdecoder方向でsteer("adding their decoder features directly to
  the residual stream")→ 検出 0.695→**0.917**(+32%)、steering
  0.165→**0.157**(−5%)、winrate 48.8%。著者自身: "**better classification
  does not directly lead to better steering**"。Probe(0.940/0.098)、SSV
  (0.912/0.026)も同型。→ **P-Bの独立再現(別modality)として引く** —
  反論ではない。我々への免罪符にもしない("We do not claim AxBench licenses
  our conditioning")。
- 検出軸の弱さ(0.695)は我々の限界の住処: 本手法は概念→latent引き当てを
  一切行わない(条件付けは観測された特徴差分のtop-k)が、**運用時に仕様を
  概念から作る問題はこの0.695の世界に落ちる**(→ Limitationsで正面から書く)。
- その他: Kantamneni et al. (ICML 2025) SAEプローブはロジスティック回帰に
  一貫して勝てない(ただし「一貫した優位の不在」であり個別勝ちはある)。
  frozen/random SAE (arXiv:2602.14111): 見出しの0.73はcosine球内で学習する
  Soft-Frozen、完全ランダムは0.55-0.62 < 学習済み0.72-0.74。DeepMind自身が
  Gemma Scope公開時に「公平なベースラインに勝つか」を未解決問題として列挙。

## D.3 — 離散編集: 操作は貢献ではない

- 定義枠: Malmi et al. NAACL 2022 tutorial("predicting edit operations
  which are applied to the inputs")。
- 語彙の先行: LevT(挿入・削除を原子化)、LaserTagger、LEWIS(名前の由来;
  insert/keep/replace/delete)。**"We claim no novelty in the edit
  vocabulary."**
- **最近接の先行 = 制約付きLevT系譜**: Susanto et al. ACL 2020(Wiktionary/
  IATE用語辞書を推論時注入)、EDITOR(TACL 2021、"specify preferences in
  output lexical choice")。彼らも**仕様で離散編集を条件付けている**。
  → 差分は仕様の**種類**: 表層の用語辞書 vs **モデル内部から回収した特徴辞書**。
  "The contribution is the *kind* of specification, not the existence of one."
- **Edit Flowsとの関係(過剰主張しない)**: 生成機構はEdit Flows(Havasi et
  al. 2025)のCTMC。**rate×tokenの因子分解 u_t = λ·Q は式(13)の継承**であり
  我々の2ヘッドはその積そのもの。"decoder" は彼らの論文に0回 — opの適用は
  CTMCの状態遷移そのもので学習対象ではない(彼我共通)。我々が変えたのは:
  スクラッチ生成→ソース係留、ランダムアラインメント→最小編集の決定的
  アラインメント、自由レート→hazard解析形、条件付け=prefix/画像/CFG→
  SAE特徴トークン。

## D.4 — Steeringの入力毎の不安定性

- Tan et al. (NeurIPS 2024): "steerability takes on a large range of values
  across different inputs, including negative values"(anti-steerability)。
- 我々の対比: 彼らは**入力毎**(データセット平均は正のまま)、我々の負の
  per-phenomenon net-FRR(past_tense −0.13*、expressive −0.50、
  subject_verb_inversion −0.30)は**集約平均が負** — 同じ失敗のより鋭い形。
  (*数値は04の per-feature 表から引くこと)
- 彼らはCAA/Llama-2/多肢選択でSAE無し → 「現象の引用」であってSAE steering
  の測定としては引かない。**OOD側の主張(プロンプト変更に脆い)は引用禁止**
  (検証で棄却済み)。

## D.5 — Judgeを裁く(新規性は限界コストのみ)

- 先行(すべて実在・確認済み): Sage(ラベルフリーjudge評価)、Shi et al.
  Repetition Stability(同一クエリ反復)、Haldar & Hockenmaier "Rating
  Roulette"(intra-rater Krippendorff's α)、Wang et al. Conflict Rate
  (順序入替)、Zheng et al.(順序ランダム化)、Norman et al. MVVP。
- **我々の主張はここだけ**: 先行は全て**専用の再実行予算**(2〜20×)で重複を
  人工的に作る。我々は**評価データ内に自然発生する exact 一致ペア**から
  追加コストゼロで自己一致率を回収する(exact一致ペアでは judge(src,out) と
  judge(src,tgt) が同一比較になるため)。
- degeneracy反論(常にAを選ぶjudgeが満点)への免疫: gold順とsystem順を
  **独立のrngストリーム**から引くため、always-A judge は 0.50 に落ちる
  (結合していれば1.00だった — 実際バグ下のgemmaがその退化を示した)。
- **(c-4) attenuation は新規性が残る**: 対応のある差は Rogan–Gladen 反転で
  `p_A − p_B = (θ_A − θ_B)(q0+q1−1)` — スケールされるがシフトしない。
  前提=非差異的誤分類で、**per-system自己一致率がその検査**(1つの測定が
  judge選定根拠と妥当性検査の2役)。Chen et al. (arXiv:2601.05420) を引く。

## D.6 — 必要編集量によるルーティング

- 先行: RouteLLM(学習ルータ)、MoECE(エラー型でゲート)、ESC、APR。
  近い順に: **AdaEdit**(Cheng et al., Findings of ACL 2026)— diff/全文再生成
  を事例毎に切替、ただし**教師ありFTで暗黙内在化**。**CAST**(Lee et al.,
  ICLR 2025)— steeringの推論時ゲート自体は先行、ただし基準はプロンプトの
  意味カテゴリ・両枝ともAR生成・ラベル付きグリッドサーチ。
- 我々の差分は**着想ではなく条件付け信号**: count-rule は「離散編集器自身の
  λ場が発火させたハンク数」= **明示的・教師なし・自己言及的**な編集規模
  推定器。ルータの学習なし、ラベルなし、検証集合でのフィット無し。

## 🔴 棄却リスト — 書いてはいけないこと(X / X-2 / 軸4より)

- 「LinguaLens自身がSAE介入の信頼性の低さを認めている」(有利すぎる論拠)。
- 「Edit Flowsの語彙はちょうどINS/DEL/SUB」「LaserTaggerはちょうど3種」等の
  語彙の断定(要再確認のまま棄却)。
- Tan et al. のOOD側主張。
- 「LEWISはSAE解釈可能性の系譜全体に先行」(Yun+ 2021, SPINE 2018が先行。
  「Gemma Scope(2024)とLLM-SAE系譜(2023)に先行」と狭く書く)。
- 「LEWISには言語的特徴条件付けが一切ない」(スタイル分類器attentionは
  学習された帰属信号 — 最近接の隣人。差別化は**辞書としての解釈可能性**)。
- 「AdaEditはフォーマット切替であってルーティングではない」「AdaEditは
  コスト効率基準だから対象外」(どちらも0-3棄却 — 逃げられない)。
- 「EdiT5は経路分岐を持つ」(事実誤り: 全入力が無条件固定順)。
- 「CASTより先にsteeringゲートを付けた」(CASTが先行)。
- 「SAEの不振はNeuronpediaラベルのartifact」(強い版は棄却; AxBenchの留保は
  "at least partially" 止まり)。
- 「judge品質を測れる」「人手ラベル不要のjudge評価は新規」「反復一致率は
  新規」「順序ランダム化は新規」。

## 手続き規則(執筆時)

1. **逐語引用は全て自前のPDF抽出で再確認**(要約器の捏造を計9件検出済み)。
   手順: `curl -sL -o x.pdf <URL>` → pypdf全ページ → NFKD正規化+リガチャ
   展開+ハイフン折返し除去 → 自分で regex 検索。
2. grepの0ヒットを信じない(リガチャ偽陰性)。
3. 「調べていない」と「探したが無い」を区別: 「探したが無い」と書けるのは
   **(a)(b)(d)のみ**。(c)は先行あり。
4. 投稿時に版番号・題名を再確認(LinguaLensは旧題と現題が異なる。Tan v6、
   Edit Flows v3、AxBenchはPMLR v267 camera-readyの数値を引く)。
   AxBenchの掲載venue(ICML/PMLR表記)は投稿前に最終確認(残タスク)。
