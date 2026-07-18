# 公式実装忠実性の監査(2026-07-18、EF_LM_LOSS_PLAN §2)

対象: THU-KEG/LinguaLens(+OpenSAE)、stanfordnlp/axbench の公式コードと
我々の再現・移植コードの逐語照合。

## 1. LinguaLens(lingualens/intervener.py + OpenSAE)

| 項目 | 公式 | 我々 | 判定 |
|---|---|---|---|
| 介入mode | set(enhancement=10.0 / ablation=0.0)、prompt_only=False | 同 | ✅ |
| set意味論 | activeな対象latentは値で上書き、**非activeはminスロット強制置換**(OpenSAE `_apply_intervention_add_or_set`) | SaeClampHook 同実装 | ✅ |
| control | multiply×1.0(intervention_indices付き)= SAE再構成パススルー | `recon`モード 同 | ✅ |
| 生成 | 素プロンプト(chat templateなし)、temperature=1.0、do_sample、max_new_tokens=100 | 同 | ✅ |
| 試行数 | repo既定 num_generations=10(batch既定5)。論文は50 | n=5採用(ユーザー決定)は**repo batch既定と一致** | ✅ |
| **judge対象テキスト** | `tokenizer.decode(generated_ids[0])` = **プロンプト込み全文** | v1は継続のみをdecode | 🔴 **不一致 → 修正済み**(eval_ll_repro_gen.py、2026-07-18) |
| judge | 論文プロトコル(repoに判定コードなし)、**GPT-4o** | gpt-4o | ✅ |
| FIC式 | E_abl=(Pt−Pb)/Pt、E_enh=(Pt−Pb)/(1−Pb)、負値はw=0.5で符号反転算入、調和平均(App. E.2) | judge_ll_repro.py(Table 2検算済み) | ✅ |
| FIC原実験の範囲 | **5 feature × 手書きプロンプト1本(App. D.1)× リサンプル** — LinguaLens-Data不使用 | 全feature×実ペア版は「指標のデータセット拡張」と明記する | 📌 論文での書き分け必須 |

## 2. AxBench steering(axbench/models/interventions.py)

```python
# AdditionIntervention.forward (verbatim):
steering_vec = subspaces["max_act"] * subspaces["mag"] * self.proj.weight[subspaces["idx"]]
output = base + steering_vec   # 全位置にブロードキャスト
```

- **h + factor(mag) × max_act × W_dec[latent]、全位置** — 我々の再現
  (eval_axbench_repro_gen.py AdditionHook)と一致 ✅
- 生成: temperature=1.0、do_sample、eval_output_length=128 ✅
- A2腕(LinguaLens-Data移植)の設計: この加算機構を事例レベル仕様の
  dvec に適用(h + α·dvec 相当)。強度規約の対応は α ↔ factor×max_act。

## 3. AxBench prompting(models/prompt.py + utils/dataset.py)

公式の組み立て(逐語確認):
1. **steering promptをgpt-4o-miniに生成させる**(meta-prompt
   T_GENERATE_PREPEND_STEERING_PROMPT: "Direct the model to include
   content related to {CONCEPT} ... even if it doesn't directly answer
   the question")
2. `steered_input = f"{steering_prompt}\n\nQuestion: {instruction}"`
3. chat template適用 → temperature=1.0、128トークン生成

**A3腕(LinguaLens-Data移植)の設計案(要ユーザー確認)**:
- enhancement: 彼らのmeta-promptそのまま、CONCEPT=featureの用語+gloss。
  `steered_input = f"{steering_prompt}\n\nQuestion: {src文}"` + chat template
- ablation: 彼らのrepoに抑制用テンプレートが無いため、meta-promptの
  Objective行を "avoid content related to {CONCEPT}" に置換(最小改変、
  論文に明記)
- 生成は彼らの既定(temp 1.0、128tok)

## 4. 修正済み事項(このコミットまで)

- eval_ll_repro_gen.py: judge対象をプロンプト込み全文に(§1の🔴)
- judge_axbench_repro.py: parse_ratingを公式パーサ等価に(gpt-4o-miniは
  "Rating: 0" と裸で回答 — [[x]]必須の旧regexが全ゼロの根本原因だった)
- train_ef_editor.py: max_steps境界のループ脱出ckpt保存(probe100ゲートが
  無言でスキップされていた)
