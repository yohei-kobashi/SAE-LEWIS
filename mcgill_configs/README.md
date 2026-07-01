# McGill-NLP/llm2vec training configs (Gemma-2-2b)

These JSON configs are consumed by the vendored McGill training scripts
(`vendored/mcgill_llm2vec/experiments/run_mntp.py` /
`run_simcse.py`) via `scripts/train_mcgill_llm2vec.sh`.

## Provenance

Adapted from upstream `train_configs/mntp/Sheared-Llama.json` and
`train_configs/simcse/Sheared-Llama.json` in the McGill-NLP/llm2vec
repo (same-scale published recipe: Sheared-LLaMA-1.3B → we mirror
for Gemma-2-2b at 2B).

## Deviations from upstream

- `model_name_or_path` : `google/gemma-2-2b`
- `output_dir`         : `runs/mcgill_gemma_repro/{mntp,simcse}`
- `attn_implementation`: `sdpa` (not `flash_attention_2`)
  Flash-Attention-2 has no ARM aarch64 wheel in our HPC stack; SDPA
  is the same math, just slower. Gemma's SDPA path respects the
  BiGemma2Model subclass' bidirectional attention.

Everything else — LoRA rank, batch size, LR, mlm_probability=0.2 for
MNTP, simcse_dropout=0.3 for SimCSE, loss_scale=20 (= τ 0.05),
stop_after_n_steps=1000, gradient_checkpointing=true — matches the
upstream Sheared-Llama recipe verbatim.

## Why not the Mistral recipe?

Mistral (7B) uses `mlm_probability=0.8` and `data_collator_type=all_mask`
(replaces 100% of selected tokens with [MASK]) — a much more aggressive
schedule. That's the recipe upstream chose for the 7B scale. At 1.3B
(Sheared-LLaMA) they use the standard 80/10/10 collator at 20% —
which is what we mirror for the similarly-sized Gemma-2-2b.

## No JSON comments

HfArgumentParser rejects unknown keys with
`ValueError: Some keys are not used by the HfArgumentParser: ['...']`.
So the configs themselves must stay strictly recipe-only. All
prose about the recipe lives in this README.

## Sheared-LLaMA-1.3B configs (infrastructure validation)

`mntp/Sheared-LLaMA.json` and `simcse/Sheared-LLaMA.json` are
essentially verbatim copies of the McGill upstream
`train_configs/{mntp,simcse}/Sheared-Llama.json`, with the same
attn_implementation=sdpa override.

Purpose: validate that our training infrastructure (isolated venv,
orchestration, data loading) actually reproduces paper-level
STS-B (~70+) on the exact base LLM the paper used.

Why we run this before touching Gemma-2 subclass writing:
- McGill code natively supports Sheared-LLaMA-1.3B, so training
  runs without any subclass modifications on our side.
- If we get STS-B ~70+, we know the pipeline is healthy — then
  the only remaining work for Gemma-2 is writing the subclass
  itself.
- If STS-B < 60, there's a pipeline bug we need to find BEFORE
  writing new code that would depend on it.

Invocation:
    MNTP_CONFIG=mcgill_configs/mntp/Sheared-LLaMA.json \
    SIMCSE_CONFIG=mcgill_configs/simcse/Sheared-LLaMA.json \
    BASE_MODEL=princeton-nlp/Sheared-LLaMA-1.3B \
    RUN_ROOT=./runs/mcgill_sheared_repro \
      bash scripts/train_mcgill_llm2vec.sh
