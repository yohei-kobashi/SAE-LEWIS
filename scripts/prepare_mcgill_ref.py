"""
Assemble McGill-NLP's public LLM2Vec checkpoint into a self-contained
HF directory that `eval_llm2vec.py` can read.

McGill publishes the LLM2Vec recipe as a CHAIN of LoRA adapters on top
of Mistral-7B-Instruct-v0.2:

    base = mistralai/Mistral-7B-Instruct-v0.2
    + LoRA adapter: McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp
    + LoRA adapter: McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse

The adapter repos contain only adapter_config.json + adapter_model.safetensors
(no config.json), so `AutoTokenizer.from_pretrained(adapter_dir)` and our
eval script's `AutoModelForCausalLM.from_pretrained(...)` both fail.

This script:
  1. Downloads the base + both adapters into the HF cache.
  2. Loads base, applies MNTP adapter, merge_and_unload() bakes MNTP into base.
  3. Applies the SimCSE adapter on top, merge_and_unload() again.
  4. save_pretrained() the resulting GemmaForCausalLM (=> Mistral) as a
     plain HF dir.
  5. Writes a minimal `llm2vec_meta.json` so eval_llm2vec.py can detect
     the checkpoint as "complete" and pick its baseline LLM.

Idempotent: if --output-dir already has a `config.json` + `llm2vec_meta.json`
we skip everything.

Usage:
    python scripts/prepare_mcgill_ref.py \\
        --output-dir ./runs/mcgill_ref/llm2vec_simcse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="./runs/mcgill_ref/llm2vec_simcse",
        help="Where to write the merged HF checkpoint.",
    )
    p.add_argument(
        "--base-llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model for the LoRA adapters.",
    )
    p.add_argument(
        "--mntp-adapter",
        default="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    )
    p.add_argument(
        "--simcse-adapter",
        default="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    )
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--force", action="store_true",
                   help="Re-merge even if output-dir looks complete.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = out_dir / "config.json"
    meta = out_dir / "llm2vec_meta.json"
    if not args.force and cfg.exists() and meta.exists():
        print(f"[mcgill] {out_dir} already has config.json + "
              f"llm2vec_meta.json — skipping merge. Pass --force to redo.")
        return

    # PEFT is required for the LoRA stacking. Import lazily so the helpful
    # error message is the missing dep rather than a stack trace.
    try:
        from peft import PeftModel
    except ImportError as e:
        raise SystemExit(
            f"[mcgill] PEFT is required: `pip install peft`. Underlying "
            f"error: {e}"
        )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[mcgill] loading base {args.base_llm} (dtype={args.dtype})...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_llm,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    print(f"[mcgill] base loaded: vocab_size={base.config.vocab_size}")

    print(f"[mcgill] applying MNTP adapter {args.mntp_adapter}...")
    mntp = PeftModel.from_pretrained(base, args.mntp_adapter)
    print("[mcgill] merging MNTP adapter into base...")
    base = mntp.merge_and_unload()

    print(f"[mcgill] applying SimCSE adapter {args.simcse_adapter}...")
    simcse = PeftModel.from_pretrained(base, args.simcse_adapter)
    print("[mcgill] merging SimCSE adapter...")
    final = simcse.merge_and_unload()

    print(f"[mcgill] saving merged checkpoint to {out_dir}...")
    final.save_pretrained(out_dir, safe_serialization=False)

    # Use the base tokenizer — neither LoRA adapter changes the vocabulary
    # (canonical LLM2Vec uses the existing mask_token; no [INS]/[DEL]).
    tokenizer = AutoTokenizer.from_pretrained(args.base_llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out_dir)

    # Minimal meta so eval_llm2vec.py treats this as a finished run.
    # Set dolma_max_files=0 because the public ckpt was trained on
    # Wikipedia subsets, not Dolma — no shard-level holdout applies and
    # we don't want eval to skip-ahead in our Dolma cache.
    meta_obj = {
        "base_llm": args.base_llm,
        "source_mntp_adapter": args.mntp_adapter,
        "source_simcse_adapter": args.simcse_adapter,
        "simcse": {"source": args.simcse_adapter},
        "lora": {"merged": True, "stacked_chain": ["mntp", "simcse"]},
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "dolma_max_files": 0,
    }
    meta.write_text(json.dumps(meta_obj, indent=2))
    print(f"[mcgill] wrote {meta}")
    print(f"[mcgill] done. Now run:\n"
          f"  SKIP_TRAIN=1 \\\n"
          f"  SIMCSE_DIR={out_dir} \\\n"
          f"  EVAL_DIR={out_dir.parent}/eval \\\n"
          f"  RUN_DIR={out_dir.parent} \\\n"
          f"  bash scripts/train_eval_llm2vec.sh")


if __name__ == "__main__":
    main()
