"""
Stage 1: LLM2Vec-style MNTP training.

Turn causal Gemma into a bidirectional encoder via:
  (1) monkey-patching the attention mask to be bidirectional,
  (2) MNTP-style training with span corruption on sentence-segmented Dolma.

Before training we add the SAE-LEWIS-specific special tokens [INS] and [DEL]
to the tokenizer. Their embedding rows are jointly learned during MNTP. The
[MASK] token is the standard tokenizer mask.

Output: a HF-format checkpoint at --output-dir, loaded later by
`BidirectionalLLM(<output-dir>)` from `model.py`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from model import _patch_attention_bidirectional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--data-cache-dir", default="./dolma_cache")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--span-lambda", type=float, default=3.0)
    p.add_argument("--span-max", type=int, default=10)
    p.add_argument("--mask-coverage", type=float, default=0.15)
    p.add_argument("--sentence-splitter", choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--sent-min-chars", type=int, default=16)
    p.add_argument("--sent-max-chars", type=int, default=2000)
    p.add_argument("--max-sentences-per-text", type=int, default=None,
                   help="Cap on qualifying sentences kept per source document. "
                        "None = use every sentence.")
    p.add_argument("--sentence-sample-strategy",
                   choices=["head", "random", "stride"], default="head")
    p.add_argument("--no-quality-filter", action="store_true",
                   help="Disable the looks-like-sentence heuristic filter.")

    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class DolmaSentenceTokenStream(IterableDataset):
    """Stream Dolma → sentences → token ids."""

    def __init__(
        self,
        shard_paths,
        tokenizer,
        max_seq_length,
        sentence_splitter: str,
        sent_min_chars: int = 16,
        sent_max_chars: int = 2000,
        max_sentences_per_text: Optional[int] = None,
        sample_strategy: str = "head",
        seed: int = 42,
        quality_filter: bool = True,
    ):
        self.shard_paths = list(shard_paths)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sentence_splitter = sentence_splitter
        self.sent_min_chars = sent_min_chars
        self.sent_max_chars = sent_max_chars
        self.max_sentences_per_text = max_sentences_per_text
        self.sample_strategy = sample_strategy
        self.seed = seed
        self.quality_filter = quality_filter

    def __iter__(self) -> Iterator[Dict]:
        worker = get_worker_info()
        shards = self.shard_paths
        worker_seed = self.seed
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
            worker_seed = self.seed + 1000 * worker.id
        text_iter = iter_dolma_texts(shards, min_chars=64)
        sent_iter = iter_sentences(
            text_iter,
            splitter=self.sentence_splitter,
            min_chars=self.sent_min_chars,
            max_chars=self.sent_max_chars,
            max_sentences_per_text=self.max_sentences_per_text,
            sample_strategy=self.sample_strategy,
            seed=worker_seed,
            quality_filter=self.quality_filter,
        )
        for sent in sent_iter:
            enc = self.tokenizer(
                sent, truncation=True, max_length=self.max_seq_length,
                add_special_tokens=True,
            )
            ids = enc["input_ids"]
            if len(ids) < 4:
                continue
            yield {"input_ids": np.asarray(ids, dtype=np.int32)}


def sample_corruption_mask(
    T: int,
    coverage: float,
    span_lambda: float,
    span_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target = max(1, int(coverage * T))
    mask = np.zeros(T, dtype=bool)
    n_masked = 0
    for _ in range(8 * target):
        if n_masked >= target:
            break
        L = int(rng.poisson(span_lambda))
        L = max(1, min(L, span_max, T))
        s = int(rng.integers(0, max(1, T - L + 1)))
        s = min(max(0, s), T - L)
        added = 0
        for i in range(s, s + L):
            if not mask[i]:
                mask[i] = True
                added += 1
        n_masked += added
        if added == 0:
            break
    return mask


class SpanCorruptionCollator:
    def __init__(
        self, mask_token_id, pad_token_id, max_seq_length,
        mask_coverage=0.15, span_lambda=3.0, span_max=10, seed=42,
    ):
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.max_seq_length = int(max_seq_length)
        self.mask_coverage = float(mask_coverage)
        self.span_lambda = float(span_lambda)
        self.span_max = int(span_max)
        self._base_seed = int(seed)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self._base_seed + random.randint(0, 1 << 30))
        items = []
        for ex in batch:
            ids = np.asarray(ex["input_ids"], dtype=np.int64)
            T = int(min(len(ids), self.max_seq_length))
            if T < 4:
                continue
            ids = ids[:T]
            mask = sample_corruption_mask(
                T, self.mask_coverage, self.span_lambda, self.span_max, rng,
            )
            masked = np.where(mask, self.mask_token_id, ids)
            labels = np.where(mask, ids, -100)
            items.append((masked, labels, T))
        if not items:
            raise RuntimeError("empty batch")

        T_max = max(t for _, _, t in items)
        B = len(items)
        ids_arr = np.full((B, T_max), self.pad_token_id, dtype=np.int64)
        attn_arr = np.zeros((B, T_max), dtype=np.int64)
        lbl_arr = np.full((B, T_max), -100, dtype=np.int64)
        for b, (m, l, T) in enumerate(items):
            ids_arr[b, :T] = m
            attn_arr[b, :T] = 1
            lbl_arr[b, :T] = l
        return {
            "input_ids": torch.from_numpy(ids_arr),
            "attention_mask": torch.from_numpy(attn_arr),
            "labels": torch.from_numpy(lbl_arr),
        }


class BidirMNTPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def gradient_checkpointing_enable(self, **kwargs):
        return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.model.gradient_checkpointing_disable()

    @property
    def config(self):
        return self.model.config

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=None, use_cache=False,
        )
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                labels.reshape(-1), ignore_index=-100,
            )
        return {"loss": loss, "logits": out.logits}

    def save_pretrained(self, save_directory, **kwargs):
        return self.model.save_pretrained(save_directory, **kwargs)


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    # Special tokens for the SAE-LEWIS pipeline. [MASK] is the standard one
    # (added if missing). [INS] / [DEL] are new and learned during MNTP.
    added = 0
    if tokenizer.mask_token is None:
        added += tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    added += tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[INS]", "[DEL]"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"[llm2vec] tokens: [MASK]={tokenizer.mask_token_id} "
        f"[INS]={tokenizer.convert_tokens_to_ids('[INS]')} "
        f"[DEL]={tokenizer.convert_tokens_to_ids('[DEL]')}  added={added}"
    )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.llm_dtype]
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=dtype)
    if added > 0 or model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    _patch_attention_bidirectional(model.model)
    model.config.use_cache = False

    shard_paths = download_dolma_shards(args.data_cache_dir, max_files=args.max_files)
    dataset = DolmaSentenceTokenStream(
        shard_paths, tokenizer,
        max_seq_length=args.max_seq_length,
        sentence_splitter=args.sentence_splitter,
        sent_min_chars=args.sent_min_chars,
        sent_max_chars=args.sent_max_chars,
        max_sentences_per_text=args.max_sentences_per_text,
        sample_strategy=args.sentence_sample_strategy,
        seed=args.seed,
        quality_filter=not args.no_quality_filter,
    )

    collator = SpanCorruptionCollator(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_length=args.max_seq_length,
        mask_coverage=args.mask_coverage,
        span_lambda=args.span_lambda,
        span_max=args.span_max,
        seed=args.seed,
    )

    wrapper = BidirMNTPWrapper(model)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        report_to=[],
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        seed=args.seed,
    )

    Trainer(
        model=wrapper, args=targs, train_dataset=dataset, data_collator=collator,
    ).train()

    out_dir = Path(args.output_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "llm2vec_meta.json").write_text(json.dumps({
        "base_llm": args.llm,
        "max_seq_length": args.max_seq_length,
        "mask_coverage": args.mask_coverage,
        "span_lambda": args.span_lambda,
        "span_max": args.span_max,
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "ins_token_id": tokenizer.convert_tokens_to_ids("[INS]"),
        "del_token_id": tokenizer.convert_tokens_to_ids("[DEL]"),
        "seed": args.seed,
    }, indent=2))
    print(f"[llm2vec] saved to {out_dir}")


if __name__ == "__main__":
    main()
