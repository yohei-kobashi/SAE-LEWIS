"""
End-to-end diagnostic: figure out why McGill's published LLM2Vec ckpt
gives STS-B 0.50 in our pipeline vs the paper's 0.79.

Hypothesis A (merge issue) was RULED OUT by an earlier run that showed
non-trivial weight deltas at layer-0 q_proj AND layer-15 down_proj after
both LoRA merges, so the published prepare_mcgill_ref.py is producing a
correctly-merged Mistral checkpoint.

This version drops the `llm2vec` library dependency — it pins
transformers<=4.44.2 / tokenizers<0.20, which downgrades the env and
breaks our newer-format tokenizer.json files. Instead we reimplement
the library's one essential contribution for STS evaluation: tokenize
the instruction separately so we know how many tokens to mask out of
the mean pool.

Tests in one run:

  | route                                              | STS-B Spearman |
  |----------------------------------------------------|----------------|
  | our merged McGill ckpt — no instruction            | (already 0.50) |
  | our merged McGill ckpt — naive prepend             | (sanity check) |
  | our merged McGill ckpt — proper inst-masked pool   | (decisive)     |
  | our Gemma LoRA ckpt — no instruction               | (already 0.54) |
  | our Gemma LoRA ckpt — naive prepend                |                |
  | our Gemma LoRA ckpt — proper inst-masked pool      |                |

Interpretation:
  - proper-masked ≫ no-inst → hypothesis C confirmed: instruction is
    essential. Add instruction support to eval_llm2vec.py.
  - proper-masked ≈ no-inst → instructions don't help this model. Either
    our recipe gives genuinely lower-quality embeddings (look elsewhere)
    or McGill's paper number requires supervised fine-tune we haven't
    done.
  - naive ≈ proper → the masking detail doesn't matter much; only the
    presence of the instruction prefix matters.

The merge sanity check from the previous version is retained as an
independent confirmation that the saved ckpt isn't just base Mistral.

Usage:
    python scripts/diagnose_eval_pipeline.py
    python scripts/diagnose_eval_pipeline.py --max-pairs 200   # quick smoke

NB: if you ran `pip install llm2vec` earlier and got env breakage, run
    pip install --upgrade transformers tokenizers
first to restore the working transformers/tokenizers pair.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


STS_INSTRUCTION = "Retrieve semantically similar text: "


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--our-mcgill-dir", default="./runs/mcgill_ref/llm2vec_simcse",
                   help="Output of prepare_mcgill_ref.py (merged Mistral ckpt).")
    p.add_argument("--our-gemma-dir", default="./runs/llm2vec_lora/llm2vec_simcse",
                   help="Our Gemma LoRA Bi+MNTP+SimCSE ckpt.")
    p.add_argument("--max-pairs", type=int, default=1379)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=256)

    p.add_argument("--skip-our-mcgill", action="store_true")
    p.add_argument("--skip-our-gemma", action="store_true")
    p.add_argument("--skip-merge-check", action="store_true")

    p.add_argument("--output-json", default="./runs/diagnose_eval_pipeline.json")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_stsb(max_pairs: int) -> Tuple[List[str], List[str], List[float]]:
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    s1 = list(ds["sentence1"])[:max_pairs]
    s2 = list(ds["sentence2"])[:max_pairs]
    gold = [float(x) for x in ds["score"][:max_pairs]]
    return s1, s2, gold


def _spearman(z1, z2, gold) -> float:
    if isinstance(z1, torch.Tensor):
        z1 = z1.detach().float().cpu().numpy()
    if isinstance(z2, torch.Tensor):
        z2 = z2.detach().float().cpu().numpy()
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    z1n = z1 / np.linalg.norm(z1, axis=1, keepdims=True).clip(1e-12)
    z2n = z2 / np.linalg.norm(z2, axis=1, keepdims=True).clip(1e-12)
    cos = (z1n * z2n).sum(axis=1)
    return float(spearmanr(cos, gold).statistic)


# --------------------------------------------------------------------------- #
# Encode functions (replace the llm2vec library)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _encode(
    inner_backbone, tokenizer, sentences: List[str], *,
    instruction: Optional[str] = None,
    mask_instruction: bool = False,
    batch_size: int = 8, max_seq_length: int = 256, device: str = "cuda",
) -> np.ndarray:
    """Encode sentences with optional instruction handling.

    instruction=None          : plain mean pool over the sentence's
                                non-pad positions.
    instruction=s, mask=False : naive prepend — instruction text becomes
                                part of the sentence, pooled with it.
    instruction=s, mask=True  : prepend AND mask the instruction's token
                                positions out of the pool. This is what
                                the llm2vec library does for MTEB STS.
                                The mask span starts at index 1 (BOS) and
                                covers `len(tokenize(instruction))` tokens.
    """
    inst_len = 0
    if instruction is not None:
        # Tokenize WITHOUT special tokens to count exactly the instruction's
        # token contribution; the BOS is added later by the full tokenize.
        inst_ids = tokenizer.encode(instruction, add_special_tokens=False)
        inst_len = len(inst_ids)
        prepended = [instruction + s for s in sentences]
    else:
        prepended = sentences

    embs: List[np.ndarray] = []
    for i in range(0, len(prepended), batch_size):
        chunk = prepended[i : i + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        ).to(device)
        out = inner_backbone(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            use_cache=False,
        )
        h = out.last_hidden_state  # (B, T, d)
        for b in range(h.shape[0]):
            mask = enc.attention_mask[b].float()
            if mask_instruction and inst_len > 0:
                # Drop BOS (index 0) + instruction tokens (indices 1..inst_len).
                # If after masking nothing remains (very short sentence
                # truncated below the instruction), fall back to no-mask.
                trial = mask.clone()
                trial[: 1 + inst_len] = 0
                if trial.sum() > 0:
                    mask = trial
            h_b = h[b].float()
            pooled = (h_b * mask.unsqueeze(-1)).sum(dim=0) / mask.sum().clamp_min(1e-9)
            pooled = F.normalize(pooled, dim=-1)
            embs.append(pooled.cpu().numpy())
    return np.stack(embs)


# --------------------------------------------------------------------------- #
# Test entry point
# --------------------------------------------------------------------------- #
def test_route(
    model_dir: str, label: str, s1, s2, gold, args,
) -> Dict[str, float]:
    import sys
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from eval_llm2vec import load_model  # type: ignore

    print(f"[diag] ({label}) loading {model_dir}...")
    model = load_model(model_dir, _dtype(args.dtype), args.device, bidir_patch=True)
    inner = model.model  # GemmaModel / MistralModel
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine instruction token count (just for logging).
    inst_ids = tokenizer.encode(STS_INSTRUCTION, add_special_tokens=False)
    print(f"[diag] ({label}) STS_INSTRUCTION tokenized to {len(inst_ids)} tokens "
          f"(will be masked out of the pool for the proper-inst test)")

    print(f"[diag] ({label}) [a] no instruction...")
    z1 = _encode(inner, tokenizer, s1,
                 batch_size=args.batch_size,
                 max_seq_length=args.max_seq_length, device=args.device)
    z2 = _encode(inner, tokenizer, s2,
                 batch_size=args.batch_size,
                 max_seq_length=args.max_seq_length, device=args.device)
    no_inst = _spearman(z1, z2, gold)
    print(f"[diag] ({label}) [a] no instruction          STS-B = {no_inst:.4f}")

    print(f"[diag] ({label}) [b] naive prepend (inst in pool)...")
    z1 = _encode(inner, tokenizer, s1, instruction=STS_INSTRUCTION,
                 mask_instruction=False, batch_size=args.batch_size,
                 max_seq_length=args.max_seq_length, device=args.device)
    z2 = _encode(inner, tokenizer, s2, instruction=STS_INSTRUCTION,
                 mask_instruction=False, batch_size=args.batch_size,
                 max_seq_length=args.max_seq_length, device=args.device)
    naive_inst = _spearman(z1, z2, gold)
    print(f"[diag] ({label}) [b] naive prepend           STS-B = {naive_inst:.4f}")

    print(f"[diag] ({label}) [c] proper inst-masked pool (canonical)...")
    z1 = _encode(inner, tokenizer, s1, instruction=STS_INSTRUCTION,
                 mask_instruction=True, batch_size=args.batch_size,
                 max_seq_length=args.max_seq_length, device=args.device)
    z2 = _encode(inner, tokenizer, s2, instruction=STS_INSTRUCTION,
                 mask_instruction=True, batch_size=args.batch_size,
                 max_seq_length=args.max_seq_length, device=args.device)
    proper_inst = _spearman(z1, z2, gold)
    print(f"[diag] ({label}) [c] proper inst-masked      STS-B = {proper_inst:.4f}")

    del model, inner
    _free()
    return {
        "no_inst": no_inst,
        "naive_inst": naive_inst,
        "proper_inst": proper_inst,
    }


# --------------------------------------------------------------------------- #
# Merge sanity (independent confirmation that the merged ckpt ≠ base)
# --------------------------------------------------------------------------- #
def test_merge_sanity(our_mcgill_dir: str, dtype: str) -> Dict[str, float]:
    print("[diag] (merge-check) computing weight delta vs base Mistral...")
    dt = _dtype(dtype)
    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=dt,
    )
    base_w0 = base.model.layers[0].self_attn.q_proj.weight.detach().float().cpu().clone()
    base_w15 = base.model.layers[15].mlp.down_proj.weight.detach().float().cpu().clone()
    del base
    _free()

    ours = AutoModelForCausalLM.from_pretrained(our_mcgill_dir, torch_dtype=dt)
    our_w0 = ours.model.layers[0].self_attn.q_proj.weight.detach().float().cpu().clone()
    our_w15 = ours.model.layers[15].mlp.down_proj.weight.detach().float().cpu().clone()
    del ours
    _free()

    d0 = (our_w0 - base_w0).abs()
    d15 = (our_w15 - base_w15).abs()
    info = {
        "layer0_q_proj_mean_abs":  float(d0.mean()),
        "layer0_q_proj_max_abs":   float(d0.max()),
        "layer15_down_proj_mean_abs": float(d15.mean()),
        "layer15_down_proj_max_abs":  float(d15.max()),
    }
    for k, v in info.items():
        print(f"[diag] (merge-check) {k} = {v:.4e}")
    return info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_summary(results: dict) -> None:
    print()
    print("=" * 82)
    print(" STS-B Spearman summary")
    print("=" * 82)
    print(f"{'route':<32s} {'no_inst':>12s} {'naive_inst':>12s} {'proper_inst':>12s} {'Δ proper':>10s}")
    print("-" * 82)
    for route in ("our_mcgill", "our_gemma"):
        r = results.get(route, {})
        if "error" in r:
            print(f"{route:<32s}    ({r['error'][:50]})")
            continue
        delta = r["proper_inst"] - r["no_inst"]
        print(f"{route:<32s} {r['no_inst']:>12.4f} {r['naive_inst']:>12.4f} "
              f"{r['proper_inst']:>12.4f} {delta:>+10.4f}")
    print()
    m = results.get("merge_check", {})
    if "layer0_q_proj_mean_abs" in m:
        print(f"merge sanity (Mistral base → our merged ckpt):")
        print(f"  layer-0  q_proj  Δ mean_abs = {m['layer0_q_proj_mean_abs']:.4e}")
        print(f"  layer-15 down_proj Δ mean_abs = {m['layer15_down_proj_mean_abs']:.4e}")
        if m["layer0_q_proj_mean_abs"] > 1e-6 and m["layer15_down_proj_mean_abs"] > 1e-6:
            print("  → BOTH adapter stages produced real weight changes (merge OK).")
    print()


def _print_decision(results: dict) -> None:
    print("=" * 82)
    print(" Interpretation")
    print("=" * 82)
    for label, key in [("McGill (Mistral-7B)", "our_mcgill"),
                       ("our Gemma LoRA", "our_gemma")]:
        r = results.get(key, {})
        if "error" in r:
            continue
        no = r["no_inst"]; naive = r["naive_inst"]; proper = r["proper_inst"]
        best = max(no, naive, proper)
        delta_naive = naive - no
        delta_proper = proper - no
        print(f"\n  {label}:")
        print(f"    best = {best:.4f}  (no={no:.4f}, naive={naive:.4f}, proper={proper:.4f})")
        if proper - no > 0.10:
            print("    → INSTRUCTION matters substantially. Add proper-mask "
                  "instruction support to eval_llm2vec.py.")
        elif proper - no > 0.03:
            print("    → instruction helps modestly.")
        elif abs(proper - no) <= 0.03:
            print("    → instruction barely moves the needle for this ckpt.")
        if proper - naive > 0.03:
            print("    → masking the instruction out of the pool is also "
                  "essential (naive prepend isn't enough).")
        elif abs(proper - naive) <= 0.03:
            print("    → masking detail doesn't matter; naive prepend suffices.")

    # Cross-model interpretation
    rm = results.get("our_mcgill", {})
    rg = results.get("our_gemma", {})
    if "proper_inst" in rm and "proper_inst" in rg:
        print()
        print(f"  Cross-model (best-of-three):")
        print(f"    McGill ckpt = {max(rm['no_inst'], rm['naive_inst'], rm['proper_inst']):.4f}")
        print(f"    our Gemma   = {max(rg['no_inst'], rg['naive_inst'], rg['proper_inst']):.4f}")
        if max(rm.values()) >= 0.75:
            print("    → McGill reaches paper range under proper encoding. "
                  "If our Gemma is materially lower, our recipe needs work.")
        elif max(rm.values()) < 0.65:
            print("    → Even McGill's official ckpt doesn't reach 0.79 with "
                  "non-supervised eval. Our Gemma in the same range is fine; "
                  "downstream-quality is the actual gate.")
    print()


def main():
    args = parse_args()

    print(f"[diag] loading STS-B ({args.max_pairs} pairs)...")
    s1, s2, gold = _load_stsb(args.max_pairs)
    print(f"[diag] loaded {len(s1)} pairs")

    results: dict = {}

    if not args.skip_merge_check:
        try:
            results["merge_check"] = test_merge_sanity(args.our_mcgill_dir, args.dtype)
        except Exception as e:
            print(f"[diag] (merge-check) failed: {type(e).__name__}: {e}")
            results["merge_check"] = {"error": str(e)}

    if not args.skip_our_mcgill:
        try:
            results["our_mcgill"] = test_route(
                args.our_mcgill_dir, "our-mcgill", s1, s2, gold, args,
            )
        except Exception as e:
            print(f"[diag] (our-mcgill) failed: {type(e).__name__}: {e}")
            results["our_mcgill"] = {"error": str(e)}

    if not args.skip_our_gemma:
        try:
            results["our_gemma"] = test_route(
                args.our_gemma_dir, "our-gemma", s1, s2, gold, args,
            )
        except Exception as e:
            print(f"[diag] (our-gemma) failed: {type(e).__name__}: {e}")
            results["our_gemma"] = {"error": str(e)}

    _print_summary(results)
    _print_decision(results)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
