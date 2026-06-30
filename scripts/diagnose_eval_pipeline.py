"""
End-to-end diagnostic: figure out why McGill's published LLM2Vec ckpt
gives STS-B 0.50 in our pipeline vs the paper's 0.79.

Tests three hypotheses in one run:
  A. Merge issue   — prepare_mcgill_ref.py produced a partially-merged ckpt
  B. Pipeline bug  — our LLM2VecEncoder / mean-pool / cosine differs from canonical
  C. Instruction   — paper numbers require task-specific instruction prefix

Test matrix (each cell = STS-B Spearman):

  | route                                    | no inst | with inst |
  |------------------------------------------|---------|-----------|
  | McGill ckpt via official `llm2vec` lib   | (1)     | (2)       |
  | Our merged McGill ckpt via OUR encoder   | (3)     | (4)       |
  | Our Gemma LoRA ckpt via OUR encoder      | (5)     | (6)       |

Decision tree:
  (1) ≈ 0.79 → official path works without instruction. If (3) ≪ (1), our
              merge or encoder is broken. If (3) ≈ (1), it must be something
              else (e.g. tokenizer config).
  (1) ≈ 0.50, (2) ≈ 0.79 → C confirmed; add instruction support to eval.
  (1) ≈ 0.50, (2) ≈ 0.50 → McGill ckpt does not reach paper numbers
              without supervised fine-tuning. Our 0.54 on Gemma is correct.
  (3) ≈ (1) → our encoder agrees with official; merge is OK.
  (3) ≪ (1) → our encoder is broken even when given the correctly-merged
              ckpt — bug is in LLM2VecEncoder.encode, not in prepare_mcgill_ref.

Also reports weight delta between our merged McGill ckpt and base Mistral
as a sanity check on the merge itself (independent of any encoder issue).

Usage:
    pip install llm2vec
    python scripts/diagnose_eval_pipeline.py
    # or with limited tests:
    python scripts/diagnose_eval_pipeline.py --skip-official  # no llm2vec dep
    python scripts/diagnose_eval_pipeline.py --max-pairs 200  # quick smoke
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
    p.add_argument("--max-pairs", type=int, default=1379,
                   help="STS-B pairs to evaluate. Full test split = 1379. "
                        "Use --max-pairs 200 for a quick smoke (~30s).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=256)

    p.add_argument("--skip-official", action="store_true",
                   help="Skip tests (1)/(2) — useful if `pip install llm2vec` "
                        "is unavailable or you've already tested.")
    p.add_argument("--skip-our-mcgill", action="store_true",
                   help="Skip tests (3)/(4) — useful if our merged McGill dir "
                        "doesn't exist or you only care about the official "
                        "vs paper comparison.")
    p.add_argument("--skip-our-gemma", action="store_true",
                   help="Skip tests (5)/(6) — useful if Gemma ckpt is absent.")
    p.add_argument("--skip-merge-check", action="store_true",
                   help="Skip the McGill merge weight-delta sanity check.")

    p.add_argument("--output-json", default="./runs/diagnose_eval_pipeline.json")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _free(obj_name: str = "<model>") -> None:
    """Aggressively release CUDA memory between loads of 7B-class models."""
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


def _spearman_from_embeds(z1, z2, gold) -> float:
    """Cosine spearman. Accepts numpy or torch (N, d). Auto-L2-normalizes."""
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
# Test 1+2: McGill ckpt via the official llm2vec library
# --------------------------------------------------------------------------- #
def test_official_llm2vec(s1, s2, gold, args) -> Dict[str, float]:
    from llm2vec import LLM2Vec  # type: ignore

    print("[diag] (official) loading McGill ckpt via llm2vec library...")
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
        device_map=args.device,
        torch_dtype=_dtype(args.dtype),
    )

    print("[diag] (official) encoding without instruction...")
    z1 = l2v.encode(s1, batch_size=args.batch_size, show_progress_bar=False)
    z2 = l2v.encode(s2, batch_size=args.batch_size, show_progress_bar=False)
    no_inst = _spearman_from_embeds(z1, z2, gold)
    print(f"[diag] (official) no instruction  STS-B = {no_inst:.4f}")

    print("[diag] (official) encoding with instruction...")
    s1_inst = [[STS_INSTRUCTION, s] for s in s1]
    s2_inst = [[STS_INSTRUCTION, s] for s in s2]
    z1i = l2v.encode(s1_inst, batch_size=args.batch_size, show_progress_bar=False)
    z2i = l2v.encode(s2_inst, batch_size=args.batch_size, show_progress_bar=False)
    with_inst = _spearman_from_embeds(z1i, z2i, gold)
    print(f"[diag] (official) with instruction STS-B = {with_inst:.4f}")

    del l2v, z1, z2, z1i, z2i
    _free("official-l2v")
    return {"no_inst": no_inst, "with_inst": with_inst}


# --------------------------------------------------------------------------- #
# Test 3-6: arbitrary HF-format ckpt via OUR LLM2VecEncoder
# --------------------------------------------------------------------------- #
def test_our_encoder(
    model_dir: str, label: str, s1, s2, gold, args,
) -> Dict[str, float]:
    """Re-uses the SAME LLM2VecEncoder / load_model that eval_llm2vec.py uses,
    so this measures exactly what our eval reports. With-instruction path
    naively prepends the prefix into the sentence text — pooling will include
    the instruction tokens (no masking). Sufficient to see directional effect."""
    import sys
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from eval_llm2vec import LLM2VecEncoder, load_model  # type: ignore

    print(f"[diag] ({label}) loading {model_dir} via our pipeline...")
    model = load_model(model_dir, _dtype(args.dtype), args.device, bidir_patch=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = LLM2VecEncoder(
        model=model, tokenizer=tokenizer, device=args.device,
        pooling="mean", max_seq_length=args.max_seq_length,
    )

    print(f"[diag] ({label}) encoding without instruction...")
    z1 = encoder.encode(s1, batch_size=args.batch_size)
    z2 = encoder.encode(s2, batch_size=args.batch_size)
    no_inst = _spearman_from_embeds(z1, z2, gold)
    print(f"[diag] ({label}) no instruction  STS-B = {no_inst:.4f}")

    print(f"[diag] ({label}) encoding with instruction (naive prepend)...")
    s1_inst = [STS_INSTRUCTION + s for s in s1]
    s2_inst = [STS_INSTRUCTION + s for s in s2]
    z1i = encoder.encode(s1_inst, batch_size=args.batch_size)
    z2i = encoder.encode(s2_inst, batch_size=args.batch_size)
    with_inst = _spearman_from_embeds(z1i, z2i, gold)
    print(f"[diag] ({label}) with instruction STS-B = {with_inst:.4f}")

    del model, encoder, z1, z2, z1i, z2i
    _free(label)
    return {"no_inst": no_inst, "with_inst": with_inst}


# --------------------------------------------------------------------------- #
# Merge sanity check: weight diff between base Mistral and our merged ckpt
# --------------------------------------------------------------------------- #
def test_merge_sanity(our_mcgill_dir: str, dtype: str) -> Dict[str, float]:
    """Compare q_proj.weight at layer 0 (and a deeper layer) between base
    Mistral-7B-Instruct-v0.2 and our merged ckpt. Both stages of LoRA should
    have produced non-trivial weight changes; mean_abs < 1e-6 ⇒ silent
    no-op somewhere."""
    print("[diag] (merge-check) computing weight delta vs base Mistral...")
    dt = _dtype(dtype)

    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=dt,
    )
    base_w0 = base.model.layers[0].self_attn.q_proj.weight.float().cpu().clone()
    base_w15 = base.model.layers[15].mlp.down_proj.weight.float().cpu().clone()
    del base
    _free("base-mistral")

    ours = AutoModelForCausalLM.from_pretrained(
        our_mcgill_dir, torch_dtype=dt,
    )
    our_w0 = ours.model.layers[0].self_attn.q_proj.weight.float().cpu().clone()
    our_w15 = ours.model.layers[15].mlp.down_proj.weight.float().cpu().clone()
    del ours
    _free("our-merged")

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
    if info["layer0_q_proj_mean_abs"] < 1e-6:
        print("[diag] (merge-check) WARNING: layer-0 q_proj barely changed — "
              "MNTP merge may have silently failed.")
    if info["layer15_down_proj_mean_abs"] < 1e-6:
        print("[diag] (merge-check) WARNING: layer-15 down_proj barely "
              "changed — SimCSE adapter may have silently failed.")
    return info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_summary(results: dict, paper_target: float = 0.79) -> None:
    print()
    print("=" * 78)
    print(" STS-B Spearman summary")
    print("=" * 78)
    print(f"{'route':<40s} {'no_inst':>12s} {'with_inst':>12s} {'Δ':>10s}")
    print("-" * 78)
    for route in ("official", "our_mcgill", "our_gemma"):
        if route not in results or "error" in results[route]:
            err = results.get(route, {}).get("error", "skipped")
            print(f"{route:<40s} {'-':>12s} {'-':>12s}    ({err[:30]})")
            continue
        r = results[route]
        delta = r["with_inst"] - r["no_inst"]
        print(f"{route:<40s} {r['no_inst']:>12.4f} {r['with_inst']:>12.4f} "
              f"{delta:>+10.4f}")
    print()
    if "merge_check" in results and "error" not in results["merge_check"]:
        m = results["merge_check"]
        print(f"merge sanity (base Mistral → our merged ckpt):")
        print(f"  layer-0  q_proj  Δ mean_abs = {m['layer0_q_proj_mean_abs']:.4e}")
        print(f"  layer-15 down_proj Δ mean_abs = {m['layer15_down_proj_mean_abs']:.4e}")
    print()


def _print_decision(results: dict) -> None:
    print("=" * 78)
    print(" Interpretation")
    print("=" * 78)
    off = results.get("official", {})
    ours_m = results.get("our_mcgill", {})

    if "no_inst" in off:
        o_no = off["no_inst"]; o_w = off["with_inst"]
        if o_no >= 0.75:
            print(f"  * Official no-inst ({o_no:.3f}) reaches paper range — "
                  "instruction is NOT essential.")
            if "no_inst" in ours_m:
                m_no = ours_m["no_inst"]
                if m_no < o_no - 0.10:
                    print(f"  * Our pipeline gives {m_no:.3f} for the SAME "
                          f"merged ckpt while official gives {o_no:.3f}.")
                    print("    → OUR PIPELINE IS BROKEN. Bug is in "
                          "LLM2VecEncoder.encode / pooling / bidir-patch / "
                          "tokenizer setup.")
                else:
                    print(f"  * Our pipeline gives {m_no:.3f} ≈ official "
                          f"{o_no:.3f} → eval pipeline is fine.")
        elif o_w >= 0.75 and o_no < 0.65:
            print(f"  * Official no-inst {o_no:.3f} ≪ with-inst {o_w:.3f}: "
                  "INSTRUCTION IS ESSENTIAL.")
            print("    → Add instruction support to eval_llm2vec.py "
                  "(tokenize instruction separately, mask out instruction "
                  "tokens during pooling).")
        elif o_no < 0.65 and o_w < 0.65:
            print(f"  * Official no-inst {o_no:.3f} AND with-inst "
                  f"{o_w:.3f} are both below paper. McGill ckpt may not "
                  "reach the paper 0.79 without supervised fine-tuning.")
            print("    → Our LoRA recipe ≈ McGill's published quality.")
        else:
            print(f"  * Official no-inst {o_no:.3f}, with-inst {o_w:.3f} — "
                  "ambiguous; manual interpretation needed.")
    else:
        print("  (skipped official llm2vec test — decision tree limited)")

    if "merge_check" in results and "error" not in results["merge_check"]:
        m = results["merge_check"]
        if m["layer0_q_proj_mean_abs"] < 1e-6:
            print("  * Merge sanity says MNTP adapter didn't change weights.")
        if m["layer15_down_proj_mean_abs"] < 1e-6:
            print("  * Merge sanity says SimCSE adapter didn't change weights.")
    print()


def main():
    args = parse_args()

    print(f"[diag] loading STS-B ({args.max_pairs} pairs)...")
    s1, s2, gold = _load_stsb(args.max_pairs)
    print(f"[diag] loaded {len(s1)} pairs")

    results: dict = {}

    # ----- Merge sanity (cheapest; only loads weights, no inference) -----
    if not args.skip_merge_check:
        try:
            results["merge_check"] = test_merge_sanity(args.our_mcgill_dir, args.dtype)
        except Exception as e:
            print(f"[diag] (merge-check) failed: {type(e).__name__}: {e}")
            results["merge_check"] = {"error": str(e)}
    else:
        results["merge_check"] = {"error": "skipped"}

    # ----- Test 1+2: official llm2vec on McGill -----
    if not args.skip_official:
        try:
            results["official"] = test_official_llm2vec(s1, s2, gold, args)
        except ImportError:
            msg = "llm2vec library not installed — `pip install llm2vec`"
            print(f"[diag] (official) {msg}")
            results["official"] = {"error": msg}
        except Exception as e:
            print(f"[diag] (official) failed: {type(e).__name__}: {e}")
            results["official"] = {"error": str(e)}
    else:
        results["official"] = {"error": "skipped"}

    # ----- Test 3+4: our encoder on our merged McGill -----
    if not args.skip_our_mcgill:
        try:
            results["our_mcgill"] = test_our_encoder(
                args.our_mcgill_dir, "our-mcgill", s1, s2, gold, args,
            )
        except Exception as e:
            print(f"[diag] (our-mcgill) failed: {type(e).__name__}: {e}")
            results["our_mcgill"] = {"error": str(e)}
    else:
        results["our_mcgill"] = {"error": "skipped"}

    # ----- Test 5+6: our encoder on our Gemma LoRA -----
    if not args.skip_our_gemma:
        try:
            results["our_gemma"] = test_our_encoder(
                args.our_gemma_dir, "our-gemma", s1, s2, gold, args,
            )
        except Exception as e:
            print(f"[diag] (our-gemma) failed: {type(e).__name__}: {e}")
            results["our_gemma"] = {"error": str(e)}
    else:
        results["our_gemma"] = {"error": "skipped"}

    _print_summary(results)
    _print_decision(results)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
