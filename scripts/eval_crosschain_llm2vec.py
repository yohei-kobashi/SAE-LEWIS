"""
Cross-chain diagnostic for the -7pt STS-B gap between our own training
pipeline (65.75) and the McGill reference chain (72.52 / paper 73.72).

Runs three mix-and-match configs on STSBenchmark:

  A. McGill public MNTP-merged base + OUR SimCSE adapter
     → tests whether OUR SimCSE is the bottleneck.
     If A ≈ 72 : OUR SimCSE is fine, MNTP is the culprit.
     If A ≈ 65 : OUR SimCSE degrades things (or doesn't compose with
                 McGill's MNTP-adapted base).

  B. OUR MNTP-merged base + McGill public SimCSE adapter
     → tests whether OUR MNTP is the bottleneck.
     Requires an MNTP-only merged checkpoint at --our-mntp-merged-dir
     (produced by mcgill_merge_and_expand.py with no --simcse-adapter).
     If B ≈ 72 : OUR MNTP is fine, SimCSE is the culprit.
     If B ≈ 65 : OUR MNTP degrades things.

  C. McGill public MNTP-merged base + McGill public SimCSE adapter
     (reference chain, sanity check that this reproduces ~72.52).

Interpretation:
  If A ≈ 72 and B ≈ 65 : OUR MNTP is the bottleneck.
  If A ≈ 65 and B ≈ 72 : OUR SimCSE is the bottleneck.
  If both ≈ 68-69       : both contribute equally (~1000 steps just isn't
                          enough for either stage).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


MCGILL_MNTP_BASE = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
MCGILL_SIMCSE_ADAPTER = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _eval_stsb(l2v, batch_size: int = 8) -> float:
    from datasets import load_dataset
    from scipy.stats import spearmanr

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    s1 = [r["sentence1"] for r in ds]
    s2 = [r["sentence2"] for r in ds]
    labels = [r["score"] for r in ds]

    print(f"  encoding side 1 ({len(s1)} texts, bs={batch_size})...")
    t0 = time.time()
    e1 = l2v.encode(s1, batch_size=batch_size)
    print(f"    done in {time.time() - t0:.1f}s")
    print("  encoding side 2 ...")
    t0 = time.time()
    e2 = l2v.encode(s2, batch_size=batch_size)
    print(f"    done in {time.time() - t0:.1f}s")

    # LLM2Vec.encode may return either a torch.Tensor or a numpy.ndarray;
    # normalise to torch before computing cosine.
    if not isinstance(e1, torch.Tensor):
        e1 = torch.as_tensor(e1)
    if not isinstance(e2, torch.Tensor):
        e2 = torch.as_tensor(e2)
    e1 = e1.to(torch.float32)
    e2 = e2.to(torch.float32)
    cos = torch.nn.functional.cosine_similarity(e1, e2, dim=-1)
    return float(spearmanr(cos.cpu().numpy(), labels).correlation)


def _load_chain(base_ref: str, simcse_adapter: str | None, dtype: str,
                device: str, max_seq_len: int):
    from llm2vec import LLM2Vec  # type: ignore

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    print(f"  base:    {base_ref}")
    print(f"  simcse:  {simcse_adapter or '<none>'}")
    t0 = time.time()
    kwargs = dict(
        peft_model_name_or_path=simcse_adapter,
        enable_bidirectional=True,
        torch_dtype=torch_dtype,
        device_map=device,
        max_length=max_seq_len,
    )
    if simcse_adapter is None:
        # No adapter to merge — but merge_peft defaults True which errors
        # if peft_model_name_or_path is None. Force merge_peft=False.
        kwargs["merge_peft"] = False
    l2v = LLM2Vec.from_pretrained(base_ref, **kwargs)
    print(f"  loaded in {time.time() - t0:.1f}s")
    return l2v


def _run_config(name: str, base_ref: str, simcse_adapter: str | None,
                args, results: dict):
    print()
    print("=" * 72)
    print(f"  {name}")
    print("=" * 72)
    try:
        l2v = _load_chain(base_ref, simcse_adapter, args.dtype, args.device,
                          args.max_seq_length)
        score = _eval_stsb(l2v, args.batch_size)
        print(f"  STSBenchmark cosine_spearman = {score:.4f}  "
              f"({score * 100:.2f})")
        results[name] = {
            "base": base_ref,
            "simcse_adapter": simcse_adapter,
            "stsb_spearman": score,
        }
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results[name] = {"base": base_ref, "simcse_adapter": simcse_adapter,
                         "error": f"{type(e).__name__}: {e}"}
    finally:
        # Drop references before allocating the next model.
        for var in ("l2v",):
            if var in locals():
                del locals()[var]
        _free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-simcse", required=True,
                    help="Path to our SimCSE adapter dir (e.g. "
                         "runs/mcgill_sheared_repro/simcse/checkpoint-1000).")
    ap.add_argument("--our-mntp-merged-dir", default=None,
                    help="Path to an MNTP-only merged HF checkpoint (produced "
                         "by mcgill_merge_and_expand.py with no "
                         "--simcse-adapter). Enables config B. If omitted, "
                         "only A and C run.")
    ap.add_argument("--output-json", default="./runs/eval_crosschain.json")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-c", action="store_true",
                    help="Skip config C (McGill+McGill reference re-run) if "
                         "you already have it from llm2vec_repro_eval.py.")
    args = ap.parse_args()

    print(f"[crosschain] our SimCSE adapter: {args.our_simcse}")
    print(f"[crosschain] our MNTP merged dir: {args.our_mntp_merged_dir}")

    results: dict = {}

    # ---- A: McGill public MNTP + OUR SimCSE ----
    _run_config(
        "A: McGill MNTP + OUR SimCSE",
        MCGILL_MNTP_BASE, args.our_simcse, args, results,
    )

    # ---- B: OUR MNTP + McGill public SimCSE ----
    if args.our_mntp_merged_dir:
        _run_config(
            "B: OUR MNTP + McGill SimCSE",
            args.our_mntp_merged_dir, MCGILL_SIMCSE_ADAPTER, args, results,
        )
    else:
        print("\n[skip] Config B not run — pass --our-mntp-merged-dir to enable.")

    # ---- C: McGill + McGill sanity check ----
    if not args.skip_c:
        _run_config(
            "C: McGill MNTP + McGill SimCSE (paper reference)",
            MCGILL_MNTP_BASE, MCGILL_SIMCSE_ADAPTER, args, results,
        )

    # ---- summary ----
    print()
    print("=" * 72)
    print("  crosschain summary")
    print("=" * 72)
    for name, r in results.items():
        s = r.get("stsb_spearman")
        if s is None:
            print(f"  {name:<48} FAILED ({r.get('error')})")
        else:
            print(f"  {name:<48} STS-B = {s * 100:.2f}")

    print()
    print("  Interpretation guide:")
    print("    A ≈ 72: OUR SimCSE is fine; OUR MNTP is the bottleneck.")
    print("    A ≈ 65: OUR SimCSE degrades — either it's undertrained, or")
    print("            it doesn't compose with McGill's MNTP-adapted weights.")
    print("    B ≈ 72: OUR MNTP is fine; OUR SimCSE is the bottleneck.")
    print("    B ≈ 65: OUR MNTP degrades.")

    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {outp}")


if __name__ == "__main__":
    main()
