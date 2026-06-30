"""
Reproduce LLM2Vec paper numbers using the OFFICIAL McGill llm2vec library.

Run inside the dedicated venv (set up by `scripts/setup_llm2vec_repro.sh`):

    source ~/venvs/llm2vec_repro/bin/activate
    python scripts/llm2vec_repro_eval.py

This bypasses every piece of our local pipeline — we want to know whether
LLM2Vec paper numbers reproduce AT ALL on this hardware, before deciding
whether the gap we've been chasing on our own Gemma recipe is recipe-level
or eval-level.

Defaults:
  - Base + adapters: McGill-NLP/LLM2Vec-Sheared-LLaMA-* (1.3 B params,
    fits easily on H200, fast to load)
  - Variants: unsup-simcse and supervised
  - Tasks: STSBenchmark (direct) + STS17 (mteb wrapper) by default

Paper-reported numbers (from LLM2Vec Table 1 / mteb leaderboard) used as
reference; flagged in the summary if our number deviates by > 3 points.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Reference values pulled from LLM2Vec paper Table 1 and the public MTEB
# leaderboard entries for these exact checkpoints. STSBenchmark is the
# 1379-pair English test split (cosine spearman, 0-100 scale).
PAPER_REFERENCE: Dict[str, Dict[str, float]] = {
    "unsup-simcse": {
        "STSBenchmark": 73.72,    # paper / leaderboard, approx
        "MTEB-eng-avg": 56.97,    # paper Table 1
    },
    "supervised": {
        "STSBenchmark": 81.67,    # paper / leaderboard, approx
        "MTEB-eng-avg": 62.42,    # paper Table 1
    },
}

VARIANT_TO_PEFT = {
    "unsup-simcse": "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
    "supervised":   "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
}

BASE_REPO = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+",
                   default=["unsup-simcse"],
                   choices=list(VARIANT_TO_PEFT.keys()),
                   help="Which adapter chain(s) to evaluate. Default: just "
                        "unsup-simcse (the main reproduction target).")
    p.add_argument("--tasks", nargs="+",
                   default=["STSBenchmark"],
                   help="Tasks. STSBenchmark uses a direct datasets + scipy "
                        "implementation (no mteb dep). Other names route to "
                        "the mteb library if installed. Use 'STS17' or any "
                        "MTEB task name.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512,
                   help="The llm2vec library's default; reduce if you OOM.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-json", default="./runs/llm2vec_repro/results.json")
    p.add_argument("--no-mteb", action="store_true",
                   help="Skip mteb-routed tasks even if mteb is installed.")
    return p.parse_args()


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_l2v(variant: str, args):
    """Load the published checkpoint chain via the official library."""
    from llm2vec import LLM2Vec  # type: ignore

    peft_path = VARIANT_TO_PEFT[variant]
    print(f"[repro] loading {variant} via official llm2vec lib")
    print(f"[repro]   base : {BASE_REPO}")
    print(f"[repro]   adapter: {peft_path}")
    t0 = time.time()
    l2v = LLM2Vec.from_pretrained(
        BASE_REPO,
        peft_model_name_or_path=peft_path,
        device_map=args.device,
        torch_dtype=_dtype(args.dtype),
        max_length=args.max_seq_length,
    )
    print(f"[repro]   loaded in {time.time() - t0:.1f}s")
    return l2v


def _eval_stsb_direct(l2v, batch_size: int) -> Dict[str, float]:
    """Direct STSBenchmark eval — independent of mteb version churn."""
    from datasets import load_dataset
    from scipy.stats import spearmanr

    print("[repro] (STSBenchmark) loading mteb/stsbenchmark-sts test...")
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    s1 = list(ds["sentence1"])
    s2 = list(ds["sentence2"])
    gold = np.asarray([float(x) for x in ds["score"]])
    print(f"[repro]   {len(s1)} pairs")

    print("[repro] encoding side 1...")
    z1 = l2v.encode(s1, batch_size=batch_size, show_progress_bar=False)
    print("[repro] encoding side 2...")
    z2 = l2v.encode(s2, batch_size=batch_size, show_progress_bar=False)

    if isinstance(z1, torch.Tensor):
        z1 = z1.detach().float().cpu().numpy()
    if isinstance(z2, torch.Tensor):
        z2 = z2.detach().float().cpu().numpy()
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)

    z1n = z1 / np.linalg.norm(z1, axis=1, keepdims=True).clip(1e-12)
    z2n = z2 / np.linalg.norm(z2, axis=1, keepdims=True).clip(1e-12)
    cos = (z1n * z2n).sum(axis=1)
    rho = float(spearmanr(cos, gold).statistic)
    return {"cosine_spearman": rho, "n_pairs": int(len(s1))}


def _eval_via_mteb(l2v, task_names: List[str]) -> Dict[str, Dict]:
    """Route the requested tasks through the mteb library."""
    try:
        import mteb  # type: ignore
    except ImportError:
        print("[repro] mteb not installed; skipping mteb-routed tasks.")
        return {}

    print(f"[repro] running mteb tasks: {task_names}")
    # mteb 1.x API
    try:
        evaluation = mteb.MTEB(tasks=task_names)
    except TypeError:
        # mteb 2.x — convert to task objects first.
        task_objs = mteb.get_tasks(tasks=task_names)
        evaluation = mteb.MTEB(tasks=task_objs)

    out_dir = Path("./runs/llm2vec_repro/mteb_raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = evaluation.run(l2v, output_folder=str(out_dir), overwrite_results=True)

    # Normalise raw results into a flat dict.
    results: Dict[str, Dict] = {}
    for name in task_names:
        # raw is usually a list[TaskResult] or dict.
        item = None
        if isinstance(raw, list):
            for r in raw:
                meta = getattr(r, "task_name", None) or (
                    getattr(r, "metadata", {}) or {}).get("name")
                if meta == name:
                    item = r
                    break
        elif isinstance(raw, dict):
            item = raw.get(name)
        if item is None:
            results[name] = {"error": "task result not found in mteb output"}
            continue
        # Best-effort headline extraction.
        if hasattr(item, "to_dict"):
            try:
                blob = item.to_dict()
            except Exception:
                blob = None
        else:
            blob = None
        if blob is None:
            blob = getattr(item, "scores", None)
        results[name] = {"raw": blob}
    return results


def _print_summary(all_results: Dict[str, Dict]) -> None:
    print()
    print("=" * 78)
    print(" LLM2Vec reproduction summary (Sheared-LLaMA-1.3B)")
    print("=" * 78)
    print(f"{'variant':<18s} {'task':<20s} {'ours':>10s} "
          f"{'paper':>10s} {'Δ':>10s}")
    print("-" * 78)
    for variant, tasks in all_results.items():
        ref = PAPER_REFERENCE.get(variant, {})
        for task, val in tasks.items():
            if isinstance(val, dict) and "cosine_spearman" in val:
                ours = val["cosine_spearman"] * 100      # to 0-100 scale
            elif isinstance(val, dict) and "main_score" in val:
                ours = float(val["main_score"]) * 100
            else:
                print(f"{variant:<18s} {task:<20s} {'(no score)':>10s}")
                continue
            paper = ref.get(task)
            paper_s = f"{paper:.2f}" if paper is not None else "    -"
            delta_s = f"{(ours - paper):+.2f}" if paper is not None else "    -"
            print(f"{variant:<18s} {task:<20s} {ours:>10.2f} "
                  f"{paper_s:>10s} {delta_s:>10s}")
    print()
    # Verdict.
    failures = []
    for variant, tasks in all_results.items():
        ref = PAPER_REFERENCE.get(variant, {})
        for task, val in tasks.items():
            if isinstance(val, dict) and "cosine_spearman" in val:
                ours = val["cosine_spearman"] * 100
            elif isinstance(val, dict) and "main_score" in val:
                ours = float(val["main_score"]) * 100
            else:
                continue
            paper = ref.get(task)
            if paper is None:
                continue
            if abs(ours - paper) > 3.0:
                failures.append((variant, task, ours, paper))
    if not failures:
        print(" → Reproduction WITHIN ±3 points of paper. Numbers match.")
    else:
        print(" → MISMATCH (Δ > 3) on:")
        for v, t, ours, paper in failures:
            print(f"     - {v} / {t}: ours {ours:.2f} vs paper {paper:.2f} "
                  f"(Δ {(ours - paper):+.2f})")


def main():
    args = parse_args()

    # Verify we're inside the dedicated venv with llm2vec available.
    try:
        import llm2vec  # noqa: F401
    except ImportError:
        sys.exit("[repro] llm2vec library not installed in the active env. "
                 "Run: bash scripts/setup_llm2vec_repro.sh && "
                 "source ~/venvs/llm2vec_repro/bin/activate")

    print(f"[repro] using python {sys.executable}")
    print(f"[repro] variants: {args.variants}")
    print(f"[repro] tasks   : {args.tasks}")

    all_results: Dict[str, Dict] = {}
    for variant in args.variants:
        l2v = _load_l2v(variant, args)
        variant_results: Dict = {}

        # STSBenchmark via direct path.
        if "STSBenchmark" in args.tasks:
            res = _eval_stsb_direct(l2v, args.batch_size)
            variant_results["STSBenchmark"] = res
            print(f"[repro] {variant} STSBenchmark cosine_spearman = "
                  f"{res['cosine_spearman']:.4f}")

        # Other tasks via mteb (if available).
        other_tasks = [t for t in args.tasks if t != "STSBenchmark"]
        if other_tasks and not args.no_mteb:
            mteb_res = _eval_via_mteb(l2v, other_tasks)
            variant_results.update(mteb_res)

        all_results[variant] = variant_results

        del l2v
        _free()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "results": all_results,
        "reference": PAPER_REFERENCE,
        "config": vars(args),
    }, indent=2, default=str))
    _print_summary(all_results)
    print(f"\n[repro] wrote {out_path}")


if __name__ == "__main__":
    main()
