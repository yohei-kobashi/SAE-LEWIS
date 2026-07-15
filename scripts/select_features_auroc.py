"""
P-B2: AxBench's SAE-A feature selection, ported to LinguaLens phenomena.

P-B showed that conditioning the editor on LinguaLens's FRC-identified
phenomenon features collapses exact match. But P-B confounds two things:
whether the selection is PHENOMENON-LEVEL (aggregated over a corpus) and
whether FRC is the selector. A reviewer's obvious question — "FRC is a causal
criterion built for interpretability; would a discriminative selector do
better?" — is unanswered, and AxBench answers it only for steering.

So this script implements AxBench's selector on our phenomena, verbatim from
their paper: "For each feature, we compute its max-pooled activations per
Equation (2) over each training example, compute AUROC over the dataset given
true labels, and select the highest-scoring feature by this metric."

It differs from identify_features_frc.py exactly where AxBench differs from
LinguaLens:
  FRC   binary activity -> PS = P(active | positive), PN = P(inactive |
        counterfactual), harmonic mean. Throws the magnitude away.
  AUROC continuous max-pooled activation, ranked against 1/0 labels. Uses the
        magnitude, and is the standard discriminative metric.

Labels come from the dataset, not a judge: sentence1 is the positive (the
phenomenon is present), sentence2 the counterfactual (it is not) — the same
convention identify_features_frc.py uses for PS/PN.

Output is the SAME JSON format identify_features_frc.py emits, so it drops
straight into editflow_probe.py --feature-sets:
    {phenomenon: [[feature_id, auroc], ...]}   (top-r, descending)

Emitting several --top-r values is cheap: the encoding is cached in a sidecar
.acts.jsonl and re-used.

Usage:
    python scripts/select_features_auroc.py \
        --out runs/auroc/identified_l12_16k_r32.json --top-r 32 \
        --explanations runs/np_explanations/gemma-2-2b_12-res-16k.json \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import SAEFeatureExtractor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-sample-size", type=int, default=500,
                   help="held out from selection — the eval pairs, excluded "
                        "exactly as identify_features_frc.py excludes them")
    p.add_argument("--top-r", type=int, default=32)
    p.add_argument("--explanations", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U with average ranks for ties. SAE activations are sparse,
    so most scores are 0 and ties dominate — a naive AUROC that breaks ties
    arbitrarily would inflate it."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # average rank
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def main():
    args = parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)

    rng = np.random.default_rng(args.seed)
    eval_idx = set(rng.choice(len(ds), size=min(args.eval_sample_size,
                                                len(ds)),
                              replace=False).tolist())
    print(f"[auroc] {len(ds)} pairs; {len(eval_idx)} held out (eval sample)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acts_path = out_path.parent / "auroc_acts.jsonl"   # shared across --top-r
    done = {}
    if acts_path.exists():
        with open(acts_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[int(r["idx"])] = r
        print(f"[auroc] RESUME: {len(done)} pairs cached")

    todo = [k for k in range(len(ds)) if k not in eval_idx and k not in done]
    if todo:
        extractor = SAEFeatureExtractor(
            llm_name=args.llm, sae_repo=args.sae_repo,
            sae_path=args.sae_path, layer=args.sae_layer,
            sae_type=args.sae_type, sae_k=args.sae_k, device=args.device)

        @torch.no_grad()
        def maxpool(text: str):
            z = extractor.encode_text(text)              # (T, d_sae)
            v = z.max(dim=0).values.float().cpu()        # AxBench: max-pooled
            nz = torch.nonzero(v > 0).flatten()
            return {int(i): float(v[i]) for i in nz}     # sparse

        af = open(acts_path, "a")
        for n, k in enumerate(todo, 1):
            ex = ds[k]
            row = {"idx": k, "feature": ex.get("feature") or "?",
                   "pos": maxpool(ex["sentence1"]),      # phenomenon present
                   "neg": maxpool(ex["sentence2"])}      # counterfactual
            done[k] = row
            af.write(json.dumps(row) + "\n")
            af.flush()
            if n % 100 == 0:
                print(f"[auroc] encoded {n}/{len(todo)}")
        af.close()

    by_ph = defaultdict(list)
    for r in done.values():
        by_ph[r["feature"]].append(r)
    print(f"[auroc] {len(by_ph)} phenomena")

    expl = {}
    if args.explanations and Path(args.explanations).exists():
        expl = json.loads(Path(args.explanations).read_text())

    result = {}
    lines = ["# AUROC-selected features per phenomenon "
             "(AxBench SAE-A protocol on Gemma Scope l12/16k)", "",
             "AxBench: \"compute its max-pooled activations ..., compute "
             "AUROC over the dataset given true labels, and select the "
             "highest-scoring feature by this metric\".", "",
             f"labels: sentence1=1 (phenomenon present), sentence2=0; "
             f"selection excludes the {len(eval_idx)}-pair eval sample; "
             f"top-{args.top_r} by AUROC", ""]
    for ph in sorted(by_ph):
        rows = by_ph[ph]
        n = len(rows)
        cand = set()
        for r in rows:
            cand.update(int(f) for f in r["pos"])
            cand.update(int(f) for f in r["neg"])
        if not cand:
            continue
        cand = sorted(cand)
        cidx = {f: j for j, f in enumerate(cand)}
        # rows: 2n sentences x |cand| latents, label 1 = positive sentence
        X = np.zeros((2 * n, len(cand)), dtype=np.float32)
        y = np.zeros(2 * n, dtype=int)
        for i, r in enumerate(rows):
            for f, v in r["pos"].items():
                X[2 * i, cidx[int(f)]] = v
            y[2 * i] = 1
            for f, v in r["neg"].items():
                X[2 * i + 1, cidx[int(f)]] = v
        scored = [(f, auroc(X[:, cidx[f]], y)) for f in cand]
        scored.sort(key=lambda x: -x[1])
        top = scored[:args.top_r]
        result[ph] = [[int(f), round(a, 4)] for f, a in top]
        lines.append(f"## {ph} (n={n} pairs, {len(cand)} candidate latents)")
        for f, a in top[:8]:
            lab = expl.get(str(int(f)), "")[:70]
            lines.append(f"- {int(f)} AUROC={a:.3f} {lab}")
        lines.append("")

    out_path.write_text(json.dumps(result))
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"[auroc] wrote {out_path} (+ .md); {len(result)} phenomena, "
          f"top-{args.top_r} each")
    best = [v[0][1] for v in result.values() if v]
    if best:
        print(f"[auroc] best-latent AUROC: mean {np.mean(best):.4f}, "
              f"median {np.median(best):.4f}, min {min(best):.4f}, "
              f"max {max(best):.4f}")
        print("[auroc] AxBench's SAE-A detection average was 0.917 on their "
              "concepts; this is the same protocol on our phenomena, so a "
              "comparable number here means the selector is working and any "
              "downstream collapse is NOT a selection failure.")


if __name__ == "__main__":
    main()
