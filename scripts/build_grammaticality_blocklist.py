"""
Identify GENERIC grammaticality-responsive SAE features and emit a
blocklist for confound-controlled conditioning (README §6.2.9-to-be).

Motivation. v4c centres on VOICE / structural transformations. Even a
correct passive/cleft/inversion pair carries some generic
"markedness / anomaly" SAE signal, and any residual rule failure carries a
lot of it; if such features dominate the conditioning diff, the learned
conditioning semantics degenerate from "passivize" to "change
grammaticality". The fix: pre-identify features that respond to
grammaticality PER SE and mask them before the top-k extraction of
z_amp / z_sup (and in the ranker's sae_align pooling).

Identification corpus: BLiMP (nyu-mll/blimp) — 67 paradigms × 1000
minimal pairs (sentence_good vs sentence_bad). Its intentionally
ungrammatical members, useless as training targets, are exactly what is
needed here.

The key design point is the CROSS-PARADIGM GENERALITY criterion. A
passive-voice feature responds to the passive-related paradigms — and
must be KEPT; a generic anomaly detector responds across many unrelated
paradigms with a consistent sign — and must be BLOCKED. Per feature f and
paradigm p we compute the effect size of δ = pool(SAE(good)) −
pool(SAE(bad)); f enters the blocklist iff it "responds" (|effect| ≥
--effect-thresh AND |mean δ| ≥ --rel-act-thresh × its global activation
scale) in ≥ --min-paradigms paradigms with ≥ --sign-consistency sign
agreement.

Built-in validation:
  * capture   — fraction of BLiMP diff top-k mass that the blocklist
                removes (want HIGH: the generic signal is caught);
  * preserve  — fraction of LinguaLens s1↔s2 diff top-k mass removed
                (both sides grammatical; want LOW: phenomenon features
                survive). Skipped with --lingualens-sample 0.

Outputs under --output-dir:
  blocklist.npy          int64 feature ids (sorted)
  blocklist_meta.json    thresholds + dataset info + validation numbers
  responses.npy          (n_paradigms, d_sae) float32 mean-δ matrix
  blocklist_report.md    human-readable summary

Usage:
    python scripts/build_grammaticality_blocklist.py \
        --output-dir runs/blocklist \
        --pairs-per-paradigm 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"], default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--blimp-dataset", default="nyu-mll/blimp")
    p.add_argument("--paradigms", default="all",
                   help="'all' or comma list of BLiMP config names.")
    p.add_argument("--pairs-per-paradigm", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=64)

    # Selection criterion (see module docstring)
    p.add_argument("--effect-thresh", type=float, default=0.5,
                   help="Min |mean δ| / std δ per paradigm to count as a "
                        "response.")
    p.add_argument("--rel-act-thresh", type=float, default=0.05,
                   help="Min |mean δ| relative to the feature's global "
                        "pooled-activation scale.")
    p.add_argument("--min-paradigms", type=int, default=20,
                   help="Respond in at least this many paradigms → generic.")
    p.add_argument("--sign-consistency", type=float, default=0.8,
                   help="Fraction of responding paradigms sharing the "
                        "majority sign.")

    # Validation
    p.add_argument("--val-topk", type=int, default=8,
                   help="Top-k of |δ| used for capture/preserve mass checks "
                        "(matches the conditioning candidate pool K_top).")
    p.add_argument("--lingualens-sample", type=int, default=500,
                   help="0 disables the LinguaLens preservation check.")
    p.add_argument("--lingualens-dataset", default="THU-KEG/LinguaLens-Data")

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pure selection / validation logic (unit-testable without models)
# ---------------------------------------------------------------------------
def select_blocklist(
    mean_delta: np.ndarray,       # (P, F) per-paradigm mean of δ
    std_delta: np.ndarray,        # (P, F) per-paradigm std of δ
    act_scale: np.ndarray,        # (F,)  global pooled-activation scale
    effect_thresh: float,
    rel_act_thresh: float,
    min_paradigms: int,
    sign_consistency: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (blocked feature ids, diagnostics).

    A feature responds in paradigm p iff
        |mean_delta| / (std_delta + eps) >= effect_thresh
        AND |mean_delta| >= rel_act_thresh * (act_scale + eps).
    Blocked iff it responds in >= min_paradigms paradigms and the majority
    sign covers >= sign_consistency of the responding paradigms.
    """
    eps = 1e-6
    effect = np.abs(mean_delta) / (std_delta + eps)                # (P, F)
    responds = (effect >= effect_thresh) & (
        np.abs(mean_delta) >= rel_act_thresh * (act_scale + eps)[None, :])
    n_resp = responds.sum(axis=0)                                  # (F,)
    pos = (responds & (mean_delta > 0)).sum(axis=0)
    neg = (responds & (mean_delta < 0)).sum(axis=0)
    major = np.maximum(pos, neg)
    consistency = np.where(n_resp > 0, major / np.maximum(n_resp, 1), 0.0)
    blocked = (n_resp >= min_paradigms) & (consistency >= sign_consistency)
    ids = np.nonzero(blocked)[0].astype(np.int64)
    return ids, {"n_resp": n_resp, "consistency": consistency,
                 "pos": pos, "neg": neg}


def topk_mass_blocked(
    deltas: Sequence[np.ndarray], blocked_ids: np.ndarray, k: int,
) -> float:
    """Mean fraction of |δ| top-k mass that falls on blocked features."""
    blocked = set(int(i) for i in blocked_ids)
    fracs = []
    for d in deltas:
        a = np.abs(d)
        if a.sum() <= 0:
            continue
        idx = np.argsort(-a)[:k]
        mass = a[idx].sum()
        if mass <= 0:
            continue
        bmass = sum(a[i] for i in idx if int(i) in blocked)
        fracs.append(bmass / mass)
    return float(np.mean(fracs)) if fracs else float("nan")


# ---------------------------------------------------------------------------
@torch.no_grad()
def pooled_features(extractor, texts: List[str], batch_size: int,
                    max_length: int) -> torch.Tensor:
    """Batched sentence-level pool-max SAE features, (N, d_sae) fp32 cpu."""
    device = next(extractor.llm.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        enc = extractor.llm_tokenizer(
            chunk, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt").to(device)
        h = extractor.llm(**enc, output_hidden_states=True,
                          use_cache=False).hidden_states[extractor.layer_idx]
        z = extractor.sae.encode(h.to(extractor.sae.W_enc.dtype))  # (B,T,F)
        mask = enc["attention_mask"].unsqueeze(-1).to(z.dtype)
        pooled = (z * mask).max(dim=1).values                      # (B,F)
        out.append(pooled.float().cpu())
    return torch.cat(out, dim=0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import get_dataset_config_names, load_dataset
    from model import SAEFeatureExtractor

    if args.paradigms == "all":
        paradigms = sorted(get_dataset_config_names(args.blimp_dataset))
    else:
        paradigms = [x.strip() for x in args.paradigms.split(",") if x.strip()]
    print(f"[blocklist] {len(paradigms)} BLiMP paradigms")

    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    d_sae = int(extractor.d_sae)

    mean_delta = np.zeros((len(paradigms), d_sae), dtype=np.float32)
    std_delta = np.zeros((len(paradigms), d_sae), dtype=np.float32)
    act_sum = np.zeros(d_sae, dtype=np.float64)
    act_n = 0
    val_deltas: List[np.ndarray] = []   # capture-check subset
    rng = np.random.default_rng(args.seed)

    for pi, para in enumerate(paradigms):
        ds = load_dataset(args.blimp_dataset, para, split="train")
        n = min(args.pairs_per_paradigm, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        good = [ds[int(i)]["sentence_good"] for i in idx]
        bad = [ds[int(i)]["sentence_bad"] for i in idx]
        zg = pooled_features(extractor, good, args.batch_size, args.max_length)
        zb = pooled_features(extractor, bad, args.batch_size, args.max_length)
        delta = (zg - zb).numpy()                                  # (n, F)
        mean_delta[pi] = delta.mean(axis=0)
        std_delta[pi] = delta.std(axis=0)
        act_sum += zg.numpy().sum(axis=0) + zb.numpy().sum(axis=0)
        act_n += 2 * n
        # keep a few raw deltas per paradigm for the capture check
        for row in delta[:20]:
            val_deltas.append(row)
        print(f"[blocklist] {pi + 1}/{len(paradigms)} {para}: {n} pairs")

    act_scale = (act_sum / max(1, act_n)).astype(np.float32)

    blocked_ids, diag = select_blocklist(
        mean_delta, std_delta, act_scale,
        effect_thresh=args.effect_thresh,
        rel_act_thresh=args.rel_act_thresh,
        min_paradigms=args.min_paradigms,
        sign_consistency=args.sign_consistency,
    )
    print(f"[blocklist] blocked {len(blocked_ids)} / {d_sae} features")

    capture = topk_mass_blocked(val_deltas, blocked_ids, args.val_topk)
    print(f"[blocklist] BLiMP capture (top-{args.val_topk} |δ| mass "
          f"blocked): {capture:.3f}  (want HIGH)")

    preserve_removed = float("nan")
    if args.lingualens_sample > 0:
        ll = load_dataset(args.lingualens_dataset, split="train")
        ll = ll.filter(lambda r: r["language"] == "English")
        order = rng.choice(len(ll), size=min(args.lingualens_sample, len(ll)),
                           replace=False)
        s1 = [ll[int(i)]["sentence1"] for i in order]
        s2 = [ll[int(i)]["sentence2"] for i in order]
        z1 = pooled_features(extractor, s1, args.batch_size, args.max_length)
        z2 = pooled_features(extractor, s2, args.batch_size, args.max_length)
        ll_deltas = [(a - b).numpy() for a, b in zip(z2, z1)]
        preserve_removed = topk_mass_blocked(ll_deltas, blocked_ids,
                                             args.val_topk)
        print(f"[blocklist] LinguaLens removal (top-{args.val_topk} |δ| "
              f"mass blocked): {preserve_removed:.3f}  (want LOW)")

    # ---- outputs -------------------------------------------------------- #
    np.save(out_dir / "blocklist.npy", blocked_ids)
    np.save(out_dir / "responses.npy", mean_delta)
    meta = {
        "d_sae": d_sae,
        "n_blocked": int(len(blocked_ids)),
        "paradigms": paradigms,
        "pairs_per_paradigm": int(args.pairs_per_paradigm),
        "thresholds": {
            "effect_thresh": args.effect_thresh,
            "rel_act_thresh": args.rel_act_thresh,
            "min_paradigms": args.min_paradigms,
            "sign_consistency": args.sign_consistency,
        },
        "validation": {
            "val_topk": args.val_topk,
            "blimp_capture_mass": capture,
            "lingualens_removed_mass": preserve_removed,
        },
        "sae": {"repo": args.sae_repo, "path": args.sae_path,
                "layer": args.sae_layer},
        "seed": args.seed,
    }
    (out_dir / "blocklist_meta.json").write_text(json.dumps(meta, indent=2))

    n_resp = diag["n_resp"]
    order = np.argsort(-n_resp)
    lines = ["# Generic grammaticality-feature blocklist", "",
             f"SAE: `{args.sae_repo}/{args.sae_path}`  d_sae={d_sae}",
             f"paradigms: {len(paradigms)}  pairs/paradigm: "
             f"{args.pairs_per_paradigm}", "",
             f"**blocked: {len(blocked_ids)} features**  (criterion: "
             f"respond in ≥{args.min_paradigms} paradigms, effect ≥ "
             f"{args.effect_thresh}, sign consistency ≥ "
             f"{args.sign_consistency})", "",
             "## Validation", "",
             f"- BLiMP top-{args.val_topk} |δ| mass blocked (capture, want "
             f"high): **{capture:.3f}**",
             f"- LinguaLens top-{args.val_topk} |δ| mass blocked (want "
             f"low): **{preserve_removed:.3f}**", "",
             "## Response-count histogram (paradigms responded)", ""]
    hist = np.bincount(np.minimum(n_resp, 67), minlength=68)
    for lo in range(0, 68, 10):
        c = int(hist[lo:lo + 10].sum())
        lines.append(f"- {lo:2d}-{min(lo + 9, 67):2d} paradigms: {c} features")
    lines += ["", "## Top blocked features by generality", "",
              "| feature | paradigms responded | sign (+/-) |", "|---|---|---|"]
    shown = 0
    blocked_set = set(int(i) for i in blocked_ids)
    for f in order:
        if int(f) not in blocked_set:
            continue
        lines.append(f"| {int(f)} | {int(n_resp[f])} | "
                     f"+{int(diag['pos'][f])}/-{int(diag['neg'][f])} |")
        shown += 1
        if shown >= 30:
            break
    lines.append("")
    (out_dir / "blocklist_report.md").write_text("\n".join(lines))
    print(f"[blocklist] wrote {out_dir}/blocklist.npy, blocklist_meta.json, "
          f"blocklist_report.md")


if __name__ == "__main__":
    main()
