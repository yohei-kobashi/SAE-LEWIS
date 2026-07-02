"""
Held-out evaluation of the trained SAE-LEWIS tagger and editor.

Runs Level 1 (held-out accuracy) and Level 2 (conditioning causality) on a
DEV corruption cache — a cache generated from Dolma sentences disjoint from
the training cache (see `corruption.py --skip-sentences`).

Level 1 — per-stage metrics (README §13.1):
  tagger : per-class P/R/F1 over KEEP/REPL/INS/DEL, confusion matrix,
           macro-F1 vs the all-KEEP baseline, edit-span IoU
  editor : per-position-type argmax accuracy (KEEP identity, REPL/INS
           restoration top-1/top-5, DEL → [DEL]), overgenerate /
           undergenerate rates, mean CE loss, [DEL]-logit separation

Level 2 — conditioning causality (the central SAE-LEWIS validity check):
  every sample is evaluated under three paired conditions —
    true   : diff-based z_amp/z_sup (as in training, empty-prob = 0)
    empty  : z_amp = z_sup = 0
    random : same k / same magnitudes as `true`, random feature indices
  Interpretation:
    Δ(true − empty) ≈ 0  → the model IGNORES the conditioning
    Δ(true − random) ≪ Δ(true − empty)
                         → conditioning is used as an opaque "something
                           changed" flag, not as SAE-grounded features

Outputs `eval_report.md` + `eval_metrics.json` under --output-dir.

Usage:
    python eval_tagger_editor.py \
        --corruption-dir runs/prod/corruption_dev \
        --llm2vec-dir    runs/prod/llm2vec_merged \
        --tagger-ckpt    runs/prod/tagger/tagger-final.pt \
        --editor-ckpt    runs/prod/editor/editor-final.pt \
        --output-dir     runs/prod/eval_tagger_editor
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data import CorruptionCollator, CorruptionDataset
from editor import load_editor_from_checkpoint
from intervene import diff_to_sparse
from lewis_ops import NUM_OPS, OP_NAMES
from tagger import load_tagger_from_checkpoint

CONDITIONS = ["true", "empty", "random"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True,
                   help="DEV corruption cache (disjoint from training; "
                        "generate with corruption.py --skip-sentences).")
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--tagger-ckpt", default=None,
                   help="tagger-final.pt (skip tagger eval if omitted).")
    p.add_argument("--editor-ckpt", default=None,
                   help="editor-final.pt (skip editor eval if omitted).")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--k-top", type=int, default=8,
                   help="Diff candidate pool size (match training).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Paired conditioning construction
# ---------------------------------------------------------------------------
def build_conditions(
    z_X: torch.Tensor,          # (B, d_sae)
    z_X_prime: torch.Tensor,    # (B, d_sae)
    k_top: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Build the three paired (z_amp, z_sup) variants for a batch.

    The per-sample k_amp / k_sup draw matches the training distribution
    (uniform over {0..3}) but empty-conditioning is disabled so every
    sample carries signal under `true`. `random` reuses the exact nonzero
    counts and (permuted) magnitudes of `true` at uniformly random feature
    indices, so any accuracy gap vs `true` isolates feature IDENTITY.
    """
    B, d_sae = z_X.shape
    amp_t = torch.zeros_like(z_X)
    sup_t = torch.zeros_like(z_X)
    amp_r = torch.zeros_like(z_X)
    sup_r = torch.zeros_like(z_X)

    for b in range(B):
        k_amp = int(rng.integers(0, 4))
        k_sup = int(rng.integers(0, 4))
        a, s = diff_to_sparse(
            z_X[b], z_X_prime[b],
            k_top=k_top, k_amp=k_amp, k_sup=k_sup,
            rng=rng, empty_conditioning_prob=0.0,
        )
        amp_t[b] = a
        sup_t[b] = s
        for true_vec, rand_vec in ((a, amp_r), (s, sup_r)):
            nz = (true_vec > 0).nonzero(as_tuple=True)[0]
            if len(nz) == 0:
                continue
            vals = true_vec[nz].cpu().numpy()
            rng.shuffle(vals)
            rand_idx = rng.choice(d_sae, size=len(nz), replace=False)
            for i, v in zip(rand_idx, vals):
                rand_vec[b, int(i)] = float(v)

    zeros = torch.zeros_like(z_X)
    return {
        "true": (amp_t, sup_t),
        "empty": (zeros, zeros.clone()),
        "random": (amp_r, sup_r),
    }


# ---------------------------------------------------------------------------
# Tagger evaluation
# ---------------------------------------------------------------------------
def eval_tagger(args, d_sae: int, loader) -> Dict:
    dtype = _dtype(args.llm_dtype)
    print(f"[eval] loading tagger from {args.tagger_ckpt}")
    tagger = load_tagger_from_checkpoint(
        args.llm2vec_dir, args.tagger_ckpt, d_sae=d_sae, dtype=dtype,
    ).to(args.device)
    tagger.eval()

    rng = np.random.default_rng(args.seed)
    conf = {c: np.zeros((NUM_OPS, NUM_OPS), dtype=np.int64) for c in CONDITIONS}
    iou_sum = {c: 0.0 for c in CONDITIONS}
    iou_n = {c: 0 for c in CONDITIONS}
    seen = 0

    for batch in tqdm(loader, desc="[eval:tagger]", unit="batch"):
        conds = build_conditions(batch["z_X"], batch["z_X_prime"],
                                 args.k_top, rng)
        ids = batch["tagger_input_ids"].to(args.device)
        attn = batch["tagger_attention_mask"].to(args.device)
        gold = batch["tagger_gold"]
        valid = gold != -100

        for c in CONDITIONS:
            z_amp, z_sup = conds[c]
            with torch.no_grad():
                out = tagger(ids, attn,
                             z_amp.to(args.device), z_sup.to(args.device))
            pred = out["logits"].argmax(dim=-1).cpu()
            g = gold[valid].numpy()
            p = pred[valid].numpy()
            np.add.at(conf[c], (g, p), 1)

            # Edit-span IoU per sample (gold vs predicted non-KEEP sets)
            for b in range(ids.shape[0]):
                v = valid[b]
                ge = set((gold[b][v] > 0).nonzero(as_tuple=True)[0].tolist())
                pe = set((pred[b][v] > 0).nonzero(as_tuple=True)[0].tolist())
                union = ge | pe
                if union:
                    iou_sum[c] += len(ge & pe) / len(union)
                    iou_n[c] += 1

        seen += ids.shape[0]
        if seen >= args.max_samples:
            break

    metrics = {"n_samples": seen, "conditions": {}}
    for c in CONDITIONS:
        m = _prf_from_confusion(conf[c])
        m["span_iou"] = iou_sum[c] / max(1, iou_n[c])
        m["confusion"] = conf[c].tolist()
        metrics["conditions"][c] = m

    # All-KEEP baseline from the gold marginal (same for every condition)
    gold_counts = conf["true"].sum(axis=1)
    baseline_conf = np.zeros_like(conf["true"])
    baseline_conf[:, 0] = gold_counts
    metrics["all_keep_baseline"] = _prf_from_confusion(baseline_conf)

    metrics["diagnostics"] = {
        "cond_scale": float(tagger.cond_scale.item()),
        "proj_a_weight_norm": float(tagger.proj_a.weight.norm().item()),
    }
    del tagger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _prf_from_confusion(conf: np.ndarray) -> Dict:
    per_class = {}
    f1s = []
    for i, name in enumerate(OP_NAMES):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class[name] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": int(conf[i, :].sum()),
        }
        f1s.append(f1)
    total = float(conf.sum()) or 1.0
    return {
        "accuracy": float(np.trace(conf)) / total,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Editor evaluation
# ---------------------------------------------------------------------------
def eval_editor(args, d_sae: int, loader, tok) -> Dict:
    dtype = _dtype(args.llm_dtype)
    print(f"[eval] loading editor from {args.editor_ckpt}")
    editor = load_editor_from_checkpoint(
        args.llm2vec_dir, args.editor_ckpt, d_sae=d_sae, dtype=dtype,
    ).to(args.device)
    editor.eval()

    mask_id = int(tok.mask_token_id)
    ins_id = int(tok.convert_tokens_to_ids("[INS]"))
    del_id = int(tok.convert_tokens_to_ids("[DEL]"))

    rng = np.random.default_rng(args.seed)
    agg = {c: defaultdict(float) for c in CONDITIONS}
    seen = 0

    for batch in tqdm(loader, desc="[eval:editor]", unit="batch"):
        conds = build_conditions(batch["z_X"], batch["z_X_prime"],
                                 args.k_top, rng)
        ids = batch["editor_input_ids"].to(args.device)
        attn = batch["editor_attention_mask"].to(args.device)
        tgt = batch["editor_target_ids"]

        valid = tgt != -100
        is_repl = (batch["editor_input_ids"] == mask_id) & valid
        is_ins = (batch["editor_input_ids"] == ins_id) & valid
        is_del = (tgt == del_id) & valid
        is_keep = valid & ~is_repl & ~is_ins & ~is_del

        for c in CONDITIONS:
            z_amp, z_sup = conds[c]
            with torch.no_grad():
                out = editor(ids, attn,
                             z_amp.to(args.device), z_sup.to(args.device))
                logits = out["logits"].float().cpu()

            a = agg[c]
            a["loss_sum"] += float(F.cross_entropy(
                logits[valid], tgt[valid], reduction="sum").item())
            a["loss_n"] += int(valid.sum())

            pred1 = logits.argmax(dim=-1)
            for key, m in (("keep", is_keep), ("repl", is_repl),
                           ("ins", is_ins), ("del", is_del)):
                a[f"{key}_n"] += int(m.sum())
                a[f"{key}_top1"] += int((pred1[m] == tgt[m]).sum())
            for key, m in (("repl", is_repl), ("ins", is_ins)):
                if m.any():
                    top5 = logits[m].topk(5, dim=-1).indices
                    a[f"{key}_top5"] += int(
                        (top5 == tgt[m].unsqueeze(-1)).any(dim=-1).sum())
            # Overgeneration: [DEL] fired where the target is identity.
            a["overgen"] += int((pred1[is_keep] == del_id).sum())
            # [DEL]-logit separation (diagnoses the tied output-column delta)
            if is_del.any():
                a["del_logit_at_del"] += float(logits[is_del][:, del_id].sum())
            if is_keep.any():
                a["del_logit_at_keep"] += float(logits[is_keep][:, del_id].sum())

        seen += ids.shape[0]
        if seen >= args.max_samples:
            break

    metrics = {"n_samples": seen, "conditions": {}}
    for c in CONDITIONS:
        a = agg[c]
        m = {"mean_loss": a["loss_sum"] / max(1, a["loss_n"])}
        for key in ("keep", "repl", "ins", "del"):
            n = max(1, int(a[f"{key}_n"]))
            m[f"{key}_positions"] = int(a[f"{key}_n"])
            m[f"{key}_top1_acc"] = a[f"{key}_top1"] / n
        for key in ("repl", "ins"):
            m[f"{key}_top5_acc"] = a[f"{key}_top5"] / max(1, int(a[f"{key}_n"]))
        m["overgenerate_rate"] = a["overgen"] / max(1, int(a["keep_n"]))
        m["undergenerate_rate"] = 1.0 - m["del_top1_acc"]
        m["del_logit_mean_at_del"] = a["del_logit_at_del"] / max(1, int(a["del_n"]))
        m["del_logit_mean_at_keep"] = a["del_logit_at_keep"] / max(1, int(a["keep_n"]))
        metrics["conditions"][c] = m

    metrics["diagnostics"] = {
        "cond_scale": float(editor.cond_scale.item()),
        "proj_a_weight_norm": float(editor.proj_a.weight.norm().item()),
        "delta_emb_norms": {
            name: float(editor.delta_emb[slot].norm().item())
            for slot, (name, _tid) in enumerate(editor.train_token_ids.items())
        } if editor.delta_emb.numel() > 0 else {},
    }
    del editor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# Verdicts + report
# ---------------------------------------------------------------------------
def _verdict(delta_true_empty: float, delta_true_random: float,
             used_thresh: float, weak_thresh: float) -> Dict[str, str]:
    if delta_true_empty >= used_thresh:
        used = "USED"
    elif delta_true_empty >= weak_thresh:
        used = "WEAK"
    else:
        used = "IGNORED"
    if used == "IGNORED":
        grounded = "n/a"
    elif delta_true_random >= 0.5 * delta_true_empty:
        grounded = "SAE-GROUNDED"
    else:
        grounded = "OPAQUE-FLAG-WARNING"
    return {"conditioning": used, "grounding": grounded}


def build_verdicts(tagger_m: Optional[Dict], editor_m: Optional[Dict]) -> Dict:
    v = {}
    if tagger_m:
        f1 = {c: tagger_m["conditions"][c]["macro_f1"] for c in CONDITIONS}
        v["tagger"] = {
            "metric": "macro_f1",
            "true": f1["true"], "empty": f1["empty"], "random": f1["random"],
            "delta_true_empty": f1["true"] - f1["empty"],
            "delta_true_random": f1["true"] - f1["random"],
            **_verdict(f1["true"] - f1["empty"], f1["true"] - f1["random"],
                       used_thresh=0.02, weak_thresh=0.005),
        }
    if editor_m:
        for key, thresh in (("repl_top1_acc", 0.02), ("del_top1_acc", 0.02)):
            acc = {c: editor_m["conditions"][c][key] for c in CONDITIONS}
            v[f"editor_{key}"] = {
                "metric": key,
                "true": acc["true"], "empty": acc["empty"], "random": acc["random"],
                "delta_true_empty": acc["true"] - acc["empty"],
                "delta_true_random": acc["true"] - acc["random"],
                **_verdict(acc["true"] - acc["empty"], acc["true"] - acc["random"],
                           used_thresh=thresh, weak_thresh=thresh / 4),
            }
    return v


def write_report(out_dir: Path, tagger_m, editor_m, verdicts) -> None:
    lines = ["# Tagger / Editor held-out evaluation", ""]

    if tagger_m:
        lines += ["## Tagger (Level 1)", "",
                  f"samples: {tagger_m['n_samples']}", ""]
        base = tagger_m["all_keep_baseline"]
        lines += ["| condition | acc | macro-F1 | span IoU | "
                  + " | ".join(f"{n} F1" for n in OP_NAMES) + " |",
                  "|---|---|---|---|" + "---|" * NUM_OPS]
        for c in CONDITIONS:
            m = tagger_m["conditions"][c]
            row = (f"| {c} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
                   f"| {m['span_iou']:.4f} |")
            row += "".join(f" {m['per_class'][n]['f1']:.4f} |" for n in OP_NAMES)
            lines.append(row)
        lines.append(f"| all-KEEP baseline | {base['accuracy']:.4f} "
                     f"| {base['macro_f1']:.4f} | — |"
                     + " — |" * NUM_OPS)
        lines += ["", "Per-class recall (condition = true):"]
        for n in OP_NAMES:
            pc = tagger_m["conditions"]["true"]["per_class"][n]
            lines.append(f"- {n}: P={pc['precision']:.4f} R={pc['recall']:.4f} "
                         f"F1={pc['f1']:.4f} (support {pc['support']})")
        d = tagger_m["diagnostics"]
        lines += ["", f"cond_scale = {d['cond_scale']:.4f}   "
                      f"‖Proj_A.W‖ = {d['proj_a_weight_norm']:.2f}", ""]

    if editor_m:
        lines += ["## Editor (Level 1)", "",
                  f"samples: {editor_m['n_samples']}", ""]
        cols = ["mean_loss", "keep_top1_acc", "repl_top1_acc", "repl_top5_acc",
                "ins_top1_acc", "ins_top5_acc", "del_top1_acc",
                "overgenerate_rate"]
        lines += ["| condition | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for c in CONDITIONS:
            m = editor_m["conditions"][c]
            lines.append("| " + c + " |"
                         + "".join(f" {m[k]:.4f} |" for k in cols))
        m = editor_m["conditions"]["true"]
        lines += ["",
                  f"[DEL] logit mean at DEL positions: "
                  f"{m['del_logit_mean_at_del']:.2f}  vs at KEEP positions: "
                  f"{m['del_logit_mean_at_keep']:.2f} (separation must be "
                  f"positive and growing for the tied output-column delta "
                  f"to be working)", ""]
        d = editor_m["diagnostics"]
        lines.append(f"cond_scale = {d['cond_scale']:.4f}   "
                     f"‖Proj_A.W‖ = {d['proj_a_weight_norm']:.2f}")
        if d["delta_emb_norms"]:
            lines.append("delta_emb row norms: " + ", ".join(
                f"{k}={v:.3f}" for k, v in d["delta_emb_norms"].items()))
        lines.append("")

    lines += ["## Conditioning causality (Level 2)", ""]
    lines += ["| probe | metric | true | empty | random | Δ(t−e) | Δ(t−r) "
              "| conditioning | grounding |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, v in verdicts.items():
        lines.append(
            f"| {name} | {v['metric']} | {v['true']:.4f} | {v['empty']:.4f} "
            f"| {v['random']:.4f} | {v['delta_true_empty']:+.4f} "
            f"| {v['delta_true_random']:+.4f} | {v['conditioning']} "
            f"| {v['grounding']} |")
    lines += [
        "",
        "Reading guide:",
        "- `conditioning=IGNORED` — the model does not use z_amp/z_sup. "
        "Check cond_scale (→0 means it learned to mute the prefix), "
        "Proj_A gradients, or over-strong sub-sampling (C11).",
        "- `grounding=OPAQUE-FLAG-WARNING` — random features work almost "
        "as well as the true ones: the conditioning acts as a generic "
        "\"something changed\" flag rather than SAE feature identity. "
        "The random-conditioning ablation (§13.2) will likely fail.",
        "",
    ]
    (out_dir / "eval_report.md").write_text("\n".join(lines))


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def make_loader(args, d_sae: int, pad_token_id: int) -> DataLoader:
    ds = CorruptionDataset(args.corruption_dir, shuffle=False,
                           seed=args.seed, infinite=False)
    coll = CorruptionCollator(d_sae=d_sae, pad_token_id=pad_token_id)
    return DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                      collate_fn=coll)


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.tagger_ckpt and not args.editor_ckpt:
        raise SystemExit("[eval] provide --tagger-ckpt and/or --editor-ckpt")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])

    # The two phases iterate the dataset independently but with identical
    # order (shuffle=False) and identical rng seeding, so the paired
    # conditioning is the same for both models.
    tagger_m = None
    editor_m = None
    if args.tagger_ckpt:
        tagger_m = eval_tagger(args, d_sae,
                               make_loader(args, d_sae, tok.pad_token_id))
    if args.editor_ckpt:
        editor_m = eval_editor(args, d_sae,
                               make_loader(args, d_sae, tok.pad_token_id), tok)

    verdicts = build_verdicts(tagger_m, editor_m)
    payload = {
        "corruption_dir": str(args.corruption_dir),
        "llm2vec_dir": str(args.llm2vec_dir),
        "tagger_ckpt": args.tagger_ckpt,
        "editor_ckpt": args.editor_ckpt,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "tagger": tagger_m,
        "editor": editor_m,
        "verdicts": verdicts,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2))
    write_report(out_dir, tagger_m, editor_m, verdicts)

    print(f"[eval] wrote {out_dir}/eval_metrics.json and eval_report.md")
    for name, v in verdicts.items():
        print(f"[eval] {name}: true={v['true']:.4f} empty={v['empty']:.4f} "
              f"random={v['random']:.4f} → conditioning={v['conditioning']} "
              f"grounding={v['grounding']}")


if __name__ == "__main__":
    main()
