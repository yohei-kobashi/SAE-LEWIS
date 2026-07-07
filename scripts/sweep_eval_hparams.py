"""
Hyperparameter sweep for the split evaluation of tagger / editor.

Sweeps the EVALUATION-TIME conditioning hyperparameters — the diff
candidate pool size (`k_top`) and the per-sample number of conditioning
features (`k_amp`/`k_sup`, fixed or uniform-range) — plus the tagger's
INS threshold, on a held-out corruption split. Models are loaded ONCE;
each grid point re-runs only the forward passes under the `true`
condition (the sweep maximises absolute accuracy; causality verdicts
still come from a full three-condition eval_tagger_editor.py run at the
chosen setting — this script prints that command).

Run this on the SELECTION split (corruption_seldev), never on the
reporting dev split, so the tuned setting is not fitted to the reported
numbers.

Usage:
    python scripts/sweep_eval_hparams.py \
        --corruption-dir runs/prod_gemma_v4/corruption_seldev \
        --llm2vec-dir    runs/mcgill_gemma_repro_3k/final \
        --tagger-ckpt    runs/prod_gemma_v4/tagger/tagger-final.pt \
        --editor-ckpt    runs/prod_gemma_v4/editor/editor-final.pt \
        --output-dir     runs/prod_gemma_v4/hparam_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editor import load_editor_from_checkpoint          # noqa: E402
from eval_tagger_editor import (                         # noqa: E402
    _dtype, eval_editor, eval_tagger, make_loader,
)
from tagger import load_tagger_from_checkpoint           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True,
                   help="SELECTION split (corruption_seldev) — do NOT "
                        "sweep on the reporting dev split.")
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--tagger-ckpt", default=None)
    p.add_argument("--editor-ckpt", default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grid-k-top", default="4,8,16,32",
                   help="Diff candidate pool sizes (training used 8).")
    p.add_argument("--grid-k", default="1-4,2,4,8",
                   help="k_amp/k_sup specs, applied to both jointly: "
                        "'LO-HI' = per-sample uniform, bare int = fixed "
                        "(training used '1-4').")
    p.add_argument("--ins-thresholds", default="0.3,0.5,0.7,0.8,0.9",
                   help="Tagger INS thresholds, all scored from the same "
                        "forwards.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.tagger_ckpt and not args.editor_ckpt:
        raise SystemExit("[sweep] provide --tagger-ckpt and/or --editor-ckpt")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    dtype = _dtype(args.llm_dtype)

    tagger = editor = None
    if args.tagger_ckpt:
        print(f"[sweep] loading tagger from {args.tagger_ckpt}")
        tagger = load_tagger_from_checkpoint(
            args.llm2vec_dir, args.tagger_ckpt, d_sae=d_sae, dtype=dtype,
        ).to(args.device)
        tagger.eval()
    if args.editor_ckpt:
        print(f"[sweep] loading editor from {args.editor_ckpt}")
        editor = load_editor_from_checkpoint(
            args.llm2vec_dir, args.editor_ckpt, d_sae=d_sae, dtype=dtype,
        ).to(args.device)
        editor.eval()

    k_tops = [int(x) for x in args.grid_k_top.split(",")]
    k_specs = [s.strip() for s in args.grid_k.split(",")]
    thresholds = [float(t) for t in args.ins_thresholds.split(",")]

    rows = []
    n_cfg = len(k_tops) * len(k_specs)
    for i, (k_top, k_spec) in enumerate(
            ((kt, ks) for kt in k_tops for ks in k_specs), 1):
        print(f"[sweep] config {i}/{n_cfg}: k_top={k_top} k={k_spec}")
        cfg = copy(args)
        cfg.k_top = k_top
        cfg.k_amp = k_spec
        cfg.k_sup = k_spec
        cfg.ins_threshold = thresholds[0]

        row = {"k_top": k_top, "k": k_spec}
        if tagger is not None:
            tm = eval_tagger(cfg, d_sae, make_loader(cfg, d_sae, tok),
                             tagger=tagger, conditions=["true"],
                             ins_thresholds=thresholds[1:])
            t = tm["conditions"]["true"]
            by_thr = t.get("ins_by_threshold",
                           {f"{thresholds[0]:g}": t["ins"]})
            best_thr, best_ins = max(by_thr.items(),
                                     key=lambda kv: kv[1]["f1"])
            row.update({
                "tagger_macro_f1": t["macro_f1"],
                "tagger_op_acc": t["accuracy"],
                "tagger_ins_f1": best_ins["f1"],
                "tagger_ins_thr": float(best_thr),
                "tagger_ins_by_thr": {k2: v["f1"]
                                      for k2, v in by_thr.items()},
            })
        if editor is not None:
            em = eval_editor(cfg, d_sae, make_loader(cfg, d_sae, tok), tok,
                             editor=editor, conditions=["true"])
            e = em["conditions"]["true"]
            row.update({
                "editor_repl_top1": e["repl_top1_acc"],
                "editor_ins_top1": e["ins_top1_acc"],
                "editor_repl_top5": e["repl_top5_acc"],
                "editor_ins_top5": e["ins_top5_acc"],
                "editor_mean_loss": e["mean_loss"],
            })
        rows.append(row)

    (out_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2))

    # ---- report ----------------------------------------------------------
    lines = ["# Eval-hyperparameter sweep (condition: true)", "",
             f"split: `{args.corruption_dir}`  samples/config: "
             f"{args.max_samples}  seed: {args.seed}",
             "", "training reference: k_top=8, k='1-4', ins_threshold=0.5",
             ""]
    if tagger is not None:
        srt = sorted(rows, key=lambda r: -r["tagger_macro_f1"])
        lines += ["## Tagger (sorted by macro-F1)", "",
                  "| k_top | k | macro_f1 | op_acc | best ins_f1 | @thr |",
                  "|---|---|---|---|---|---|"]
        for r in srt:
            lines.append(
                f"| {r['k_top']} | {r['k']} | {r['tagger_macro_f1']:.4f} "
                f"| {r['tagger_op_acc']:.4f} | {r['tagger_ins_f1']:.4f} "
                f"| {r['tagger_ins_thr']:g} |")
        best_t = srt[0]
        lines.append("")
    if editor is not None:
        srt = sorted(rows, key=lambda r: -(r["editor_repl_top1"]
                                           + r["editor_ins_top1"]))
        lines += ["## Editor (sorted by repl_top1 + ins_top1)", "",
                  "| k_top | k | repl_top1 | ins_top1 | repl_top5 "
                  "| ins_top5 | mean_loss |",
                  "|---|---|---|---|---|---|---|"]
        for r in srt:
            lines.append(
                f"| {r['k_top']} | {r['k']} | {r['editor_repl_top1']:.4f} "
                f"| {r['editor_ins_top1']:.4f} | {r['editor_repl_top5']:.4f} "
                f"| {r['editor_ins_top5']:.4f} "
                f"| {r['editor_mean_loss']:.4f} |")
        best_e = srt[0]
        lines.append("")

    ref = best_t if tagger is not None else best_e
    cmd = ["python eval_tagger_editor.py \\",
           "    --corruption-dir <reporting dev split> \\",
           f"    --llm2vec-dir {args.llm2vec_dir} \\"]
    if args.tagger_ckpt:
        cmd.append(f"    --tagger-ckpt {args.tagger_ckpt} \\")
    if args.editor_ckpt:
        cmd.append(f"    --editor-ckpt {args.editor_ckpt} \\")
    cmd.append("    --output-dir <out> \\")
    cmd.append(f"    --k-top {ref['k_top']} --k-amp {ref['k']} "
               f"--k-sup {ref['k']}"
               + (f" --ins-threshold {ref['tagger_ins_thr']:g}"
                  if tagger is not None else ""))
    lines += [
        "## Confirm on the REPORTING dev split (all three conditions)", "",
        "```bash", *cmd, "```", "",
        "The sweep scores only the `true` condition; re-check the "
        "conditioning verdicts (Δ true−empty / true−random) at the chosen "
        "setting before adopting it.",
    ]
    report = "\n".join(lines)
    (out_dir / "sweep_report.md").write_text(report)
    print(f"[sweep] wrote {out_dir}/sweep_report.md and sweep_results.json")
    if tagger is not None:
        print(f"[sweep] tagger best: k_top={best_t['k_top']} k={best_t['k']} "
              f"macro_f1={best_t['tagger_macro_f1']:.4f} "
              f"ins_f1={best_t['tagger_ins_f1']:.4f}@{best_t['tagger_ins_thr']:g}")
    if editor is not None:
        print(f"[sweep] editor best: k_top={best_e['k_top']} k={best_e['k']} "
              f"repl_top1={best_e['editor_repl_top1']:.4f} "
              f"ins_top1={best_e['editor_ins_top1']:.4f}")


if __name__ == "__main__":
    main()
