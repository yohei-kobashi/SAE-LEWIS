"""
Gold-template probe on LinguaLens: is the OOD failure localization or fill?

eval_lingualens.py measures the full pipeline; its failures confound three
stages (tagger localization, slot-length enumeration, editor fill). This
probe removes the first two: the edit template is built DIRECTLY from the
token-level s1→s2 alignment (gold sites, gold slot lengths — the editor's
in-domain training condition), so whatever accuracy is lost here is lost
in the FILL stage alone.

It simultaneously tests the coordination hypothesis (the editor fills all
positions independently in one forward — README §13.6: coordinated
multi-site rewrites are exactly where the pipeline fails) by comparing
three decode modes over the SAME template:

  parallel  one forward, argmax at every edit position (production decode)
  seq-ltr   fill one position at a time, left→right, feeding each
            prediction back into the input before the next forward —
            matches the training condition (every other position gold)
            up to the correctness of previous fills
  seq-conf  same, but fill the currently most-confident position first
            (MaskGIT-style)

If parallel ≈ sequential ≈ high: fill is fine → the OOD bottleneck is
localization (tagger coverage/data). If parallel ≪ sequential: the
independent-fill architecture is the bottleneck → sequential infilling
is the fix. If both are low: content generation itself does not transfer
OOD (conditioning semantics), and decode order is a sideshow.

Usage:
    python scripts/lingualens_gold_template_probe.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --editor-ckpt runs/prod_gemma_v5/editor/editor-final.pt \
        --output-dir runs/prod_gemma_v5/gold_template_probe \
        --k-amp 64 --k-sup 64 --sample-size 100
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer                             # noqa: E402

from editor import load_editor_from_checkpoint                     # noqa: E402
from eval_lingualens import (                                      # noqa: E402
    _peek_d_sae, diff_intervention, pair_metrics,
)
from model import SAEFeatureExtractor                              # noqa: E402

MODES = ("parallel", "seq-ltr", "seq-conf")
EDIT_BUCKETS = ((1, 1, "1"), (2, 3, "2-3"), (4, 8, "4-8"),
                (9, 10 ** 9, "9+"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--editor-ckpt", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"],
                   default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--conditions", default="true,empty")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gold template from the s1→s2 token alignment
# ---------------------------------------------------------------------------
def build_gold_template(
    src_ids: List[int], tgt_ids: List[int], mask_id: int, ins_id: int,
) -> Tuple[List[int], List[int], List[str]]:
    """(template ids, gold ids, per-position op in {'K','R','I'}).

    DEL runs are dropped from the template (v2 convention: deletion is the
    tagger's decision; the editor never sees deleted tokens). A replace
    opcode of unequal length becomes min-length REPL plus INS (target
    longer) or REPL plus an implicit deletion (source longer). By
    construction the gold-filled template decodes to exactly the target.
    """
    tpl: List[int] = []
    gold: List[int] = []
    ops: List[str] = []
    sm = difflib.SequenceMatcher(None, src_ids, tgt_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            tpl += src_ids[i1:i2]
            gold += tgt_ids[j1:j2]
            ops += ["K"] * (i2 - i1)
        elif tag == "replace":
            n_repl = min(i2 - i1, j2 - j1)
            tpl += [mask_id] * n_repl
            gold += tgt_ids[j1:j1 + n_repl]
            ops += ["R"] * n_repl
            if (j2 - j1) > n_repl:                    # target longer → INS
                extra = (j2 - j1) - n_repl
                tpl += [ins_id] * extra
                gold += tgt_ids[j1 + n_repl:j2]
                ops += ["I"] * extra
            # source longer → the excess source tokens are deleted
        elif tag == "insert":
            tpl += [ins_id] * (j2 - j1)
            gold += tgt_ids[j1:j2]
            ops += ["I"] * (j2 - j1)
        # 'delete': dropped
    return tpl, gold, ops


# ---------------------------------------------------------------------------
# Decode modes
# ---------------------------------------------------------------------------
@torch.no_grad()
def fill_template(
    mode: str,
    full_ids: List[int],
    tpl_start: int,
    edit_pos: List[int],          # template-segment coordinates
    gold_seg: List[int],
    editor,
    z_amp: torch.Tensor,          # (1, d_sae) on device
    z_sup: torch.Tensor,
    marker_ids: List[int],
    device: str,
) -> Tuple[List[int], List[Dict]]:
    """Fill the template's edit positions; return (filled template-segment
    ids, per-position results). top-5 is taken from the logits available
    at the moment each position is filled."""
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)

    def forward_logits():
        out = editor(input_ids=ids, attention_mask=attn,
                     z_amp=z_amp, z_sup=z_sup)
        lg = out["logits"][0, tpl_start:].float()
        lg[:, marker_ids] = float("-inf")
        return lg

    results: List[Dict] = []
    if mode == "parallel":
        lg = forward_logits()
        for p in edit_pos:
            top5 = lg[p].topk(5).indices.tolist()
            pred = top5[0]
            ids[0, tpl_start + p] = pred
            results.append({"pos": p, "pred": pred, "gold": gold_seg[p],
                            "top1": pred == gold_seg[p],
                            "top5": gold_seg[p] in top5})
    else:
        remaining = list(edit_pos)
        while remaining:
            lg = forward_logits()
            if mode == "seq-ltr":
                p = remaining.pop(0)
            else:                                     # seq-conf
                probs = torch.softmax(lg[remaining], dim=-1).max(dim=-1).values
                pi = int(probs.argmax())
                p = remaining.pop(pi)
            top5 = lg[p].topk(5).indices.tolist()
            pred = top5[0]
            ids[0, tpl_start + p] = pred
            results.append({"pos": p, "pred": pred, "gold": gold_seg[p],
                            "top1": pred == gold_seg[p],
                            "top5": gold_seg[p] in top5})
    return ids[0, tpl_start:].cpu().tolist(), results


def _bucket(n: int) -> str:
    for lo, hi, name in EDIT_BUCKETS:
        if lo <= n <= hi:
            return name
    return "9+"


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}; pick from {MODES}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    from datasets import load_dataset
    print(f"[probe] loading {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:min(args.sample_size, len(order))]
    print(f"[probe] {len(ds)} pairs, sampling {len(chosen)}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    mask_id = int(tokenizer.mask_token_id)
    ins_id = int(tokenizer.convert_tokens_to_ids("[INS]"))
    sep_id = int(tokenizer.convert_tokens_to_ids("[SEP]"))
    del_id = int(tokenizer.convert_tokens_to_ids("[DEL]"))
    bos_id = tokenizer.bos_token_id
    marker_ids = [mask_id, ins_id, sep_id, del_id]

    print("[probe] loading editor / SAE extractor")
    editor = load_editor_from_checkpoint(
        args.llm2vec_dir, args.editor_ckpt,
        d_sae=_peek_d_sae(args.editor_ckpt), dtype=dtype,
    ).to(args.device).eval()
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()

    # tok[cond][mode] counters; pairagg[cond][mode][key] lists
    tok = {c: {m: defaultdict(int) for m in modes} for c in conditions}
    pairagg = {c: {m: defaultdict(list) for m in modes} for c in conditions}
    # bucketed, true condition only
    btok = {m: defaultdict(lambda: defaultdict(int)) for m in modes}
    bpair = {m: defaultdict(lambda: defaultdict(list)) for m in modes}
    records: List[Dict] = []
    skipped_no_fill = 0
    gold_mismatch = 0

    for step, k in enumerate(chosen):
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids

        tpl, gold, ops = build_gold_template(src_ids, tgt_ids, mask_id, ins_id)
        offset = 1 if (bos_id is not None and tpl and tpl[0] == bos_id) else 0
        tpl_seg, gold_seg, ops_seg = tpl[offset:], gold[offset:], ops[offset:]
        edit_pos = [j for j, o in enumerate(ops_seg) if o != "K"]
        if not edit_pos:
            skipped_no_fill += 1        # pure-deletion or identical pair
            continue
        # Sanity: the gold-filled template must decode to the target.
        gold_dec = ([bos_id] if offset else []) + list(gold_seg)
        if tokenizer.decode(gold_dec, skip_special_tokens=True).strip() \
                != tokenizer.decode(tgt_ids, skip_special_tokens=True).strip():
            gold_mismatch += 1

        full = src_ids + [sep_id] + tpl_seg
        tpl_start = len(src_ids) + 1
        n_edit = len(edit_pos)
        bucket = _bucket(n_edit)

        with torch.no_grad():
            z_src = extractor.pool_max_topk(
                extractor.encode_text(src), args.pool_topk).float().cpu()
            z_tgt = extractor.pool_max_topk(
                extractor.encode_text(tgt), args.pool_topk).float().cpu()
        z_amp_t, z_sup_t = diff_intervention(
            z_src, z_tgt, args.k_amp, args.k_sup)
        zvar = {
            "true": (z_amp_t, z_sup_t),
            "empty": (torch.zeros_like(z_amp_t), torch.zeros_like(z_sup_t)),
        }

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_edit": n_edit,
               "n_repl": sum(1 for o in ops_seg if o == "R"),
               "n_ins": sum(1 for o in ops_seg if o == "I"),
               "outputs": {}}
        for c in conditions:
            za = zvar[c][0].unsqueeze(0).to(args.device)
            zs = zvar[c][1].unsqueeze(0).to(args.device)
            rec["outputs"][c] = {}
            for m in modes:
                seg, results = fill_template(
                    m, full, tpl_start, edit_pos, gold_seg,
                    editor, za, zs, marker_ids, args.device)
                out_ids = ([bos_id] if offset else []) + seg
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                pm = pair_metrics(out_text, src, tgt)

                t = tok[c][m]
                for r in results:
                    o = ops_seg[r["pos"]]
                    t["n"] += 1
                    t["top1"] += int(r["top1"])
                    t["top5"] += int(r["top5"])
                    t[f"{o}_n"] += 1
                    t[f"{o}_top1"] += int(r["top1"])
                pa = pairagg[c][m]
                pa["exact"].append(pm["exact_match"])
                pa["sim_target"].append(pm["sim_target"])
                pa["tok_acc"].append(
                    sum(r["top1"] for r in results) / n_edit)
                if c == "true":
                    bt = btok[m][bucket]
                    bt["n"] += n_edit
                    bt["top1"] += sum(r["top1"] for r in results)
                    bp = bpair[m][bucket]
                    bp["exact"].append(pm["exact_match"])
                    bp["sim_target"].append(pm["sim_target"])
                rec["outputs"][c][m] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"],
                    "tok_top1": sum(r["top1"] for r in results) / n_edit,
                }
        records.append(rec)
        if (step + 1) % 10 == 0:
            print(f"[probe] {step + 1}/{len(chosen)} pairs "
                  f"(no-fill skipped={skipped_no_fill})")

    # ---- report ------------------------------------------------------------
    def _tokrow(t, key, nkey):
        return t[key] / t[nkey] if t[nkey] else float("nan")

    n_scored = len(records)
    lines = ["# LinguaLens gold-template probe (fill stage isolated)", "",
             f"pairs scored: {n_scored}  "
             f"(skipped, no fill positions: {skipped_no_fill}; "
             f"gold-reconstruction mismatches: {gold_mismatch})",
             f"conditioning: k_amp={args.k_amp} k_sup={args.k_sup} "
             f"pool_topk={args.pool_topk}; editor: {args.editor_ckpt}", ""]

    lines += ["## Fill accuracy over gold templates", "",
              "| condition | mode | edit-tok top-1 | top-5 | REPL top-1 "
              "| INS top-1 | exact | sim_target |",
              "|---|---|---|---|---|---|---|---|"]
    payload = {"n_scored": n_scored, "skipped_no_fill": skipped_no_fill,
               "gold_mismatch": gold_mismatch, "conditions": {}}
    for c in conditions:
        payload["conditions"][c] = {}
        for m in modes:
            t = tok[c][m]
            pa = pairagg[c][m]
            row = {
                "tok_top1": _tokrow(t, "top1", "n"),
                "tok_top5": _tokrow(t, "top5", "n"),
                "repl_top1": _tokrow(t, "R_top1", "R_n"),
                "ins_top1": _tokrow(t, "I_top1", "I_n"),
                "exact": float(np.mean(pa["exact"])),
                "sim_target": float(np.mean(pa["sim_target"])),
            }
            payload["conditions"][c][m] = row
            lines.append(
                f"| {c} | {m} | {row['tok_top1']:.4f} | {row['tok_top5']:.4f} "
                f"| {row['repl_top1']:.4f} | {row['ins_top1']:.4f} "
                f"| {row['exact']:.4f} | {row['sim_target']:.4f} |")
    lines.append("")

    lines += ["## Multi-site breakdown (condition = true)", "",
              "| n_edit | pairs | "
              + " | ".join(f"{m} top-1" for m in modes) + " | "
              + " | ".join(f"{m} exact" for m in modes) + " |",
              "|---|---|" + "---|" * (2 * len(modes))]
    payload["buckets"] = {}
    for _lo, _hi, name in EDIT_BUCKETS:
        n_pairs = len(bpair[modes[0]][name]["exact"]) \
            if modes and name in bpair[modes[0]] else 0
        if not n_pairs:
            continue
        row = {"pairs": n_pairs}
        cells_acc, cells_ex = [], []
        for m in modes:
            bt = btok[m][name]
            acc = _tokrow(bt, "top1", "n")
            ex = float(np.mean(bpair[m][name]["exact"]))
            row[m] = {"tok_top1": acc, "exact": ex,
                      "sim_target": float(np.mean(
                          bpair[m][name]["sim_target"]))}
            cells_acc.append(f"{acc:.4f}")
            cells_ex.append(f"{ex:.4f}")
        payload["buckets"][name] = row
        lines.append(f"| {name} | {n_pairs} | " + " | ".join(cells_acc)
                     + " | " + " | ".join(cells_ex) + " |")
    lines += [
        "",
        "Reading guide:",
        "- parallel ≈ sequential AND high → fill is fine; the OOD "
        "bottleneck is tagger localization (coverage/data).",
        "- parallel ≪ sequential (especially at n_edit ≥ 4) → independent "
        "per-position fill cannot coordinate multi-site rewrites; "
        "sequential infilling is the architectural fix.",
        "- both low, true ≈ empty → fill content itself does not transfer "
        "OOD; decode order is a sideshow.",
        "",
    ]
    (out_dir / "probe_report.md").write_text("\n".join(lines))
    (out_dir / "probe_metrics.json").write_text(json.dumps(payload, indent=2))
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n".join(lines))
    print(f"[probe] wrote {out_dir}/probe_report.md, probe_metrics.json, "
          f"records.jsonl")


if __name__ == "__main__":
    main()
