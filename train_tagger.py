"""
Train the SAE-LEWIS two-tag tagger (v2) on the corruption stream.

Following LEWIS (Reid & Zhong 2021, §2.1) each token carries two tags:
a 3-class non-insertion op (KEEP / REPL / DEL, class-weighted CE) and a
binary "insert phrase before this token" indicator (BCE with pos_weight).
Golds are derived from the v1 cache by data.CorruptionCollator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from data import CorruptionCollator, CorruptionDataset
from intervene import diff_to_sparse
from lewis_ops import NUM_OPS3, OP3_NAMES
from resume_utils import (
    add_resume_args, find_latest_ckpt,
    load_train_state, save_train_state,
)
from tagger import SAETagger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--proj-a-freeze-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=2000)

    p.add_argument("--k-top", type=int, default=8)
    # Match the editor's conditioning distribution (README C2: shared
    # interface): near-empty conditioning is confined to this probability,
    # not additionally produced by k_amp/k_sup = 0 draws.
    p.add_argument("--empty-cond-prob", type=float, default=0.05)

    p.add_argument("--estimate-class-weights-batches", type=int, default=200,
                   help="Number of warmup batches used to estimate inverse-freq "
                        "op-class weights and the insert-head pos_weight.")
    # 0.5 capped the raw rare-class weight at 2.0; 0.05 keeps the cap at 20x
    # while still bounding the weight against warmup-sample noise. (The v1
    # INS-starvation failure mode is gone — insertion now has its own binary
    # head — but DEL/REPL are still ~10-60x rarer than KEEP.)
    p.add_argument("--class-weight-smoothing", type=float, default=0.05)
    p.add_argument("--ins-loss-weight", type=float, default=1.0,
                   help="Multiplier on the insert-head BCE term.")
    p.add_argument("--ins-pos-weight-cap", type=float, default=50.0,
                   help="Upper bound on the BCE pos_weight (neg/pos ratio) "
                        "for the insert head.")

    p.add_argument("--init-proj-a-from", default=None,
                   help="Editor checkpoint (.pt from train_editor_phaseA.py) "
                        "whose trained Proj_A / type_emb / cond_scale warm-"
                        "start the tagger (README C2: shared conditioning "
                        "interface). Train the editor first, then pass its "
                        "editor-final.pt here.")

    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    add_resume_args(p)
    return p.parse_args()


def _estimate_weights(
    loader, n_batches: int, smoothing: float, pos_weight_cap: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Tuple[int, int]]:
    """Estimate op3 class weights and the insert-head pos_weight from a
    warmup pass. Returns (op_weights, ins_pos_weight, op_counts, (pos, neg))."""
    op_counts = Counter()
    ins_pos, ins_neg = 0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        op = batch["tagger_op3_gold"].view(-1).tolist()
        op_counts.update([g for g in op if g != -100])
        ins = batch["tagger_ins_gold"].view(-1)
        ins_pos += int((ins == 1).sum())
        ins_neg += int((ins == 0).sum())
    total = sum(op_counts.values()) or 1
    freq = np.array([op_counts.get(i, 0) / total for i in range(NUM_OPS3)],
                    dtype=np.float32)
    # Smoothed inverse-frequency; smoothing bounds the rare-class weight.
    weights = 1.0 / (freq + smoothing)
    weights /= weights.mean()
    counts_by_name = {OP3_NAMES[i]: int(op_counts.get(i, 0)) for i in range(NUM_OPS3)}
    pos_weight = min(pos_weight_cap, ins_neg / max(1, ins_pos))
    return (torch.from_numpy(weights.astype(np.float32)),
            torch.tensor(float(pos_weight), dtype=torch.float32),
            counts_by_name, (ins_pos, ins_neg))


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    del_id = tokenizer.convert_tokens_to_ids("[DEL]")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    tagger = SAETagger(args.llm2vec_dir, d_sae=d_sae, dtype=dtype).to(args.device)

    # Warm-start the conditioning interface from the trained editor. The
    # editor's all-position CE through the LM head gives Proj_A a much
    # richer gradient than the tagger's small-head losses, so we train the
    # editor first and initialize the tagger's copy from it.
    if args.init_proj_a_from:
        blob = torch.load(args.init_proj_a_from, map_location="cpu", weights_only=False)
        sd = blob["trainable"]
        tagger.proj_a.weight.data.copy_(sd["proj_a.weight"].to(tagger.proj_a.weight.dtype))
        tagger.proj_a.bias.data.copy_(sd["proj_a.bias"].to(tagger.proj_a.bias.dtype))
        tagger.type_emb.weight.data.copy_(sd["type_emb.weight"].to(tagger.type_emb.weight.dtype))
        if "cond_scale" in sd:
            tagger.cond_scale.data.copy_(sd["cond_scale"].to(tagger.cond_scale.dtype))
        print(f"[tagger] warm-started Proj_A/type_emb/cond_scale from "
              f"{args.init_proj_a_from}")

    optim = torch.optim.AdamW(
        [p for p in tagger.parameters()], lr=args.learning_rate,
    )
    sched = get_linear_schedule_with_warmup(
        optim, args.warmup_steps, args.max_steps,
    )

    ds = CorruptionDataset(args.corruption_dir, shuffle=True, seed=args.seed)
    coll = CorruptionCollator(
        d_sae=d_sae, pad_token_id=tokenizer.pad_token_id,
        sep_token_id=sep_id, del_token_id=del_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=coll,
    )

    # Estimate class weights / pos_weight from a small warmup pass
    print("[tagger] estimating class weights ...")
    cw_iter = iter(DataLoader(
        CorruptionDataset(args.corruption_dir, shuffle=True, seed=args.seed + 1),
        batch_size=args.batch_size, num_workers=0, collate_fn=coll,
    ))
    cw_batches = [next(cw_iter) for _ in range(args.estimate_class_weights_batches)]
    weights, ins_pos_weight, counts_by_name, (ins_pos, ins_neg) = _estimate_weights(
        iter(cw_batches), args.estimate_class_weights_batches,
        args.class_weight_smoothing, args.ins_pos_weight_cap,
    )
    total_counted = sum(counts_by_name.values()) or 1
    print("[tagger] op3 class counts (estimate):")
    for name in OP3_NAMES:
        c = counts_by_name.get(name, 0)
        print(f"          {name:6s}  count={c:>8d}  freq={c/total_counted:.4f}")
    print(f"[tagger] op3 class weights = "
          f"{ {n: round(float(weights[i]), 3) for i, n in enumerate(OP3_NAMES)} }")
    print(f"[tagger] ins head: pos={ins_pos} neg={ins_neg} "
          f"pos_weight={float(ins_pos_weight):.2f} (cap {args.ins_pos_weight_cap})")
    weights = weights.to(args.device)
    ins_pos_weight = ins_pos_weight.to(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    # Resume from the latest <out_dir>/tagger-step{N}.pt if requested.
    # The model class checkpoints ONLY the trainable params (the LLM2Vec
    # backbone is reloaded fresh), so we go through `load_trainable`.
    # Optim / sched / step / RNG come from a sidecar state file.
    if args.resume:
        latest = find_latest_ckpt(out_dir, "tagger")
        if latest is not None:
            ckpt_path, ckpt_step = latest
            print(f"[tagger] RESUME: loading {ckpt_path} (step {ckpt_step})")
            blob = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
            tagger.load_trainable(blob["trainable"])
            restored = load_train_state(ckpt_path, optim, sched, device=args.device)
            step = restored if restored is not None else ckpt_step
            if step >= args.max_steps:
                print(f"[tagger] RESUME: already past max_steps ({step} >= {args.max_steps}); nothing to do")
                tagger.save(str(out_dir / "tagger-final.pt"))
                return

    proj_a_unfrozen = step >= args.proj_a_freeze_steps
    if proj_a_unfrozen:
        for p in tagger.proj_a.parameters():
            p.requires_grad_(True)
    else:
        for p in tagger.proj_a.parameters():
            p.requires_grad_(False)
    loss_window = []
    pbar = tqdm(total=args.max_steps, initial=step,
                desc="[tagger]", unit="step", dynamic_ncols=True)

    for batch in loader:
        if step >= args.max_steps:
            break

        if not proj_a_unfrozen and step >= args.proj_a_freeze_steps:
            for p in tagger.proj_a.parameters():
                p.requires_grad_(True)
            proj_a_unfrozen = True
            print(f"[tagger] step={step} unfroze Proj_A")

        z_X = batch["z_X"]
        z_X_prime = batch["z_X_prime"]
        B = z_X.shape[0]
        z_amp = torch.zeros_like(z_X)
        z_sup = torch.zeros_like(z_X)
        for b in range(B):
            # {1..4}: the fully-empty case is handled by --empty-cond-prob
            # alone (same scheme as the editor).
            k_amp = int(rng.integers(1, 5))
            k_sup = int(rng.integers(1, 5))
            a, s = diff_to_sparse(
                z_X[b], z_X_prime[b],
                k_top=args.k_top, k_amp=k_amp, k_sup=k_sup,
                rng=rng, empty_conditioning_prob=args.empty_cond_prob,
            )
            z_amp[b] = a
            z_sup[b] = s

        out = tagger(
            input_ids=batch["tagger_input_ids"].to(args.device),
            attention_mask=batch["tagger_attention_mask"].to(args.device),
            z_amp=z_amp.to(args.device),
            z_sup=z_sup.to(args.device),
            op_labels=batch["tagger_op3_gold"].to(args.device),
            ins_labels=batch["tagger_ins_gold"].to(args.device),
            class_weights=weights,
            ins_pos_weight=ins_pos_weight,
            ins_loss_weight=args.ins_loss_weight,
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()

        loss_window.append(float(loss.item()))
        if step % args.logging_steps == 0:
            avg = sum(loss_window[-args.logging_steps:]) / max(1, min(len(loss_window), args.logging_steps))
            op_l = float(out["op_loss"].item()) if out["op_loss"] is not None else float("nan")
            ins_l = float(out["ins_loss"].item()) if out["ins_loss"] is not None else float("nan")
            cs = float(tagger.cond_scale.detach().float())
            print(f"[tagger] step={step} loss={avg:.4f} op={op_l:.4f} "
                  f"ins={ins_l:.4f} cond_scale={cs:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")

        if step > 0 and step % args.save_steps == 0:
            ckpt = out_dir / f"tagger-step{step}.pt"
            tagger.save(str(ckpt))
            save_train_state(ckpt, optim, sched, step)
            print(f"[tagger] saved {ckpt}")

        step += 1
        pbar.update(1)

    pbar.close()
    tagger.save(str(out_dir / "tagger-final.pt"))
    print(f"[tagger] done at step {step}")


if __name__ == "__main__":
    main()
