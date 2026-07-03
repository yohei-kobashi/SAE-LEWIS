"""
Phase A — unified corruption pretraining for the SAE-LEWIS editor.

Per-batch:
  1. Load a batch of corruption samples (data.CorruptionDataset).
  2. Sub-sample diff-based conditioning per sample (intervene.diff_to_sparse).
  3. Editor forward with cross-entropy at every editor-input position.
  4. Optimizer step.

Editor input is the LEWIS-faithful concatenation `x' [SEP] x'_c` (built by
CorruptionCollator from the v1 cache); loss is CE over the template segment
only, with copy positions down-weighted (--keep-loss-weight).

Trainable parameters: Proj_A, type_emb[0..2], cond_scale, and the
[MASK]/[INS]/[SEP] embedding deltas. Frozen: encoder, LM head, original
input embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from data import CorruptionCollator, CorruptionDataset
from editor import SAEEditor
from intervene import diff_to_sparse
from resume_utils import (
    add_resume_args, find_latest_ckpt,
    load_train_state, save_train_state,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--proj-a-freeze-steps", type=int, default=1000,
                   help="Freeze Proj_A for the first N steps (warmup).")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--grad-accum-steps", type=int, default=1)

    # Conditioning sub-sampling
    p.add_argument("--k-top", type=int, default=8)
    # 0.15 left the editor conditioning-IGNORED for REPL (held-out probe
    # Δ(true−empty) ≈ 0): together with k_amp/k_sup allowing 0, the model
    # saw weak-or-empty conditioning often enough to learn to ignore it.
    p.add_argument("--empty-cond-prob", type=float, default=0.05)
    # CE weight on copy positions (label == input). Uniform CE (1.0) spends
    # ~95% of the gradient on copying, starving Proj_A; edit positions
    # ([MASK]/[INS] slots) always keep weight 1.
    p.add_argument("--keep-loss-weight", type=float, default=0.2)

    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    add_resume_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    ins_id = tokenizer.convert_tokens_to_ids("[INS]")
    del_id = tokenizer.convert_tokens_to_ids("[DEL]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    mask_id = tokenizer.mask_token_id
    unk_id = tokenizer.unk_token_id
    if (mask_id is None or ins_id is None or del_id is None
            or sep_id is None or sep_id == unk_id):
        raise SystemExit(
            f"[phase-a] tokenizer at {args.llm2vec_dir} lacks the special "
            f"tokens (mask={mask_id}, ins={ins_id}, del={del_id}, "
            f"sep={sep_id}) — re-run scripts/mcgill_merge_and_expand.py "
            f"(idempotent; it adds any missing specials, [SEP] included).")

    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    # [MASK] is trainable too: with the McGill LLM2Vec route the special
    # tokens are added AFTER MNTP/SimCSE by mcgill_merge_and_expand.py
    # (mean-init, never seen a gradient), so unlike the old train_llm2vec.py
    # route there is no "MNTP-trained [MASK] row" to inherit. [SEP] separates
    # x' from the template x'_c (LEWIS's `x SEP x_c` input); [DEL] no longer
    # appears anywhere in the editor's input or output (deletion is the
    # tagger's decision; DEL tokens are removed from the template), so its
    # delta row is gone.
    editor = SAEEditor(
        args.llm2vec_dir, d_sae=d_sae, dtype=dtype,
        train_token_ids={"[INS]": ins_id, "[SEP]": sep_id, "[MASK]": mask_id},
    ).to(args.device)

    # Initial freeze of Proj_A
    for p in editor.proj_a.parameters():
        p.requires_grad_(False)

    trainables = [p for p in editor.parameters() if p.requires_grad]
    print(f"[phase-a] trainable params (warmup): "
          f"{sum(p.numel() for p in trainables):,}")

    optim = torch.optim.AdamW(
        [p for p in editor.parameters()], lr=args.learning_rate,
    )
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
    )

    dataset = CorruptionDataset(args.corruption_dir, shuffle=True, seed=args.seed)
    collator = CorruptionCollator(
        d_sae=d_sae, pad_token_id=tokenizer.pad_token_id,
        sep_token_id=sep_id, del_token_id=del_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collator,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    if args.resume:
        latest = find_latest_ckpt(out_dir, "editor")
        if latest is not None:
            ckpt_path, ckpt_step = latest
            print(f"[phase-a] RESUME: loading {ckpt_path} (step {ckpt_step})")
            blob = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
            editor.load_trainable(blob["trainable"])
            restored = load_train_state(ckpt_path, optim, sched, device=args.device)
            step = restored if restored is not None else ckpt_step
            if step >= args.max_steps:
                print(f"[phase-a] RESUME: already past max_steps ({step} >= {args.max_steps}); nothing to do")
                editor.save(str(out_dir / "editor-final.pt"))
                return

    proj_a_unfrozen = step >= args.proj_a_freeze_steps
    if proj_a_unfrozen:
        for p in editor.proj_a.parameters():
            p.requires_grad_(True)
    loss_window = []
    pbar = tqdm(total=args.max_steps, initial=step,
                desc="[phase-a]", unit="step", dynamic_ncols=True)

    for batch in loader:
        if step >= args.max_steps:
            break

        if not proj_a_unfrozen and step >= args.proj_a_freeze_steps:
            for p in editor.proj_a.parameters():
                p.requires_grad_(True)
            proj_a_unfrozen = True
            print(f"[phase-a] step={step} unfroze Proj_A")

        # Per-sample diff-based sub-sampling
        z_X = batch["z_X"]
        z_X_prime = batch["z_X_prime"]
        B = z_X.shape[0]
        z_amp = torch.zeros_like(z_X)
        z_sup = torch.zeros_like(z_X)
        for b in range(B):
            # {1..4}, not {0..3}: the fully-empty case is handled by
            # --empty-cond-prob alone, so every non-empty conditioning
            # carries at least one amplified and one suppressed feature.
            k_amp = int(rng.integers(1, 5))
            k_sup = int(rng.integers(1, 5))
            a, s = diff_to_sparse(
                z_X[b], z_X_prime[b],
                k_top=args.k_top,
                k_amp=k_amp, k_sup=k_sup,
                rng=rng,
                empty_conditioning_prob=args.empty_cond_prob,
            )
            z_amp[b] = a
            z_sup[b] = s

        out = editor(
            input_ids=batch["editor_input_ids"].to(args.device),
            attention_mask=batch["editor_attention_mask"].to(args.device),
            z_amp=z_amp.to(args.device),
            z_sup=z_sup.to(args.device),
            labels=batch["editor_target_ids"].to(args.device),
            keep_loss_weight=args.keep_loss_weight,
        )
        loss = out["loss"] / args.grad_accum_steps
        loss.backward()

        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()

        loss_window.append(float(loss.item()) * args.grad_accum_steps)
        if step % args.logging_steps == 0:
            avg = sum(loss_window[-args.logging_steps:]) / max(1, min(len(loss_window), args.logging_steps))
            # cond_scale drifting toward 0 means the model is muting the
            # conditioning prefix — the precursor of a conditioning=IGNORED
            # eval verdict. Watch it recover toward ~1 once edit positions
            # dominate the loss.
            cs = float(editor.cond_scale.detach().float())
            print(f"[phase-a] step={step} loss={avg:.4f} "
                  f"cond_scale={cs:.4f} lr={sched.get_last_lr()[0]:.2e}")

        if step > 0 and step % args.save_steps == 0:
            ckpt = out_dir / f"editor-step{step}.pt"
            editor.save(str(ckpt))
            save_train_state(ckpt, optim, sched, step)
            print(f"[phase-a] saved {ckpt}")

        step += 1
        pbar.update(1)

    pbar.close()
    editor.save(str(out_dir / "editor-final.pt"))
    print(f"[phase-a] done at step {step}")


if __name__ == "__main__":
    main()
