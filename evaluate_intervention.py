"""
End-to-end SAE-LEWIS inference and a thin metrics layer (v2).

  edit(text, spec, strength):
    1. tokenize
    2. tagger.predict_ops → (op3, ins_before)  [two-tag, LEWIS §2.1]
    3. apply_ops_for_editor → edit template x'_c (DEL tokens removed)
    4. enumerate INS slot counts in {1..L_MAX}^G
    5. editor forward on `x' [SEP] x'_c` per template → argmax decode over
       the template segment (special-token logits suppressed)
    6. ranker → top-1

Deletion is realised entirely by the tagger: DEL-tagged tokens never enter
the editor template, exactly as in LEWIS where the BART generator does not
see deleted tokens.

Usage:
    python evaluate_intervention.py \
        --llm2vec-dir <path> \
        --tagger-ckpt <path> \
        --editor-ckpt <path> \
        --mu <stage0 mu.npy> \
        --text "..." --spec "+1234" "-5678"
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from editor import load_editor_from_checkpoint
from intervene import FeatureSpec, build_intervention_vectors
from lewis_ops import (
    OP3_KEEP,
    apply_ops_for_editor,
    decode_editor_output,
    expand_ins_gap,
)
from model import BidirectionalLLM, SAEFeatureExtractor, load_causal_gemma
from ranker import Ranker, RankerWeights
from tagger import load_tagger_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--tagger-ckpt", required=True)
    p.add_argument("--editor-ckpt", required=True)
    p.add_argument("--mu", required=True, help="Path to mu.npy from precompute_sae.py.")

    p.add_argument("--llm", default="google/gemma-2-2b",
                   help="Base Gemma used by SAE extractor / ranker fluency.")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", default="layer_12/width_16k/average_l0_71/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"], default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--text", required=True)
    p.add_argument("--spec", nargs="+", required=True,
                   help="Intervention spec entries: '+1234' (amp) or '-5678' (sup).")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--l-max", type=int, default=5)
    p.add_argument("--ins-threshold", type=float, default=0.5,
                   help="Sigmoid threshold for the tagger's insert head; "
                        "raise toward 1.0 to suppress spurious INS gaps.")
    p.add_argument("--op-thresholds", default="0.0,0.9",
                   help="Comma list of edit-plan strictness levels (0.0 = "
                        "plain argmax; τ > 0 keeps REPL/DEL only at softmax "
                        "prob ≥ τ). Each level yields one candidate plan; "
                        "the unedited input is always a candidate too.")
    p.add_argument("--max-templates", type=int, default=256,
                   help="Editor-forward budget across all plans; plans "
                        "whose enumeration exceeds the remainder are "
                        "dropped (the identity candidate always survives).")

    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


class TemplateBudgetExceeded(RuntimeError):
    """Kept for import compatibility. Since the identity candidate joined
    the ranker pool, edit_once degrades gracefully (plans whose enumeration
    exceeds the budget are dropped) instead of raising."""


@torch.no_grad()
def edit_once(
    *,
    text: str,
    z_amp_full: torch.Tensor,        # (d_sae,) non-negative intervention
    z_sup_full: torch.Tensor,        # (d_sae,)
    tagger,
    editor,
    ranker: Ranker,
    tokenizer,
    l_max: int,
    device: str,
    ins_threshold: float = 0.5,
    op_thresholds: Tuple[float, ...] = (0.0, 0.9),
    max_templates: int = 256,
    verbose: bool = True,
) -> str:
    """Tagger → candidate plans → template enumeration → ranker top-1.

    Candidate diversity (LEWIS's beam analog): one tagger forward yields
    several EDIT PLANS at different strictness levels (`op_thresholds`:
    0.0 = plain argmax; τ > 0 keeps a REPL/DEL decision only when its
    softmax probability ≥ τ, and raises the insert bar to max(τ,
    ins_threshold)), plus the UNEDITED input as an always-present identity
    candidate. The ranker chooses across all of them, so a trigger-happy
    tagger can be overruled instead of silently degrading the output.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc.input_ids                         # (1, T)
    attn = enc.attention_mask
    z_amp_dev = z_amp_full.unsqueeze(0).to(device)
    z_sup_dev = z_sup_full.unsqueeze(0).to(device)

    mask_id = int(tokenizer.mask_token_id)
    ins_id = int(tokenizer.convert_tokens_to_ids("[INS]"))
    sep_id = int(tokenizer.convert_tokens_to_ids("[SEP]"))
    del_id = int(tokenizer.convert_tokens_to_ids("[DEL]"))
    bos_id = tokenizer.bos_token_id

    # 1. Tagger — one forward, probabilistic outputs
    t_out = tagger(input_ids, attn, z_amp_dev, z_sup_dev)
    op_probs = torch.softmax(t_out["op_logits"].float(), dim=-1)[0].cpu()  # (T, 3)
    ins_prob = torch.sigmoid(t_out["ins_logits"].float())[0].cpu()         # (T,)
    op_argmax = op_probs.argmax(dim=-1).tolist()
    token_ids = input_ids[0].cpu().tolist()

    # 2. Edit plans at increasing strictness
    plans = []
    seen = set()
    for tau in op_thresholds:
        op3 = list(op_argmax)
        ins_bar = max(ins_threshold, tau) if tau > 0 else ins_threshold
        for i in range(len(op3)):
            if op3[i] != OP3_KEEP and float(op_probs[i, op3[i]]) < tau:
                op3[i] = OP3_KEEP
        ins_before = [1 if float(p) >= ins_bar else 0 for p in ins_prob]
        # <bos> is structural: never delete/replace it or insert before it
        # (sentence-initial insertion is ins_before on the first real token).
        if bos_id is not None and token_ids and token_ids[0] == bos_id:
            op3[0] = OP3_KEEP
            ins_before[0] = 0
        key = (tuple(op3), tuple(ins_before))
        if key in seen:
            continue
        seen.add(key)
        plans.append((tau, op3, ins_before))

    marker_ids = [mask_id, ins_id, sep_id, del_id]
    candidates: List[List[int]] = []
    ins_counts: List[int] = []
    # Identity candidate: always present, so the ranker can decline to edit.
    candidates.append(list(token_ids))
    ins_counts.append(0)

    templates_used = 0
    for tau, op3, ins_before in plans:
        ed_in = apply_ops_for_editor(
            token_ids, op3, ins_before,
            mask_token_id=mask_id, ins_token_id=ins_id,
        )
        G = len(ed_in.ins_gaps)
        n_templates = (l_max ** G) if G > 0 else 1
        if templates_used + n_templates > max_templates:
            if verbose:
                print(f"[edit] plan τ={tau}: {n_templates} templates would "
                      f"exceed --max-templates {max_templates} — dropped")
            continue
        templates_used += n_templates
        if verbose:
            from collections import Counter
            op_counts = Counter(
                "KEEP" if o == 0 else ("REPL" if o == 1 else "DEL")
                for o in op3)
            print(f"[edit] plan τ={tau}: {dict(op_counts)} "
                  f"ins_gaps={int(sum(ins_before))} templates={n_templates}")

        slot_choices = ([tuple()] if G == 0 else
                        list(itertools.product(range(1, l_max + 1), repeat=G)))
        for choice in slot_choices:
            cur = ed_in
            for gi, k in enumerate(choice):
                cur = expand_ins_gap(cur, gi, k, ins_id)
            tpl_ids = cur.input_ids.tolist()
            tpl_ops = cur.op_per_pos
            offset = 0
            if bos_id is not None and tpl_ids and tpl_ids[0] == bos_id:
                offset = 1  # skip the duplicated <bos> in the template segment
            full = token_ids + [sep_id] + tpl_ids[offset:]
            tpl_start = len(token_ids) + 1

            ids = torch.tensor([full], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            out = editor(
                input_ids=ids, attention_mask=mask,
                z_amp=z_amp_dev, z_sup=z_sup_dev,
            )
            logits = out["logits"][0, tpl_start:]           # template segment
            logits[:, marker_ids] = float("-inf")           # never emit markers
            argmax = logits.argmax(dim=-1).cpu().numpy()
            decoded = decode_editor_output(
                np.asarray(tpl_ids[offset:], dtype=np.int64), argmax,
                tpl_ops[offset:],
            )
            if offset:
                decoded = [bos_id] + decoded
            candidates.append(decoded)
            ins_counts.append(sum(choice) if choice else 0)

    # 4. rank
    scores = ranker.rank(
        candidates=candidates, input_ids=token_ids,
        z_amp=z_amp_full.to(device), z_sup=z_sup_full.to(device),
        num_ins_slots_per_cand=ins_counts,
    )
    best = int(np.argmax(scores))
    return tokenizer.decode(candidates[best], skip_special_tokens=True)


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    mu = np.load(args.mu)
    d_sae = int(mu.shape[0])

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)

    print("[eval] loading tagger / editor")
    tagger = load_tagger_from_checkpoint(args.llm2vec_dir, args.tagger_ckpt, d_sae, dtype=dtype)
    tagger = tagger.to(args.device).eval()
    editor = load_editor_from_checkpoint(args.llm2vec_dir, args.editor_ckpt, d_sae, dtype=dtype)
    editor = editor.to(args.device).eval()

    print("[eval] loading ranker components (SAE / causal / bidirectional)")
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    )
    causal, _ = load_causal_gemma(args.llm2vec_dir)
    bid = BidirectionalLLM(args.llm2vec_dir, dtype=dtype)
    ranker = Ranker(extractor, causal, bid, RankerWeights(), device=args.device)

    specs = [FeatureSpec.parse(s) for s in args.spec]
    z_amp_full, z_sup_full = build_intervention_vectors(specs, mu, args.strength)
    op_taus = tuple(float(t) for t in args.op_thresholds.split(","))
    out = edit_once(
        text=args.text, z_amp_full=z_amp_full, z_sup_full=z_sup_full,
        tagger=tagger, editor=editor, ranker=ranker, tokenizer=tokenizer,
        l_max=args.l_max, device=args.device,
        ins_threshold=args.ins_threshold, op_thresholds=op_taus,
        max_templates=args.max_templates,
    )
    print("== edited ==")
    print(out)


if __name__ == "__main__":
    main()
