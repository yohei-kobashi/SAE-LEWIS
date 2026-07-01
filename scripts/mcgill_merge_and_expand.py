"""
Post-training bridge for McGill's LLM2Vec pipeline → SAE-LEWIS downstream.

McGill's `run_mntp.py` / `run_simcse.py` save LoRA adapters (`adapter_config.json`
+ `adapter_model.safetensors`) on top of the base checkpoint. Our downstream
stages (corruption, tagger, editor, length_head, eval) expect a plain HF-format
`AutoModelForCausalLM` at `--llm2vec-dir` — with the LoRA changes already merged
in — plus a tokenizer that has `[INS]`, `[DEL]`, `[MASK]` in the vocabulary.

This script bridges the two:

  1. Load the base model (LlamaForCausalLM etc.).
  2. Manually apply the MNTP LoRA delta to each targeted linear layer.
  3. Manually apply the SimCSE LoRA delta.
  4. Add `[INS]`, `[DEL]`, `[MASK]` to the tokenizer if missing.
  5. resize_token_embeddings() so the new rows are initialised via the
     mean-of-existing trick (same as `train_llm2vec.py`).
  6. Save the merged + expanded model + tokenizer as a drop-in
     `--llm2vec-dir` for downstream.

Why manual merge and not peft.PeftModel.from_pretrained().merge_and_unload():

  McGill trains its LoRA on top of an *inner* model class (LlamaBiForMNTP /
  LlamaBiForSimCSE) that inherits from LlamaModel, not LlamaForCausalLM. So
  the adapter tensor keys look like
      base_model.model.layers.X.self_attn.q_proj.lora_B.weight
  whereas peft, given a LlamaForCausalLM base, expects
      base_model.model.model.layers.X.self_attn.q_proj.lora_B.weight
  (note the extra `.model.` — LlamaForCausalLM has its layers under an inner
  LlamaModel). peft's load_state_dict runs with strict=False and silently
  ignores every mismatched key, so `PeftModel.from_pretrained` returns without
  loading any adapter weights. The subsequent merge_and_unload therefore adds
  `alpha/r · B_init · A_init = 0` and the base is unchanged. Diagnosed via
  scripts/diagnose_mcgill_adapter.py, which showed the adapter file *did*
  have well-trained lora_B tensors (max_norm ~1.5 for MNTP, ~0.53 for SimCSE).

  Manual merge sidesteps the key mismatch entirely: we read A / B, compute
  the delta ourselves, and write it directly to base.model.layers[i]... .

The expanded rows are NOT trained here (McGill's training didn't see them
either); they'll get trained in the editor/tagger stage. This matches our
previous full-FT setup's behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-2b",
                   help="HF id of the underlying base model.")
    p.add_argument("--mntp-adapter", required=True,
                   help="Path to McGill's MNTP training output (a directory "
                        "containing adapter_config.json + adapter_model.*).")
    p.add_argument("--simcse-adapter", default=None,
                   help="Path to McGill's SimCSE training output. Optional — "
                        "if omitted we save Bi+MNTP only.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the merged + expanded HF checkpoint.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--add-special-tokens", nargs="+",
                   default=["[INS]", "[DEL]", "[MASK]"],
                   help="Tokens to add to the tokenizer before saving. "
                        "[MASK] is only added if the tokenizer has no "
                        "mask_token already; [INS] / [DEL] are new either "
                        "way (SAE-LEWIS-specific).")
    return p.parse_args()


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / name
        if not p.exists():
            continue
        if p.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(p))
        return torch.load(p, map_location="cpu", weights_only=True)
    raise SystemExit(f"[bridge] no adapter_model.{{safetensors,bin}} at {adapter_dir}")


def _apply_lora_delta(base, adapter_dir: str, name: str) -> None:
    """Read a peft LoRA adapter and merge its delta directly into `base`.

    See the module docstring for why we don't use peft.merge_and_unload here.

    Iterates over the adapter's (A, B) tensor pairs, computes the low-rank
    delta `scaling · B @ A` for each targeted linear layer, resolves the
    matching submodule in the CausalLM base (probing both the McGill
    LlamaModel-flavoured path and the standard LlamaForCausalLM
    `.model.…` path), and adds the delta to that submodule's weight in
    place. Raises SystemExit if nothing merged, which strictly beats
    peft's silent no-op behaviour.
    """
    ad = Path(adapter_dir)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scaling = alpha / r
    print(f"[{name}] adapter r={r}, lora_alpha={alpha}, scaling={scaling:.3f}")

    tensors = _load_adapter_tensors(ad)

    # Group by module_path (the tensor key without the trailing .lora_{A,B}.weight)
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in tensors.items():
        if ".lora_A.weight" in k:
            groups.setdefault(k.replace(".lora_A.weight", ""), {})["A"] = v
        elif ".lora_B.weight" in k:
            groups.setdefault(k.replace(".lora_B.weight", ""), {})["B"] = v

    n_applied = 0
    n_missing_pair = 0
    n_no_submodule = 0
    n_shape_mismatch = 0
    delta_norms: list[float] = []

    for path, ab in groups.items():
        if "A" not in ab or "B" not in ab:
            n_missing_pair += 1
            continue

        # Strip peft's "base_model.model." wrapping prefix — everything after
        # that is the path into the inner training-time model class.
        clean_path = path
        for prefix in ("base_model.model.", "base_model."):
            if clean_path.startswith(prefix):
                clean_path = clean_path[len(prefix):]
                break

        # Try the inner path as-is (works if base is a LlamaModel-flavoured
        # class, matching McGill's training-time class), then fall back to
        # `model.<inner_path>` (standard LlamaForCausalLM).
        target = None
        for cand in (clean_path, f"model.{clean_path}"):
            try:
                target = base.get_submodule(cand)
                break
            except AttributeError:
                continue
        if target is None:
            n_no_submodule += 1
            continue

        A = ab["A"].to(torch.float32)  # (r, in)
        B = ab["B"].to(torch.float32)  # (out, r)
        delta = scaling * (B @ A)  # (out, in)

        if delta.shape != target.weight.shape:
            n_shape_mismatch += 1
            continue

        target.weight.data.add_(delta.to(target.weight.dtype))
        delta_norms.append(delta.norm().item())
        n_applied += 1

    print(f"[{name}] merged {n_applied} LoRA-wrapped modules "
          f"(skipped: missing_pair={n_missing_pair}, "
          f"no_submodule={n_no_submodule}, shape_mismatch={n_shape_mismatch})")
    if n_applied == 0:
        raise SystemExit(f"[{name}] FATAL: no LoRA deltas applied. "
                         "Check adapter file layout vs. base model structure.")
    print(f"[{name}]   delta L2 norms: "
          f"min={min(delta_norms):.4e} "
          f"max={max(delta_norms):.4e} "
          f"mean={sum(delta_norms)/len(delta_norms):.4e}")


def main():
    args = parse_args()

    dtype = _dtype(args.dtype)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load base ----------------------------------------------------
    print(f"[bridge] loading base model {args.base_model} (dtype={args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[bridge] base vocab_size = {base.config.vocab_size}, "
          f"tokenizer len = {len(tokenizer)}")

    # ---- 2. Merge MNTP adapter -------------------------------------------
    print(f"[bridge] applying MNTP adapter {args.mntp_adapter}")
    _apply_lora_delta(base, args.mntp_adapter, name="MNTP")

    # ---- 3. Merge SimCSE adapter (optional) ------------------------------
    if args.simcse_adapter is not None:
        print(f"[bridge] applying SimCSE adapter {args.simcse_adapter}")
        _apply_lora_delta(base, args.simcse_adapter, name="SimCSE")

    # ---- 4. Add SAE-LEWIS-specific tokens --------------------------------
    print(f"[bridge] adding special tokens: {args.add_special_tokens}")
    added_total = 0
    for tok in args.add_special_tokens:
        if tok == "[MASK]":
            if tokenizer.mask_token is None:
                added_total += tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        else:
            added_total += tokenizer.add_special_tokens(
                {"additional_special_tokens": [tok]}
            )
    print(f"[bridge]   added {added_total} new tokens; tokenizer len = {len(tokenizer)}")

    # ---- 5. Resize embeddings if we added anything -----------------------
    if added_total > 0 or base.config.vocab_size != len(tokenizer):
        old = base.config.vocab_size
        base.resize_token_embeddings(len(tokenizer))
        print(f"[bridge] resize_token_embeddings: {old} → {base.config.vocab_size}")
        print("[bridge]   new rows initialised via HF's mean-of-existing trick "
              "(same as our previous train_llm2vec.py)")
    ids = {tok: tokenizer.convert_tokens_to_ids(tok)
           for tok in ("[MASK]", "[INS]", "[DEL]")
           if tokenizer.convert_tokens_to_ids(tok) is not None}
    print(f"[bridge]   token ids: {ids}")

    # ---- 6. Save as drop-in --llm2vec-dir --------------------------------
    # Gemma ties lm_head.weight to embed_tokens.weight; safetensors refuses
    # tied tensors under the Trainer default save path, so use the legacy
    # binary format. Matches what our train_llm2vec.py has always done.
    print(f"[bridge] saving to {out_dir}")
    base.save_pretrained(out_dir, safe_serialization=False)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "base_llm": args.base_model,
        "mntp_source": str(args.mntp_adapter),
        "simcse_source": str(args.simcse_adapter) if args.simcse_adapter else None,
        "training_recipe": "McGill-NLP/llm2vec (vendored)",
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "ins_token_id": tokenizer.convert_tokens_to_ids("[INS]"),
        "del_token_id": tokenizer.convert_tokens_to_ids("[DEL]"),
        "lora": {
            "mntp":   {"merged": True, "source": str(args.mntp_adapter)},
            "simcse": ({"merged": True, "source": str(args.simcse_adapter)}
                       if args.simcse_adapter else None),
        },
        # Downstream shard-holdout eval expects this key. McGill trained on
        # Wikipedia so Dolma leakage isn't an issue — dolma_max_files=0.
        "dolma_max_files": 0,
        "simcse": ({
            "source": str(args.simcse_adapter),
            "merged": True,
        } if args.simcse_adapter else None),
    }
    (out_dir / "llm2vec_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bridge] wrote {out_dir}/llm2vec_meta.json")
    print("[bridge] done.")


if __name__ == "__main__":
    main()
