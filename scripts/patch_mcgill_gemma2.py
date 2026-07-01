"""
Patch McGill's vendored llm2vec to add Gemma-2 bidirectional support.

McGill's `llm2vec/llm2vec.py::_get_model_class` recognises only
MistralConfig, LlamaConfig, GemmaConfig (v1), and Qwen2Config — Gemma-2
is unsupported upstream (open PR #167 tries but breaks softcapping/
sliding-window). This script closes that gap locally without forking
their repo.

Three edits, all idempotent:

  1. Write  vendored/mcgill_llm2vec/llm2vec/models/bidirectional_gemma2.py
     — Gemma2BiModel + Gemma2BiForMNTP + Modified* attention subclasses.
     Overwritten every run so iterating on the file lives in this script.

  2. Append to vendored/mcgill_llm2vec/llm2vec/models/__init__.py
     — export Gemma2BiModel / Gemma2BiForMNTP alongside the other
     bidirectional variants. Skipped if already present.

  3. Patch vendored/mcgill_llm2vec/llm2vec/llm2vec.py
     — add Gemma2Config to the top-level `transformers` import,
       Gemma2BiModel to the `.models` import, and a Gemma2Config branch
       in `LLM2Vec._get_model_class`. Skipped if already present.

Gemma-2 differences from Gemma-1 (handled inside bidirectional_gemma2.py):

  * Sliding-window attention on every-other layer — disabled entirely
    for bidirectional (`is_sliding=False`, `sliding_window=None`).
  * Attention softcapping (attn_logit_softcapping) and final logit
    softcapping (final_logit_softcapping) — kept: inherited from the
    upstream Gemma-2 classes.
  * Four RMSNorms per decoder layer (input / post-attn / pre-ff /
    post-ff) instead of Gemma-1's two — mirrored in
    ModifiedGemma2DecoderLayer.
  * HybridCache instead of StaticCache — the mask helper checks for it.

Usage:
    python scripts/patch_mcgill_gemma2.py             # apply
    python scripts/patch_mcgill_gemma2.py --check     # verify only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "vendored" / "mcgill_llm2vec"


BIDIR_GEMMA2_PY = '''"""
Bidirectional Gemma-2 for LLM2Vec — mirrors bidirectional_gemma.py
but handles Gemma-2's sliding-window layers, softcapping, and 4-norm
decoder block. Written by scripts/patch_mcgill_gemma2.py in the
SAE-LEWIS repo — do not edit here.
"""
import torch
from packaging import version
import importlib.metadata

from transformers import (
    Gemma2Model,
    Gemma2ForCausalLM,
    Gemma2PreTrainedModel,
    Gemma2Config,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2Attention,
    Gemma2FlashAttention2,
    Gemma2SdpaAttention,
    Gemma2MLP,
    Gemma2RMSNorm,
)

from torch import nn
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import _is_package_available
from transformers.cache_utils import Cache, HybridCache

from peft import PeftModel

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_43():
    if not _is_package_available("transformers"):
        return False
    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.43.0"
    )


class ModifiedGemma2Attention(Gemma2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LLM2Vec bidirectional: every token attends to every other
        # token, so causal masking AND per-layer sliding-window are off.
        self.is_causal = False
        self.is_sliding = False
        self.sliding_window = None


class ModifiedGemma2FlashAttention2(Gemma2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False
        self.is_sliding = False
        self.sliding_window = None


class ModifiedGemma2SdpaAttention(Gemma2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False
        self.is_sliding = False
        self.sliding_window = None


GEMMA2_ATTENTION_CLASSES = {
    "eager": ModifiedGemma2Attention,
    "flash_attention_2": ModifiedGemma2FlashAttention2,
    "sdpa": ModifiedGemma2SdpaAttention,
}


class ModifiedGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        # Skip Gemma2DecoderLayer.__init__ (it would build the causal
        # attention + read layer_idx to decide is_sliding); rebuild the
        # module tree from scratch with our Modified* attention and no
        # sliding-window pattern.
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA2_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Bidirectional: no layer-level sliding-window handling
        # regardless of layer_idx. Gemma2DecoderLayer.forward checks
        # self.is_sliding to slice attention_mask down to a window; we
        # want full attention on every layer.
        self.is_sliding = False
        self.sliding_window = None


class Gemma2BiModel(Gemma2Model):
    _no_split_modules = ["ModifiedGemma2DecoderLayer"]

    def __init__(self, config: Gemma2Config):
        if not is_transformers_attn_greater_or_equal_4_43():
            raise ValueError(
                "Gemma2BiModel requires transformers >= 4.43.0 "
                "(Gemma-2 landed in 4.43)."
            )
        Gemma2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedGemma2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache = None,
        output_attentions: bool = False,
    ):
        # Fully-bidirectional attention mask — same skeleton as
        # bidirectional_gemma.py, but sliding-window logic is a no-op
        # because our Modified* attention classes have is_sliding=False
        # and sliding_window=None on every layer.
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_hybrid_cache = isinstance(past_key_values, HybridCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        if using_hybrid_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )
            # Bidirectional: do NOT apply the upper-triangular causal
            # mask that the base class applies here.
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class Gemma2BiForMNTP(Gemma2ForCausalLM):
    def __init__(self, config):
        Gemma2PreTrainedModel.__init__(self, config)
        self.model = Gemma2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # post_init handles tie_weights for tie_word_embeddings=True
        # (Gemma-2's default).
        self.post_init()

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.model.save_pretrained(path)
'''


def _patch_init_py(check_only: bool) -> bool:
    """Add Gemma2 export to models/__init__.py."""
    init_py = VENDOR_DIR / "llm2vec" / "models" / "__init__.py"
    text = init_py.read_text()
    if "bidirectional_gemma2" in text:
        print("[patch] ✓ models/__init__.py already exports Gemma2Bi*")
        return True
    if check_only:
        print("[patch] ✗ models/__init__.py missing Gemma2Bi* export")
        return False
    addition = "from .bidirectional_gemma2 import Gemma2BiModel, Gemma2BiForMNTP\n"
    init_py.write_text(text.rstrip() + "\n" + addition)
    print("[patch] patched models/__init__.py (added Gemma2Bi* export)")
    return True


def _patch_llm2vec_py(check_only: bool) -> bool:
    """Add Gemma2Config to imports and _get_model_class branch."""
    llm2vec_py = VENDOR_DIR / "llm2vec" / "llm2vec.py"
    text = llm2vec_py.read_text()

    if "Gemma2Config" in text and "Gemma2BiModel" in text:
        print("[patch] ✓ llm2vec.py already has Gemma2Config branch + import")
        return True
    if check_only:
        print("[patch] ✗ llm2vec.py missing Gemma2 wiring")
        return False

    # 1. Add Gemma2Config to the `from transformers import (...)` block.
    if "Gemma2Config" not in text:
        text, n = re.subn(
            r"(\n\s+)GemmaConfig,",
            r"\1GemmaConfig,\1Gemma2Config,",
            text,
            count=1,
        )
        if n == 0:
            print("[patch] FATAL: couldn't find `GemmaConfig,` in transformers import")
            sys.exit(3)

    # 2. Add Gemma2BiModel to `from .models import (...)` block.
    if "Gemma2BiModel" not in text:
        text, n = re.subn(
            r"(\n\s+)GemmaBiModel,",
            r"\1GemmaBiModel,\1Gemma2BiModel,",
            text,
            count=1,
        )
        if n == 0:
            print("[patch] FATAL: couldn't find `GemmaBiModel,` in .models import")
            sys.exit(3)

    # 3. Add Gemma2Config branch to _get_model_class.
    old_branch = (
        '        elif config_class_name == "GemmaConfig":\n'
        '            return GemmaBiModel\n'
    )
    new_branch = (
        '        elif config_class_name == "GemmaConfig":\n'
        '            return GemmaBiModel\n'
        '        elif config_class_name == "Gemma2Config":\n'
        '            return Gemma2BiModel\n'
    )
    if old_branch not in text:
        print("[patch] FATAL: couldn't find GemmaConfig branch in _get_model_class")
        sys.exit(3)
    text = text.replace(old_branch, new_branch, 1)

    llm2vec_py.write_text(text)
    print("[patch] patched llm2vec.py (imports + _get_model_class branch)")
    return True


def _write_bidir_gemma2(check_only: bool) -> bool:
    dst = VENDOR_DIR / "llm2vec" / "models" / "bidirectional_gemma2.py"
    if check_only:
        exists = dst.exists() and "Gemma2BiForMNTP" in dst.read_text()
        marker = "✓" if exists else "✗"
        print(f"[patch] {marker} bidirectional_gemma2.py "
              f"{'present' if exists else 'missing'}")
        return exists
    dst.write_text(BIDIR_GEMMA2_PY)
    print("[patch] wrote models/bidirectional_gemma2.py")
    return True


def _patch_experiments_script(script_path: Path, check_only: bool,
                              needs_get_model_class: bool) -> bool:
    """Patch experiments/run_{mntp,simcse}.py to know about Gemma-2.

    Both scripts carry their own `initialize_peft` (a `config.__class__
    .__name__ in [...]` allow-list). run_mntp.py additionally defines a
    top-level `get_model_class` that returns the *BiForMNTP class per
    config. We extend both, and inject a top-level import for
    Gemma2BiForMNTP so the returned class is in scope.
    """
    if not script_path.exists():
        print(f"[patch] SKIP {script_path.name}: not found")
        return True

    text = script_path.read_text()
    already = ("Gemma2Config" in text) and ("Gemma2BiForMNTP" in text or
                                            not needs_get_model_class)
    if already:
        print(f"[patch] ✓ experiments/{script_path.name} already has Gemma-2")
        return True
    if check_only:
        print(f"[patch] ✗ experiments/{script_path.name} missing Gemma-2")
        return False

    # 1. Inject a Gemma2BiForMNTP import once, right below the last
    #    top-level `import` / `from ... import` line — dodges the
    #    multi-line `from transformers import (` block and lands
    #    somewhere unambiguous.
    if needs_get_model_class and "Gemma2BiForMNTP" not in text:
        import_stub = "from llm2vec.models import Gemma2BiForMNTP\n"
        # Find the last non-continuation `from ... import ...` line in
        # the top ~120 lines that isn't inside a paren block.
        lines = text.splitlines(keepends=True)
        insert_idx = None
        depth = 0
        for i, line in enumerate(lines[:150]):
            stripped = line.strip()
            depth += line.count("(") - line.count(")")
            if depth == 0 and (stripped.startswith("from ") or
                               stripped.startswith("import ")):
                insert_idx = i + 1
        if insert_idx is None:
            print(f"[patch] FATAL: no import-block anchor in {script_path.name}")
            sys.exit(3)
        lines.insert(insert_idx, import_stub)
        text = "".join(lines)

    # 2. Extend get_model_class if this script has one.
    if needs_get_model_class:
        old_branch = (
            '    elif config_class_name == "GemmaConfig":\n'
            '        return GemmaBiForMNTP\n'
        )
        new_branch = (
            '    elif config_class_name == "GemmaConfig":\n'
            '        return GemmaBiForMNTP\n'
            '    elif config_class_name == "Gemma2Config":\n'
            '        return Gemma2BiForMNTP\n'
        )
        if old_branch not in text:
            print(f"[patch] FATAL: no GemmaBiForMNTP branch in "
                  f"{script_path.name}'s get_model_class")
            sys.exit(3)
        text = text.replace(old_branch, new_branch, 1)

    # 3. Extend the `if config.__class__.__name__ in [...]` allow-list
    #    that gates LoRA target-module defaults inside initialize_peft.
    old_list = (
        '        "LlamaConfig",\n'
        '        "MistralConfig",\n'
        '        "GemmaConfig",\n'
        '        "Qwen2Config",\n'
    )
    new_list = (
        '        "LlamaConfig",\n'
        '        "MistralConfig",\n'
        '        "GemmaConfig",\n'
        '        "Gemma2Config",\n'
        '        "Qwen2Config",\n'
    )
    if old_list in text:
        text = text.replace(old_list, new_list, 1)
    else:
        # Not fatal — maybe the list is formatted differently on newer
        # upstream commits. Leave a warning; get_model_class alone often
        # unblocks training, and any lora-target fallback issue will
        # surface with a clear error.
        print(f"[patch] WARN: initialize_peft config allow-list not found "
              f"in {script_path.name} — LoRA defaults may fall through")

    script_path.write_text(text)
    print(f"[patch] patched experiments/{script_path.name}")
    return True


def _patch_run_mntp(check_only: bool) -> bool:
    return _patch_experiments_script(
        VENDOR_DIR / "experiments" / "run_mntp.py",
        check_only,
        needs_get_model_class=True,
    )


def _patch_run_simcse(check_only: bool) -> bool:
    return _patch_experiments_script(
        VENDOR_DIR / "experiments" / "run_simcse.py",
        check_only,
        needs_get_model_class=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify patches are applied; don't modify anything.")
    args = ap.parse_args()

    if not VENDOR_DIR.exists():
        print(f"[patch] FATAL: vendored dir not found at {VENDOR_DIR}",
              file=sys.stderr)
        print("        run: bash scripts/vendor_mcgill_llm2vec.sh",
              file=sys.stderr)
        sys.exit(2)

    ok = True
    ok &= _write_bidir_gemma2(args.check)
    ok &= _patch_init_py(args.check)
    ok &= _patch_llm2vec_py(args.check)
    ok &= _patch_run_mntp(args.check)
    ok &= _patch_run_simcse(args.check)

    if args.check and not ok:
        print("[patch] some patches missing — re-run without --check to apply")
        sys.exit(1)
    print("[patch] done")


if __name__ == "__main__":
    main()
