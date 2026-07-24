"""
Minimal manual LoRA (Hu et al. 2021) for the SAE-LEWIS backbones.

LEWIS fine-tunes its tagger (RoBERTa) and generator (BART) fully; the
frozen-backbone v1/v2 setup was a deviation, and the held-out probes
suggested the frozen encoder lacked the capacity to route the conditioning
prefix to the edit positions (only the linear Proj_A was trainable).
LoRA on the attention + MLP projections is the LEWIS-faithful middle
ground that also stays consistent with how the LLM2Vec checkpoint itself
was produced (MNTP and SimCSE are both LoRA stages): the downstream task
is a third LoRA adaptation on the merged checkpoint.

Deliberately NOT peft: the bidirectional Gemma backbone is monkey-patched
(`model._patch_attention_bidirectional`) and this repo has already been
bitten once by peft silently no-op'ing on non-standard model classes
(see scripts/mcgill_merge_and_expand.py's docstring). ~80 lines of
explicit code with stable parameter names beats a wrapper we would have
to fight.

Embedding table and LM head are intentionally NOT targeted: the editor's
outputs must stay grounded in the frozen (tied) Gemma vocabulary geometry,
and the [MASK]/[INS]/[SEP] rows are handled by the editor's delta_emb.

Parameter naming: replacing `....q_proj` with `LoRALinear(base)` yields
parameter names `...q_proj.base.weight` (frozen) and `...q_proj.lora_A` /
`...q_proj.lora_B` (trainable, fp32). Trainable checkpointing filters on
the `lora_` substring.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Gemma-2 attention + MLP projections. Embeddings / lm_head excluded by design.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank residual.

    The adapter path runs in float32 regardless of the base dtype (bf16
    adapters train poorly under AdamW); the result is cast back to the
    base output dtype.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout_p = float(dropout)
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, dtype=torch.float32))
        # Standard LoRA init: A ~ Kaiming, B = 0 → identity at step 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def add_adapter2(self, r2: int, alpha2: float = 16.0):
        """v3d-enc (2026-07-24): a SECOND, independent low-rank residual.
        The original adapter (the zero-shot model's) stays frozen and
        untouched; adapter2 starts at exact zero (B2 = 0) and carries a
        runtime scale knob (adapter2_scale; 0.0 recovers zero-shot)."""
        self.r2 = int(r2)
        self.scaling2 = float(alpha2) / float(r2)
        self.adapter2_scale = 1.0
        self.lora_A2 = nn.Parameter(torch.empty(
            r2, self.base.in_features, dtype=torch.float32))
        self.lora_B2 = nn.Parameter(torch.zeros(
            self.base.out_features, r2, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        h = x.to(self.lora_A.dtype)
        if self.dropout_p > 0.0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        delta = (h @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        if getattr(self, "lora_A2", None) is not None:
            d2 = ((h @ self.lora_A2.t()) @ self.lora_B2.t()
                  * self.scaling2 * float(self.adapter2_scale))
            delta = delta + d2
        return y + delta.to(y.dtype)


def apply_lora(
    root: nn.Module,
    r: int,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
) -> int:
    """Replace every targeted nn.Linear under `root` with a LoRALinear.

    Returns the number of wrapped modules; raises if nothing matched
    (a silent no-op here would reproduce exactly the peft failure mode
    this module exists to avoid).
    """
    targets = set(target_modules)
    to_wrap: List[tuple] = []
    for name, module in root.named_modules():
        for child_name, child in module.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                to_wrap.append((module, child_name, child))
    for parent, child_name, child in to_wrap:
        setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha,
                                               dropout=dropout))
    if not to_wrap:
        raise RuntimeError(
            f"apply_lora: no nn.Linear named {sorted(targets)} found under "
            f"{type(root).__name__} — wrong backbone class?")
    return len(to_wrap)


def add_adapter2_everywhere(root: nn.Module, r2: int,
                            alpha2: float = 16.0) -> int:
    n = 0
    for m in root.modules():
        if isinstance(m, LoRALinear):
            m.add_adapter2(r2, alpha2)
            n += 1
    if n == 0:
        raise RuntimeError("add_adapter2_everywhere: no LoRALinear found")
    return n


def set_adapter2_scale(root: nn.Module, s: float) -> None:
    for m in root.modules():
        if isinstance(m, LoRALinear) and getattr(m, "lora_A2", None) is not None:
            m.adapter2_scale = float(s)


def lora_parameters(root: nn.Module):
    """Iterate (name, param) over all LoRA adapter parameters under root."""
    for n, p in root.named_parameters():
        if "lora_" in n:
            yield n, p


def lora_state_dict(root: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in lora_parameters(root)}


def load_lora_state_dict(root: nn.Module, sd: Dict[str, torch.Tensor]):
    own = dict(lora_parameters(root))
    missing = set(sd) - set(own)
    absent = set(own) - set(sd)
    if missing or absent:
        raise ValueError(
            f"LoRA state mismatch: checkpoint-only keys {sorted(missing)[:3]}"
            f"…, model-only keys {sorted(absent)[:3]}… — was the model "
            f"constructed with the same lora_r/target set?")
    for n, p in own.items():
        p.data.copy_(sd[n].to(p.dtype))
