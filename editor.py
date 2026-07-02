"""
SAE-LEWIS bidirectional editor.

Forward (see README §4.3):

    [INT_amp, INT_sup, e(editor_input_1), …, e(editor_input_{T'})]
            │                                       │
            └── LLM2Vec'd Gemma (frozen) ──────────►│
                                                    │
                                       Gemma LM head (frozen)
                                                    │
                                            token logits per position

Trainable: Proj_A (d_sae → d_model), type_emb[0..2], embedding rows for
[INS] and [DEL]. Everything else is frozen.

At inference, template enumeration sets [INS] slot counts per gap; the
ranker then scores each template's argmax output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import BidirectionalLLM


class SAEEditor(nn.Module):
    def __init__(
        self,
        llm2vec_dir: str,
        d_sae: int,
        dtype: torch.dtype = torch.bfloat16,
        train_token_ids: Optional[Dict[str, int]] = None,
    ):
        """
        Parameters
        ----------
        llm2vec_dir : str
            Path to the MNTP'd Gemma checkpoint (output of train_llm2vec.py).
        d_sae : int
            SAE feature dimension.
        train_token_ids : Optional[Dict[str, int]]
            Optional dict of {token_name: id} for tokens whose embedding rows
            should be trained. By default we unfreeze [INS] and [DEL] rows.
        """
        super().__init__()
        self.encoder = BidirectionalLLM(llm2vec_dir, dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # LM head is the causal Gemma's output projection. We load the
        # causal model only to grab its LM head module — frozen.
        causal = AutoModelForCausalLM.from_pretrained(llm2vec_dir, torch_dtype=dtype)
        self.lm_head = causal.get_output_embeddings()
        for p in self.lm_head.parameters():
            p.requires_grad_(False)
        # Drop the rest of the causal model to free memory
        del causal

        d_model = self.encoder.config.hidden_size
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)

        # Trainable Proj_A: d_sae → d_model. Float32 for stability.
        self.proj_a = nn.Linear(d_sae, d_model, bias=True)
        nn.init.normal_(self.proj_a.weight, std=0.02)
        nn.init.zeros_(self.proj_a.bias)

        # type_emb[0..2]: text / amp / sup
        self.type_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        # Conditioning scale calibration. Gemma multiplies inputs_embeds by
        # sqrt(hidden_size) internally, and z values are raw SAE activation
        # deltas whose magnitude varies per feature by orders of magnitude.
        # Each cond vector is RMS-normalized to the median token-embedding
        # row RMS so the prefix neither vanishes against type_emb nor pushes
        # the frozen encoder off distribution. cond_scale is a learnable
        # global gain on top (init 1.0).
        with torch.no_grad():
            emb_w = self.encoder.get_input_embeddings().weight
            row_rms = emb_w.float().pow(2).mean(dim=-1).sqrt()
            target_rms = row_rms.median()
        self.register_buffer("cond_target_rms", target_rms.to(torch.float32))
        self.cond_scale = nn.Parameter(torch.ones(1))

        # Trainable rows for [INS] and [DEL] embeddings within the encoder's
        # input embedding. We carry a parameter slice that is added at the
        # right token-id rows during forward — keeps the encoder frozen
        # while letting the new tokens learn.
        self.train_token_ids = train_token_ids or {}
        if self.train_token_ids:
            self.delta_emb = nn.Parameter(
                torch.zeros(len(self.train_token_ids), d_model, dtype=torch.float32)
            )
            # Map token_id → slot in delta_emb
            self._delta_slots = {
                int(tid): slot for slot, tid in enumerate(self.train_token_ids.values())
            }
        else:
            self.delta_emb = nn.Parameter(torch.zeros(0, d_model))
            self._delta_slots = {}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def encoder_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.encoder.get_input_embeddings()(input_ids)
        if self._delta_slots:
            # Add the trainable delta at positions whose id ∈ delta_slots
            slots = torch.full_like(input_ids, -1, dtype=torch.long)
            for tid, slot in self._delta_slots.items():
                slots = torch.where(input_ids == tid, torch.tensor(slot, device=input_ids.device), slots)
            mask = (slots >= 0)
            if mask.any():
                slot_idx = slots.clamp(min=0)
                # Add per-position
                d_add = self.delta_emb.to(emb.dtype)[slot_idx]
                emb = emb + d_add * mask.unsqueeze(-1).to(emb.dtype)
        return emb

    def _calibrate_cond(self, x: torch.Tensor) -> torch.Tensor:
        """RMS-normalize a (B, d_model) cond vector to the calibrated target."""
        # eps INSIDE the sqrt: empty-conditioning samples (z all-zero) give
        # x == 0, and sqrt'(0) is infinite. With eps added after the sqrt the
        # gradient still blows up the instant Proj_A is unfrozen, NaN-ing the
        # whole run. Folding eps under the sqrt keeps the gradient finite.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x / rms * (self.cond_target_rms * self.cond_scale)

    def cond_embeds(self, z_amp: torch.Tensor, z_sup: torch.Tensor) -> torch.Tensor:
        """Build the (B, 2, d_model) prefix conditioning tensor."""
        # Proj_A in float32; cast back to encoder dtype at the boundary.
        amp = self._calibrate_cond(self.proj_a(z_amp.to(self.proj_a.weight.dtype)))  # (B, d_model)
        sup = self._calibrate_cond(self.proj_a(z_sup.to(self.proj_a.weight.dtype)))
        # Add type embeddings
        type_amp = self.type_emb(torch.full((amp.shape[0],), 1, device=amp.device, dtype=torch.long))
        type_sup = self.type_emb(torch.full((sup.shape[0],), 2, device=sup.device, dtype=torch.long))
        amp = amp + type_amp
        sup = sup + type_sup
        return torch.stack([amp, sup], dim=1)                  # (B, 2, d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T) editor input (with [MASK] / [INS] / γ / etc.)
        attention_mask: torch.Tensor,      # (B, T)
        z_amp: torch.Tensor,               # (B, d_sae)
        z_sup: torch.Tensor,               # (B, d_sae)
        labels: Optional[torch.Tensor] = None,   # (B, T) -100 = ignore
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        tok_embs = self.encoder_embed(input_ids)               # (B, T, d_model)
        cond = self.cond_embeds(z_amp, z_sup).to(tok_embs.dtype)  # (B, 2, d_model)
        full_embs = torch.cat([cond, tok_embs], dim=1)         # (B, T+2, d_model)

        full_mask = torch.cat([
            torch.ones(B, 2, dtype=attention_mask.dtype, device=device),
            attention_mask,
        ], dim=1)

        h = self.encoder(
            inputs_embeds=full_embs, attention_mask=full_mask,
        ).last_hidden_state                                     # (B, T+2, d_model)

        # Drop the 2 prefix positions before projecting to logits
        h_text = h[:, 2:, :]                                    # (B, T, d_model)
        logits = self.lm_head(h_text.to(self.lm_head.weight.dtype))

        # Tied output-side correction for the trainable special tokens.
        # The McGill-merged checkpoint's [MASK]/[INS]/[DEL] rows are
        # mean-init and frozen inside the LM head, so [DEL] could never win
        # an argmax through the frozen column alone. Reuse the input-side
        # delta rows as logit-column deltas (Gemma ties embed_tokens and
        # lm_head, so input row == output column for every real token too).
        if self._delta_slots:
            tids = list(self._delta_slots.keys())
            slots = list(self._delta_slots.values())
            delta_cols = self.delta_emb[slots]                          # (k, d_model)
            extra = h_text.to(delta_cols.dtype) @ delta_cols.t()        # (B, T, k)
            logits[..., tids] = logits[..., tids] + extra.to(logits.dtype)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).long(),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "hidden_states": h_text}

    # ------------------------------------------------------------------
    # Save / load only trainable parts
    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            "proj_a.weight": self.proj_a.weight.detach().cpu(),
            "proj_a.bias": self.proj_a.bias.detach().cpu(),
            "type_emb.weight": self.type_emb.weight.detach().cpu(),
            "cond_scale": self.cond_scale.detach().cpu(),
        }
        if self.delta_emb.numel() > 0:
            sd["delta_emb"] = self.delta_emb.detach().cpu()
        return sd

    def load_trainable(self, state_dict: Dict[str, torch.Tensor]):
        self.proj_a.weight.data.copy_(state_dict["proj_a.weight"])
        self.proj_a.bias.data.copy_(state_dict["proj_a.bias"])
        self.type_emb.weight.data.copy_(state_dict["type_emb.weight"])
        if "cond_scale" in state_dict:  # absent in pre-calibration checkpoints
            self.cond_scale.data.copy_(state_dict["cond_scale"])
        if "delta_emb" in state_dict and self.delta_emb.numel() > 0:
            self.delta_emb.data.copy_(state_dict["delta_emb"])

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trainable": self.trainable_state_dict(),
            "d_sae": int(self.d_sae),
            "d_model": int(self.d_model),
            "train_token_ids": self.train_token_ids,
        }, path)


def load_editor_from_checkpoint(
    llm2vec_dir: str, ckpt_path: str, d_sae: int,
    dtype: torch.dtype = torch.bfloat16,
) -> SAEEditor:
    blob = torch.load(ckpt_path, map_location="cpu")
    editor = SAEEditor(
        llm2vec_dir, d_sae=d_sae, dtype=dtype,
        train_token_ids=blob.get("train_token_ids", {}),
    )
    editor.load_trainable(blob["trainable"])
    return editor
