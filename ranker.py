"""
SAE-aware candidate ranker (see README §4.4).

    score(c) =  α · sae_align(c, z_amp, z_sup)
              + β · fluency(c)
              + γ · content_preservation(input, c)
              − η · num_INS_slots(c)

- sae_align: cosine of pool-max SAE features with z_amp minus with z_sup
- fluency:   mean log-likelihood under frozen causal Gemma
- content:   cosine between LLM2Vec sentence embeddings of input and candidate

All sub-scores are computed offline at inference (no gradient).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

from model import BidirectionalLLM, SAEFeatureExtractor


@dataclass
class RankerWeights:
    sae_align: float = 1.0
    fluency: float = 0.3
    content: float = 0.2
    length_penalty: float = 0.05


class Ranker:
    def __init__(
        self,
        extractor: SAEFeatureExtractor,
        causal_llm,
        bid_llm: BidirectionalLLM,
        weights: RankerWeights = RankerWeights(),
        device: str = "cuda",
    ):
        self.extractor = extractor.to(device).eval()
        self.causal_llm = causal_llm.to(device).eval()
        self.bid_llm = bid_llm.to(device).eval()
        self.weights = weights
        self.device = device

    @torch.no_grad()
    def _sae_pool_max(self, token_ids: List[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        z = self.extractor.encode_token_ids(ids)         # (T, d_sae)
        return z.max(dim=0).values                        # (d_sae,)

    @torch.no_grad()
    def _causal_logprob(self, token_ids: List[int]) -> float:
        if len(token_ids) < 2:
            return 0.0
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        out = self.causal_llm(input_ids=ids, labels=ids, use_cache=False)
        return -float(out.loss.item())

    @torch.no_grad()
    def _sentence_embed(self, token_ids: List[int]) -> torch.Tensor:
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(ids)
        h = self.bid_llm(input_ids=ids, attention_mask=attn).last_hidden_state[0]
        return h.mean(dim=0)

    @torch.no_grad()
    def component_scores(
        self,
        cand_ids: List[int],
        input_ids: List[int],
        z_amp: torch.Tensor,
        z_sup: torch.Tensor,
        num_ins_slots: int,
    ) -> dict:
        """Raw sub-scores for one candidate — combined by `combine()`.

        Exposed separately so weight calibration can grid-search
        RankerWeights offline over cached components
        (scripts/calibrate_ranker.py) without re-running the three models.
        """
        z_cand = self._sae_pool_max(cand_ids).to(z_amp.device)
        eps = 1e-8
        sa_amp = torch.dot(z_cand, z_amp) / (z_cand.norm() * z_amp.norm() + eps)
        sa_sup = torch.dot(z_cand, z_sup) / (z_cand.norm() * z_sup.norm() + eps)

        fluency = self._causal_logprob(cand_ids)

        e_in = self._sentence_embed(input_ids).float()
        e_cd = self._sentence_embed(cand_ids).float()
        content = float((e_in @ e_cd) / (e_in.norm() * e_cd.norm() + eps))

        return {
            "sae_align": float(sa_amp - sa_sup),
            # tanh-bounded mean log-likelihood ([0,1]-ish soft scale)
            "fluency": math.tanh(fluency),
            "content": content,
            "ins_slots": int(num_ins_slots),
        }

    def combine(self, comp: dict) -> float:
        w = self.weights
        return (
            w.sae_align * comp["sae_align"]
            + w.fluency * comp["fluency"]
            + w.content * comp["content"]
            - w.length_penalty * comp["ins_slots"]
        )

    @torch.no_grad()
    def score_candidate(
        self,
        cand_ids: List[int],
        input_ids: List[int],
        z_amp: torch.Tensor,
        z_sup: torch.Tensor,
        num_ins_slots: int,
    ) -> float:
        return self.combine(self.component_scores(
            cand_ids, input_ids, z_amp, z_sup, num_ins_slots))

    @torch.no_grad()
    def rank(
        self,
        candidates: List[List[int]],
        input_ids: List[int],
        z_amp: torch.Tensor,
        z_sup: torch.Tensor,
        num_ins_slots_per_cand: List[int],
    ) -> List[float]:
        scores: List[float] = []
        for cand, k in zip(candidates, num_ins_slots_per_cand):
            scores.append(self.score_candidate(cand, input_ids, z_amp, z_sup, k))
        return scores
