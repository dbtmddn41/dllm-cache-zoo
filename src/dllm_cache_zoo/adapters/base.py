"""The interface a cache method uses to run a host model one piece at a time.

A method never touches model classes. It asks the adapter to embed tokens,
project queries, keys and values for some positions, attend, and run the MLP.
The adapter uses the modules and weights of the model the host already loaded,
so no method needs its own copy of the model code.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from ..positions import Positions

# Opaque rotary table for a set of positions, produced by ``ModelAdapter.rotary``.
Rotary = Any


class ModelAdapter(ABC):
    """Uniform view of one model family.

    Shapes: hidden states are ``(B, N, C)``; queries are ``(B, H, N, D)``; keys
    and values are ``(B, H_kv, N, D)``. Keys and queries come back with rotary
    position embeddings already applied for the absolute positions passed in, so
    a stored key stays valid for as long as its token does.
    """

    #: ``"bidirectional"`` (every position attends to every position) or
    #: ``"block-causal"`` (a block attends to itself and earlier blocks).
    family: ClassVar[str]
    #: Name shown in error messages and support tables.
    name: ClassVar[str]

    model: torch.nn.Module
    num_layers: int
    num_heads: int
    num_kv_heads: int
    mask_token_id: int | None
    #: How far the head looks ahead: the hidden state at position ``i`` predicts the token
    #: at ``i + logit_shift``. 0 for LLaDA and SDAR, 1 for Fast-dLLM v2.
    logit_shift: int = 0
    #: Block size the checkpoint was trained with, when its configuration names one.
    block_length: int | None = None

    @classmethod
    @abstractmethod
    def matches(cls, model: torch.nn.Module) -> bool:
        """Whether this adapter can drive ``model`` (decided from its module layout)."""

    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def rotary(self, positions: Positions, length: int, like: torch.Tensor) -> Rotary:
        """Rotary table for ``positions`` of a sequence with ``length`` positions."""

    @abstractmethod
    def attn_norm(self, layer: int, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def project_q(self, layer: int, x_normed: torch.Tensor, rot: Rotary) -> torch.Tensor: ...

    @abstractmethod
    def project_k(self, layer: int, x_normed: torch.Tensor, rot: Rotary) -> torch.Tensor: ...

    @abstractmethod
    def project_v(self, layer: int, x_normed: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def attn_out(self, layer: int, mixed: torch.Tensor) -> torch.Tensor:
        """Output projection on merged-head attention output ``(B, N, C)``."""

    @abstractmethod
    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward branch including its norm. The caller adds the residual."""

    @abstractmethod
    def head(self, x: torch.Tensor) -> torch.Tensor:
        """Final norm and unembedding."""

    # ------------------------------------------------------------------ shared

    def attention_scale(self, layer: int) -> float | None:
        """Scale on the attention scores. ``None`` is the usual ``1 / sqrt(head_dim)``."""
        return None

    def attend(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attention followed by the output projection.

        ``mask`` is boolean, broadcastable to ``(B, H, N_q, N_k)``, True where a
        query may read a key. With ``need_weights`` the attention runs in eager
        form and the head-wise weights ``(B, H, N_q, N_k)`` are returned as well.
        """
        if k.size(1) != q.size(1):
            rep = q.size(1) // k.size(1)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scale = self.attention_scale(layer)
        weights = None
        if need_weights:
            eager_scale = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
            scores = torch.matmul(q, k.transpose(-2, -1)) * eager_scale
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            mixed = torch.matmul(weights, v)
        else:
            mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        b, _, n, _ = mixed.shape
        mixed = mixed.transpose(1, 2).contiguous().view(b, n, -1)
        return self.attn_out(layer, mixed), weights
