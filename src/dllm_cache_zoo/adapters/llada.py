"""Adapter for LLaDA checkpoints (LLaDA-8B-Base, LLaDA-8B-Instruct, LLaDA-1.5).

The released model code keeps its layers in ``model.model.transformer.blocks``
and has no cache path of its own: its forward asserts that no key/value cache
is passed. This adapter reproduces that forward from the block's own modules,
with rotary embeddings taken at absolute positions, so a subset of positions
can run against stored keys and values.

Reference: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/main/modeling_llada.py
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F

from ..positions import Positions
from .base import ModelAdapter, Rotary

LLADA_MASK_TOKEN_ID = 126336


class LLaDAAdapter(ModelAdapter):
    family = "bidirectional"
    name = "llada"

    @classmethod
    def matches(cls, model: torch.nn.Module) -> bool:
        transformer = getattr(getattr(model, "model", None), "transformer", None)
        if transformer is None:
            return False
        blocks = getattr(transformer, "blocks", None)
        if blocks is None or len(blocks) == 0:
            return False
        block = blocks[0]
        needed = (
            "attn_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_out",
            "ff_norm",
            "ff_proj",
            "up_proj",
            "ff_out",
            "rotary_emb",
        )
        return all(hasattr(block, name) for name in needed)

    def __init__(self, model: torch.nn.Module):
        self.model = model
        m = cast(Any, model)
        self.config = m.config
        self.transformer = m.model.transformer
        self.blocks = self.transformer.blocks
        self.num_layers = len(self.blocks)
        self.num_heads = int(self.config.n_heads)
        n_kv = getattr(self.config, "effective_n_kv_heads", None)
        if n_kv is None:
            n_kv = getattr(self.config, "n_kv_heads", None) or self.num_heads
        self.num_kv_heads = int(n_kv)
        self.d_model = int(self.config.d_model)
        self.head_dim = self.d_model // self.num_heads
        self.mask_token_id = getattr(self.config, "mask_token_id", None) or LLADA_MASK_TOKEN_ID
        if not getattr(self.config, "rope", True):
            raise ValueError(
                "The LLaDA adapter needs a checkpoint with rotary position embeddings."
            )
        self._full_precision = bool(getattr(self.config, "rope_full_precision", True))

    # ------------------------------------------------------------- embedding

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.transformer.wte(input_ids)
        if getattr(self.config, "input_emb_norm", False):
            x = x * (self.d_model**0.5)
        return self.transformer.emb_drop(x)

    # ---------------------------------------------------------------- rotary

    def rotary(self, positions: Positions, length: int, like: torch.Tensor) -> Rotary:
        # The released block caches sin/cos tables of shape (1, 1, T, D), shared by all blocks.
        sin, cos = self.blocks[0].rotary_emb.get_rotary_embedding(length, like.device)
        if positions is None:
            return sin[:, :, :length], cos[:, :, :length]
        if isinstance(positions, slice):
            return sin[:, :, positions], cos[:, :, positions]
        # (B, N) -> (B, 1, N, D)
        return sin[0, 0][positions].unsqueeze(1), cos[0, 0][positions].unsqueeze(1)

    def _rotate(self, t: torch.Tensor, rot: Rotary) -> torch.Tensor:
        sin, cos = rot
        t_ = t.float() if self._full_precision else t
        with torch.autocast(t.device.type, enabled=False):
            sin, cos = sin.type_as(t_), cos.type_as(t_)
            b, h, n, d = t_.shape
            halves = t_.view(b, h, n, 2, d // 2)
            x1, x2 = halves.unbind(dim=-2)
            rotated = torch.cat((-x2, x1), dim=-1)
            out = ((t_ * cos) + (rotated * sin)).to(t_.dtype)
        return out.type_as(t)

    # ------------------------------------------------------------- attention

    def attn_norm(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[layer].attn_norm(x)

    def _split(self, t: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, heads, self.head_dim).transpose(1, 2)

    def project_q(self, layer: int, x_normed: torch.Tensor, rot: Rotary) -> torch.Tensor:
        block = self.blocks[layer]
        q = block.q_proj(x_normed)
        if getattr(block, "q_norm", None) is not None:
            q = block.q_norm(q).to(dtype=x_normed.dtype)
        return self._rotate(self._split(q, self.num_heads), rot)

    def project_k(self, layer: int, x_normed: torch.Tensor, rot: Rotary) -> torch.Tensor:
        block = self.blocks[layer]
        k = block.k_proj(x_normed)
        if getattr(block, "k_norm", None) is not None:
            k = block.k_norm(k).to(dtype=x_normed.dtype)
        return self._rotate(self._split(k, self.num_kv_heads), rot)

    def project_v(self, layer: int, x_normed: torch.Tensor) -> torch.Tensor:
        return self._split(self.blocks[layer].v_proj(x_normed), self.num_kv_heads)

    def attn_out(self, layer: int, mixed: torch.Tensor) -> torch.Tensor:
        return self.blocks[layer].attn_out(mixed)

    # ------------------------------------------------------------------- mlp

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        block = self.blocks[layer]
        h = block.ff_norm(x)
        h, up = block.ff_proj(h), block.up_proj(h)
        return block.ff_out(block.act(h) * up)

    # ------------------------------------------------------------------ head

    def head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer.ln_f(x)
        if getattr(self.config, "weight_tying", False):
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if getattr(self.config, "scale_logits", False):
            logits = logits * (1 / math.sqrt(self.d_model))
        return logits
