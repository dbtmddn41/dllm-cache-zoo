"""Adapters for checkpoints laid out like Hugging Face Llama/Qwen2 decoders.

These models keep their layers in ``model.model.layers``, each with
``self_attn`` (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``), ``mlp`` and two
layer norms, followed by ``model.model.norm`` and ``model.lm_head``. SDAR uses
this layout, with an extra norm on the per-head queries and keys, and so does
Fast-dLLM v2.

References:
https://huggingface.co/JetLM/SDAR-1.7B-Chat/blob/main/modeling_sdar.py
https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B/blob/main/modeling.py
"""

from __future__ import annotations

from typing import Any, cast

import torch

from ..positions import Positions, as_index
from .base import ModelAdapter, Rotary


class HFDecoderAdapter(ModelAdapter):
    """Shared implementation for the Hugging Face decoder layout."""

    family = "bidirectional"
    name = "hf-decoder"
    model_types: tuple[str, ...] = ()

    @classmethod
    def matches(cls, model: torch.nn.Module) -> bool:
        base = getattr(model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None or len(layers) == 0 or not hasattr(model, "lm_head"):
            return False
        layer = layers[0]
        attn = getattr(layer, "self_attn", None)
        needed = ("q_proj", "k_proj", "v_proj", "o_proj")
        if attn is None or not all(hasattr(attn, name) for name in needed):
            return False
        if not all(
            hasattr(layer, n) for n in ("input_layernorm", "post_attention_layernorm", "mlp")
        ):
            return False
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
        return model_type in cls.model_types

    def __init__(self, model: torch.nn.Module):
        self.model = model
        m = cast(Any, model)
        self.config = m.config
        self.base = m.model
        self.layers = self.base.layers
        self.lm_head = m.lm_head
        self.num_layers = len(self.layers)
        attn = self.layers[0].self_attn
        self.num_heads = int(self.config.num_attention_heads)
        self.num_kv_heads = int(getattr(self.config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(attn, "head_dim", self.config.hidden_size // self.num_heads))
        self.mask_token_id = getattr(self.config, "mask_token_id", None)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.base.embed_tokens(input_ids)

    def rotary(self, positions: Positions, length: int, like: torch.Tensor) -> Rotary:
        position_ids = as_index(positions, length, like.device)
        cos, sin = self.base.rotary_emb(like, position_ids)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    @staticmethod
    def _rotate(t: torch.Tensor, rot: Rotary) -> torch.Tensor:
        cos, sin = rot
        half = t.shape[-1] // 2
        rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
        return (t * cos) + (rotated * sin)

    def attn_norm(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.layers[layer].input_layernorm(x)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, -1, self.head_dim)

    def project_q(self, layer: int, x_normed: torch.Tensor, rot: Rotary) -> torch.Tensor:
        attn = self.layers[layer].self_attn
        q = self._heads(attn.q_proj(x_normed))
        if getattr(attn, "q_norm", None) is not None:
            q = attn.q_norm(q)
        return self._rotate(q.transpose(1, 2), rot)

    def project_k(self, layer: int, x_normed: torch.Tensor, rot: Rotary) -> torch.Tensor:
        attn = self.layers[layer].self_attn
        k = self._heads(attn.k_proj(x_normed))
        if getattr(attn, "k_norm", None) is not None:
            k = attn.k_norm(k)
        return self._rotate(k.transpose(1, 2), rot)

    def project_v(self, layer: int, x_normed: torch.Tensor) -> torch.Tensor:
        return self._heads(self.layers[layer].self_attn.v_proj(x_normed)).transpose(1, 2)

    def attn_out(self, layer: int, mixed: torch.Tensor) -> torch.Tensor:
        return self.layers[layer].self_attn.o_proj(mixed)

    def attention_scale(self, layer: int) -> float | None:
        return getattr(self.layers[layer].self_attn, "scaling", None)

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        block = self.layers[layer]
        return block.mlp(block.post_attention_layernorm(x))

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.base.norm(x))


class SDARAdapter(HFDecoderAdapter):
    """SDAR block diffusion checkpoints (``JetLM/SDAR-*``).

    A block attends to itself and to earlier blocks. The checkpoint's
    configuration does not name the mask token, so pass ``mask_token_id`` to
    ``cache.request`` (the tokenizer's ``mask_token``).
    """

    family = "block-causal"
    name = "sdar"
    model_types = ("sdar",)


FAST_DLLM_V2_MASK_TOKEN_ID = 151665


class FastDLLMv2Adapter(HFDecoderAdapter):
    """Fast-dLLM v2 block diffusion checkpoints (``Efficient-Large-Model/Fast_dLLM_v2_*``).

    The model is adapted from an autoregressive one and keeps its next-token
    head: the hidden state at a position predicts the token after it. The Zoo
    returns logits aligned with the position they predict, so a sampler indexes
    them like any other model's. The first position of a block is predicted from
    the last position of the block before it.
    """

    family = "block-causal"
    name = "fast-dllm-v2"
    model_types = ("fast_dllm_qwen",)
    logit_shift = 1

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        if self.mask_token_id is None:
            self.mask_token_id = FAST_DLLM_V2_MASK_TOKEN_ID
        block = getattr(self.config, "bd_size", None)
        self.block_length = int(block) if block else None
