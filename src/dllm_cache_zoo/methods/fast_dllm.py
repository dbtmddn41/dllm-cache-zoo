"""Fast-dLLM v1 prefix and dual caches (Wu et al., ICLR 2026).

Behaviour follows the authors' LLaDA loops ``generate_with_prefix_cache`` and
``generate_with_dual_cache``:
https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L132-L272

The first pass of each block runs the whole sequence and stores every key and
value. Later passes of the block run only part of the sequence against that
store. Confidence-aware parallel decoding, the paper's other contribution, is a
sampler setting and stays with the host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request

_PAPER = "https://arxiv.org/abs/2505.22618"
_REFERENCE = "NVlabs/Fast-dLLM@a9b81e4 v1/llada/generate.py (Apache-2.0)"


class PrefixCache(CachePolicy):
    """Reuse the prompt and completed blocks; recompute the active block and the suffix."""

    name = "fast-dllm-prefix"
    summary = "Reuses keys and values before the active block; recomputes the block and suffix."
    paper = _PAPER
    reference = _REFERENCE

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        if req.block_step == 0:
            return Computed(self.run(req, tokens, None, store=True))
        pos = slice(req.active[0], tokens.size(1))
        return Computed(self.run(req, tokens, pos, store=True), pos)


class DualCache(CachePolicy):
    """Reuse both sides of the active block, including the masked suffix."""

    name = "fast-dllm-dual"
    summary = "Reuses keys and values on both sides of the active block; recomputes the block."
    paper = _PAPER
    reference = _REFERENCE

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        if req.block_step == 0:
            return Computed(self.run(req, tokens, None, store=True))
        pos = slice(*req.active)
        return Computed(self.run(req, tokens, pos, store=True), pos)
