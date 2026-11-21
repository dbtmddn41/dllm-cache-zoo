"""Sparse-dLLM dynamic cache eviction (Song et al., AAAI 2026).

Written from the paper. The authors' repository has no license file at the
checked revision, so no code is taken from it; its LLaDA loop was read to match
the step schedule and the scoring details:
https://github.com/OpenMOSS/Sparse-dLLM/tree/3fd8986bee4ddd68e70ee8041da3a8c9de44f405

Each block starts with full passes. On the last of them every layer stores its
keys and values, scores each entry outside the active block against the block's
mean query, and keeps the top share. Later steps run only the active block,
which attends to itself and to the kept entries. Evicted entries are dropped, so
memory stays close to that of the uncached model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..positions import take
from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


class SparseDLLM(CachePolicy):
    """Keep the entries the active block attends to; evict the rest.

    ``keep_ratio`` is the share of entries outside the active block that each
    layer keeps. ``kernel_size`` is the width of a max pool over neighbouring
    scores, which keeps an entry that sits next to a high-scoring one (``None``
    or 1 turns it off; it must be odd). ``delay`` is the number of full passes
    in a block before the pass that stores and evicts. The released loops use 1.
    """

    name = "sparse-dllm"
    summary = (
        "Stores keys and values on a block's second pass, evicts the lowest-scoring entries "
        "outside the block, then runs only the block."
    )
    paper = "https://arxiv.org/abs/2508.02558"
    reference = "paper; schedule and scoring matched to OpenMOSS/Sparse-dLLM@3fd8986 (no license)"

    def __init__(
        self,
        adapter,
        keep_ratio: float = 0.5,
        kernel_size: int | None = 3,
        delay: int = 1,
    ):
        super().__init__(adapter)
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must lie in (0, 1].")
        if kernel_size is not None and kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if delay < 0:
            raise ValueError("delay must not be negative.")
        self.keep_ratio = float(keep_ratio)
        self.kernel_size = kernel_size
        self.delay = int(delay)

    def config(self) -> dict[str, Any]:
        return {
            "keep_ratio": self.keep_ratio,
            "kernel_size": self.kernel_size,
            "delay": self.delay,
        }

    def on_block_start(self, req: Request, tokens: torch.Tensor) -> None:
        n = self.adapter.num_layers
        req.data["k"] = [None] * n
        req.data["v"] = [None] * n
        req.data["kept"] = [None] * n

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        step = req.block_step
        if step < self.delay:
            return Computed(self.run(req, tokens, None, store=False))
        if step == self.delay:
            return Computed(self._store_and_evict(req, tokens))
        return self._block_only(req, tokens)

    def _store_and_evict(self, req: Request, tokens: torch.Tensor) -> torch.Tensor:
        a, st = self.adapter, req.data
        start, end = req.active
        length = tokens.size(1)
        x = a.embed(tokens)
        rot = a.rotary(None, length, x)
        outside = torch.cat(
            [torch.arange(start, device=x.device), torch.arange(end, length, device=x.device)]
        )
        num_keep = int(outside.numel() * self.keep_ratio)
        for layer in range(a.num_layers):
            xn = a.attn_norm(layer, x)
            q = a.project_q(layer, xn, rot)
            k = a.project_k(layer, xn, rot)
            v = a.project_v(layer, xn)
            att, _ = a.attend(layer, q, k, v)

            k_out = torch.cat([k[:, :, :start], k[:, :, end:]], dim=2)
            v_out = torch.cat([v[:, :, :start], v[:, :, end:]], dim=2)
            # Score of an entry: its key against the block's mean query, averaged over heads.
            mean_q = q[:, :, start:end].mean(dim=2, keepdim=True)
            k_heads = k_out
            if k_out.size(1) != q.size(1):
                k_heads = k_out.repeat_interleave(q.size(1) // k_out.size(1), dim=1)
            score = torch.matmul(mean_q, k_heads.transpose(-2, -1)).squeeze(-2).mean(dim=1)
            if self.kernel_size is not None and self.kernel_size > 1:
                score = F.max_pool1d(
                    score.unsqueeze(1).float(),
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                ).squeeze(1)
            keep = score.topk(num_keep, dim=-1).indices.sort(dim=-1).values
            st["k"][layer] = take(k_out, keep, dim=2)
            st["v"][layer] = take(v_out, keep, dim=2)
            st["kept"][layer] = outside[keep]

            x = x + att
            x = x + a.mlp(layer, x)
        req.stats.record(length, length)
        return a.head(x)

    def _block_only(self, req: Request, tokens: torch.Tensor) -> Computed:
        a, st = self.adapter, req.data
        start, end = req.active
        length = tokens.size(1)
        pos = slice(start, end)
        x = a.embed(tokens[:, pos])
        rot = a.rotary(pos, length, x)
        for layer in range(a.num_layers):
            xn = a.attn_norm(layer, x)
            q = a.project_q(layer, xn, rot)
            k = torch.cat([st["k"][layer], a.project_k(layer, xn, rot)], dim=2)
            v = torch.cat([st["v"][layer], a.project_v(layer, xn)], dim=2)
            att, _ = a.attend(layer, q, k, v)
            x = x + att
            x = x + a.mlp(layer, x)
        req.stats.record(end - start, length)
        return Computed(a.head(x), pos)
