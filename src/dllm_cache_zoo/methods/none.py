"""Full recomputation: the control every cache method is compared with."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


class NoCache(CachePolicy):
    """Run the whole sequence at every step and store nothing.

    The pass goes through the adapter, so it exercises the same code as the
    cache methods and should match the model's own forward. For a block-causal
    model it runs everything up to the end of the active block under the block
    attention mask. ``block_length`` is the model's block size. By default it is
    the one the checkpoint's configuration names, or else the length of the
    first active span.
    """

    name = "none"
    families = ("bidirectional", "block-causal")
    summary = "Full recomputation at every step (baseline)."

    def __init__(self, adapter, block_length: int | None = None):
        super().__init__(adapter)
        self.block_length = block_length

    def config(self) -> dict[str, Any]:
        if self.adapter.family == "block-causal":
            return {"block_length": self.block_length or self.adapter.block_length}
        return {}

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        if self.adapter.family != "block-causal":
            return Computed(self.run(req, tokens, None, store=False))
        from .block_cache import block_causal_mask, enclosing_block

        start, end = req.active
        block_length = (
            self.block_length
            or self.adapter.block_length
            or req.data.setdefault("block_length", end - start)
        )
        block_end = enclosing_block(req.active, block_length, tokens.size(1))[1]
        mask = block_causal_mask(0, block_end, block_length, tokens.device)
        logits = self.run(req, tokens[:, :block_end], slice(0, block_end), store=False, mask=mask)
        # With a next-token head, position p is predicted by the hidden state at p - shift.
        shift = self.adapter.logit_shift
        first = max(start, shift)
        return Computed(logits[:, first - shift : end - shift], slice(first, end))
