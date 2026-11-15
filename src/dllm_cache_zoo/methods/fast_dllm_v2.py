"""Fast-dLLM v2 sub-block cache (Wu et al., ICLR 2026).

Behaviour follows the generation function shipped with the checkpoint, with
``use_block_cache`` set:
https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B/blob/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py
https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/generation_functions.py#L17-L168

The authors call the cache hierarchical. The block level stores completed
blocks, which is exact for a block diffusion model. The sub-block level works
inside the active block: one pass over the whole block stores its keys and
values, and later steps run only the sub-block being refined, reading the rest
of the block from that pass. This second level is an approximation, the dual
cache of Fast-dLLM v1 applied within a block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import Computed
from .block_cache import BlockCache, run_blocks

if TYPE_CHECKING:
    from ..request import Request


class SubBlockCache(BlockCache):
    """Completed blocks stored once, plus a dual cache inside the active block.

    Pass the sub-block being refined as ``active``. A step runs the whole block
    when the block has not been run yet or the first position of the sub-block
    is still masked, since that position is predicted from the one before it,
    which belongs to the previous sub-block. Every other step runs the sub-block
    alone. On a model without a next-token head the first step of each sub-block
    runs the whole block instead.

    ``block_length`` is the model's block size (the checkpoint's ``bd_size`` by
    default). With one sub-block per block the method reduces to the block cache.
    """

    name = "fast-dllm-v2-sub-block"
    families = ("block-causal",)
    summary = (
        "Stores completed blocks once and, inside the active block, runs only the sub-block "
        "being refined."
    )
    paper = "https://arxiv.org/abs/2509.26328"
    reference = "Efficient-Large-Model/Fast_dLLM_v2_7B@0661abf modeling.py generate (Apache-2.0)"

    def step(
        self, req: Request, tokens: torch.Tensor, block: tuple[int, int], block_length: int
    ) -> Computed:
        start, end = req.active
        shift = self.adapter.logit_shift
        if shift:
            refresh = bool((tokens[:, start] == req.mask_token_id).any())
        else:
            refresh = req.block_step == 0
        if refresh or not req.data["block_ready"] or (start, end) == block:
            return self.run_block(req, tokens, block, block_length)
        logits = run_blocks(self, req, tokens, start, end, block_length, read_end=block[1])
        assert logits is not None
        req.stats.record(end - start, block[1])
        return Computed(logits[:, : end - start - shift], slice(start + shift, end))
