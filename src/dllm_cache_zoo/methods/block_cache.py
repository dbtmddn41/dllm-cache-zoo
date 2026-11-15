"""Completed-block cache for block diffusion models.

Block diffusion (Arriola et al., ICLR 2025) limits attention so that a block
sees itself and earlier blocks. A completed block can therefore be stored once
and reused exactly. Behaviour follows the SDAR generation script, which writes
the prompt blocks first and each block once it has no masks left:
https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/generate.py#L58-L202

Here a completed block is written when the loop moves on to the next one, in
one pass over everything between the stored prefix and the new active block.
The store is preallocated and read as a view, so a denoising step copies only
the active block's keys and values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


def block_causal_mask(
    q_start: int,
    q_end: int,
    block_length: int,
    device: torch.device,
    k_end: int | None = None,
) -> torch.Tensor | None:
    """Mask for queries ``[q_start, q_end)`` over keys ``[0, k_end)``.

    A query may read a key whose block is not later than its own. Blocks are
    aligned to multiples of ``block_length`` from position 0. ``k_end`` defaults
    to ``q_end``. Returns None when every query may read every key, which is the
    case inside a single block.
    """
    q_block = torch.arange(q_start, q_end, device=device) // block_length
    k_block = torch.arange(0, q_end if k_end is None else k_end, device=device) // block_length
    mask = k_block[None, :] <= q_block[:, None]
    if bool(mask.all()):
        return None
    return mask[None, None]


def enclosing_block(span: tuple[int, int], block_length: int, length: int) -> tuple[int, int]:
    """The block that holds ``span``, as ``(start, end)`` clipped to the sequence."""
    start = (span[0] // block_length) * block_length
    if span[1] > start + block_length:
        raise ValueError(
            f"Active span {span} crosses a block boundary. With a block length of "
            f"{block_length}, pass a span inside one block."
        )
    return start, min(start + block_length, length)


def run_blocks(
    policy: CachePolicy,
    req: Request,
    tokens: torch.Tensor,
    start: int,
    end: int,
    block_length: int,
    *,
    logits: bool | str = True,
    read_end: int | None = None,
) -> torch.Tensor | None:
    """Run positions ``[start, end)`` against the store.

    Fresh keys and values are written at ``[start, end)`` and the queries read
    the stored entries ``[0, read_end)``, by default up to ``end``. The store
    grows as needed. ``logits`` is True for the logits of every position run,
    ``"last"`` for those of the last one, or False to only fill the store.
    """
    a, kv = policy.adapter, req.kv
    pos = slice(start, end)
    read_end = end if read_end is None else read_end
    x = a.embed(tokens[:, pos])
    rot = a.rotary(pos, end, x)
    mask = block_causal_mask(start, end, block_length, tokens.device, read_end)
    capacity = max(tokens.size(1), read_end)
    for layer in range(a.num_layers):
        xn = a.attn_norm(layer, x)
        q = a.project_q(layer, xn, rot)
        k = a.project_k(layer, xn, rot)
        v = a.project_v(layer, xn)
        if not kv.has(layer):
            kv.allocate(layer, capacity, k, v)
        elif kv.length(layer) < read_end:
            kv.grow(layer, max(capacity, 2 * kv.length(layer)))
        kv.put(layer, pos, k, v)
        if logits is False and layer == a.num_layers - 1:
            return None  # the last layer's output is only needed for the head
        keys, values = kv.get(layer, slice(0, read_end))
        att, _ = a.attend(layer, q, keys, values, mask=mask)
        x = x + att
        x = x + a.mlp(layer, x)
    return a.head(x[:, -1:] if logits == "last" else x)


class BlockCache(CachePolicy):
    """Store each block once it is complete; run only the active block.

    Exact for a block-causal model: the logits match a pass over the whole
    prefix with the block attention mask. ``block_length`` is the model's block
    size. By default it is the one the checkpoint's configuration names, or
    else the length of the first active span. Blocks are aligned to multiples of
    it from position 0, as in SDAR, so the last prompt tokens can share a block
    with the first generated ones. The active span may be a part of a block: the
    whole block is run and its logits are returned.

    A model with a next-token head (Fast-dLLM v2) predicts the first position of
    a block from the last position of the block before it. While that position
    is masked, a step returns its logits alone, taken from the pass that stored
    the previous block.
    """

    name = "block-cache"
    families = ("block-causal",)
    summary = "Stores keys and values of completed blocks once; runs only the active block (exact)."
    paper = "https://arxiv.org/abs/2503.09573"
    reference = "JetAstra/SDAR@4c2749b generate.py (MIT)"

    def __init__(self, adapter, block_length: int | None = None):
        super().__init__(adapter)
        self.block_length = block_length

    def config(self) -> dict[str, Any]:
        return {"block_length": self.block_length or self.adapter.block_length}

    def start(self, req: Request) -> None:
        req.data.update(
            committed=0,
            block_length=self.block_length or self.adapter.block_length,
            block_ready=False,
            next_logits=None,
        )

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        st = req.data
        shift = self.adapter.logit_shift
        if st["block_length"] is None:
            st["block_length"] = req.active[1] - req.active[0]
        block_length = st["block_length"]
        start, end = enclosing_block(req.active, block_length, tokens.size(1))
        if st["committed"] > start:
            raise ValueError(
                f"Active span {req.active} lies before position {st['committed']}, which is "
                "already stored. Block diffusion decodes blocks left to right."
            )
        if st["committed"] < start:
            # Everything before the active block is final: store it in one pass. With a
            # next-token head, its last position predicts the first one of this block.
            st["next_logits"] = run_blocks(
                self,
                req,
                tokens,
                st["committed"],
                start,
                block_length,
                logits="last" if shift else False,
            )
            req.stats.record(start - st["committed"], end)
            st["committed"] = start
            st["block_ready"] = False
        if shift and bool((tokens[:, start] == req.mask_token_id).any()):
            if st["next_logits"] is None:
                raise ValueError(
                    "This model predicts each token from the position before it, so the first "
                    "position of the sequence cannot be masked."
                )
            return Computed(st["next_logits"], slice(start, start + 1))
        return self.step(req, tokens, (start, end), block_length)

    def step(
        self, req: Request, tokens: torch.Tensor, block: tuple[int, int], block_length: int
    ) -> Computed:
        """One denoising step inside ``block``, whose prefix is stored."""
        return self.run_block(req, tokens, block, block_length)

    def run_block(
        self, req: Request, tokens: torch.Tensor, block: tuple[int, int], block_length: int
    ) -> Computed:
        start, end = block
        shift = self.adapter.logit_shift
        logits = run_blocks(self, req, tokens, start, end, block_length)
        assert logits is not None
        req.stats.record(end - start, end)
        req.data["block_ready"] = True
        return Computed(logits[:, : end - start - shift], slice(start + shift, end))
