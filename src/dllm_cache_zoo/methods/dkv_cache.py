"""dKV-Cache (Ma et al., NeurIPS 2025).

Written from the paper. The authors' repository has no license file at the
checked revision, so no code is taken from it; its LLaDA drivers were read to
match step counting and the refresh schedule:
https://github.com/horseee/dKV-Cache/tree/49a76fcc43b744ec2d960137f216e419317138b1

A decoded token's keys and values barely change once it has been seen as a real
token, so they can be stored. The delay matters: a position is stored one step
after it is decoded, when it has passed through the model with its token in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..positions import mask_to_index
from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request

_PAPER = "https://arxiv.org/abs/2505.15781"


class DKVCacheDecode(CachePolicy):
    """dKV-Cache-Decode: recompute masked positions and those decoded on the previous step.

    Within a block, step 0 runs the whole sequence without storing. Step 1 and
    every ``refresh_interval``-th step run the whole sequence and store it.
    Every other step runs only the positions that were still masked on the
    previous step, and reads the rest from the store.
    """

    name = "dkv-cache-decode"
    summary = "Reuses decoded positions one step after they are decoded, with periodic refresh."
    paper = _PAPER
    reference = "paper; step schedule matched to horseee/dKV-Cache@49a76fc (no license)"

    def __init__(self, adapter, refresh_interval: int = 8):
        super().__init__(adapter)
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be at least 1.")
        self.refresh_interval = int(refresh_interval)

    def config(self) -> dict[str, Any]:
        return {"refresh_interval": self.refresh_interval}

    def on_block_start(self, req: Request, tokens: torch.Tensor) -> None:
        req.data["stored"] = None

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        step = req.block_step
        decoded = tokens != req.mask_token_id
        if step == 0:
            return Computed(self.run(req, tokens, None, store=False))
        stored: torch.Tensor | None = req.data.get("stored")
        if step == 1 or step % self.refresh_interval == 0 or stored is None:
            logits = self.run(req, tokens, None, store=True)
            req.data["stored"] = decoded
            return Computed(logits)
        # `stored` marks positions that held a token on the previous step's input.
        # Everything else is recomputed: current masks and tokens revealed since.
        pos, _ = mask_to_index(~stored)
        logits = self.run(req, tokens, pos, store=True)
        req.data["stored"] = decoded
        return Computed(logits, pos)


def nearby_masked(
    selected: torch.Tensor, anchors: torch.Tensor, masked: torch.Tensor, window: int
) -> torch.Tensor:
    """Add up to ``window`` masked positions around each anchor to ``selected``.

    For one sequence: ``selected`` and ``masked`` are boolean ``(L,)``, ``anchors``
    holds positions. From each anchor the search steps outwards, left then right
    in turn, and takes masked positions that are not selected yet.
    """
    chosen = selected.tolist()
    is_masked = masked.tolist()
    length = len(chosen)
    for anchor in anchors.tolist():
        left, right, found, go_left = anchor - 1, anchor + 1, 0, True
        while found < window and (left >= 0 or right < length):
            ptr = left if go_left else right
            if 0 <= ptr < length and is_masked[ptr] and not chosen[ptr]:
                chosen[ptr] = True
                found += 1
            if go_left:
                left -= 1
            else:
                right += 1
            go_left = not go_left
    return torch.tensor(chosen, dtype=torch.bool, device=selected.device)


class DKVCacheGreedy(CachePolicy):
    """dKV-Cache-Greedy: recompute only the tokens around the current decoding step.

    Between refreshes a step runs the tokens revealed on the previous step, the
    positions the sampler is about to reveal, and up to ``window`` masked
    positions next to each of those. Everything else, masked positions included,
    is read from the store. Every ``refresh_interval``-th step of a block runs
    the whole sequence.

    The method needs to know which positions the sampler will reveal before the
    forward pass, so it fits samplers whose order is fixed in advance, such as
    random-order unmasking. Pass them as ``state.forward(x, reveal=...)``.
    """

    name = "dkv-cache-greedy"
    summary = (
        "Recomputes the tokens just revealed, those about to be revealed and nearby masks; "
        "needs the sampler's next positions."
    )
    paper = _PAPER
    reference = "paper; step schedule matched to horseee/dKV-Cache@49a76fc (no license)"

    def __init__(self, adapter, refresh_interval: int = 2, window: int = 4):
        super().__init__(adapter)
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be at least 1.")
        if window < 0:
            raise ValueError("window must not be negative.")
        self.refresh_interval = int(refresh_interval)
        self.window = int(window)

    def config(self) -> dict[str, Any]:
        return {"refresh_interval": self.refresh_interval, "window": self.window}

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        if req.block_step % self.refresh_interval == 0 or not req.kv.has(0):
            return Computed(self.run(req, tokens, None, store=True))
        if req.reveal is None:
            raise ValueError(
                "dkv-cache-greedy needs the positions the sampler will reveal at this step. "
                "Pass them as `state.forward(x, reveal=...)`, or use dkv-cache-decode with a "
                "sampler that chooses positions from the logits."
            )
        selected = req.newly_revealed(tokens) | req.reveal
        if self.window > 0:
            masked = tokens == req.mask_token_id
            rows = [
                nearby_masked(
                    selected[b], req.reveal[b].nonzero().squeeze(-1), masked[b], self.window
                )
                for b in range(tokens.size(0))
            ]
            selected = torch.stack(rows)
        pos, _ = mask_to_index(selected)
        return Computed(self.run(req, tokens, pos, store=True), pos)
