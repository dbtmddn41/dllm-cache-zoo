"""Elastic-Cache (Nguyen-Tri et al., ICLR 2026).

Behaviour follows the authors' LLaDA implementation:
https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/generate.py#L44-L147
https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/model/modeling_llada.py#L729-L838

A step runs three groups of positions: the masked positions of the active
window, the tokens revealed on the previous step, and the tokens that the
window attended to most on the previous step (the tracked tokens). At every
layer the method compares the tracked tokens' attention with what the stored
keys would have given. When the two drift apart, that layer's successors are
recomputed for the whole sequence, starting from the stored inputs of the next
layer. The cache therefore refreshes when and where attention says it is stale,
from the deeper layers first.

The sliding window and confidence-aware decoding of the authors' recipe belong
to the host loop: pass the window as ``active``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..positions import put_
from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


class ElasticCache(CachePolicy):
    """Attention-aware refresh with layer-wise restart.

    ``gamma`` is the similarity below which a refresh starts, ``track_num`` the
    number of most-attended tokens each layer watches. With ``block_caching``
    (the authors' default) only the masked positions inside the active span are
    run; without it every masked position is. The reference handles one
    sequence at a time, and so does this policy.
    """

    name = "elastic-cache"
    summary = (
        "Runs the active window, new tokens and the most-attended tokens; refreshes deeper "
        "layers for the whole sequence when tracked attention drifts."
    )
    paper = "https://arxiv.org/abs/2510.14973"
    reference = "VILA-Lab/Elastic-Cache@1960d8f llada/ (Apache-2.0)"

    def __init__(self, adapter, gamma: float = 0.9, track_num: int = 1, block_caching: bool = True):
        super().__init__(adapter)
        self.gamma = float(gamma)
        self.track_num = int(track_num)
        self.block_caching = bool(block_caching)

    def config(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "track_num": self.track_num,
            "block_caching": self.block_caching,
        }

    def start(self, req: Request) -> None:
        n = self.adapter.num_layers
        req.data.update(x=[None] * n, q=[None] * n, track=None, refreshed_from=[])

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        if tokens.size(0) != 1:
            raise NotImplementedError(
                "elastic-cache runs one sequence at a time, as its reference does. "
                "Open one request per sequence."
            )
        a, kv, st = self.adapter, req.kv, req.data
        num_layers = a.num_layers
        length = tokens.size(1)
        device = tokens.device

        is_mask = tokens[0] == req.mask_token_id
        masked = is_mask.nonzero().squeeze(-1)
        if self.block_caching:
            start, end = req.active
            window = masked[(masked >= start) & (masked < end)]
        else:
            window = masked

        first = st["track"] is None
        if first:
            query = torch.arange(length, device=device)
            restart = -1  # every layer runs the whole sequence
            n_track = 0
        else:
            revealed = req.newly_revealed(tokens)[0].nonzero().squeeze(-1)
            track = st["track"]
            # A tracked token cannot be one of the others: it held a token on the
            # previous step, and the others were masked then.
            query = torch.cat([track, revealed, window])
            restart = num_layers  # no layer restarts unless attention drifts
            n_track = track.numel()

        pos = None if first else query.unsqueeze(0)
        x = a.embed(tokens if first else tokens[:, query])
        rot_query = a.rotary(pos, length, x)
        rot_all = rot_query if first else a.rotary(None, length, x)
        is_full = first
        tracked: list[torch.Tensor] = []
        computed = 0.0

        for layer in range(num_layers):
            if layer > restart:
                st["x"][layer] = x
            else:
                put_(st["x"][layer], pos, x, dim=1)
                if layer == restart:
                    # Drift was found in the previous layer: continue with the whole
                    # sequence, taking unvisited positions from this layer's stored input.
                    x = st["x"][layer]
                    is_full = True
            refresh = layer >= restart
            computed += x.size(1)

            xn = a.attn_norm(layer, x)
            rot = rot_all if refresh else rot_query
            q = a.project_q(layer, xn, rot)
            k = a.project_k(layer, xn, rot)
            v = a.project_v(layer, xn)

            past_weights = None
            if refresh:
                st["q"][layer] = q
                kv.set(layer, k, v)
            else:
                # What the tracked tokens attended to with the keys stored so far.
                past_q = st["q"][layer][:, :, track]
                past_k, _ = kv.get(layer)
                scale = a.attention_scale(layer) or past_q.size(-1) ** -0.5
                past_weights = torch.softmax(past_q @ past_k.transpose(-2, -1) * scale, dim=-1)
                kv.put(layer, pos, k, v)
                put_(st["q"][layer], pos, q, dim=2)
                k, v = kv.get(layer)

            att, weights = a.attend(layer, q, k, v, need_weights=True)
            assert weights is not None

            if refresh:
                window_weights = weights[:, :, window, :]
            else:
                window_weights = weights[:, :, weights.size(2) - window.numel() :, :]
                assert past_weights is not None
                drift = F.cosine_similarity(past_weights, weights[:, :, :n_track, :], dim=1).mean()
                if drift < self.gamma:
                    restart = layer + 1

            # Track the tokens the window attends to most. Masked positions do not count.
            attended = window_weights.sum(dim=(0, 1, 2))
            attended[masked] = 0.0
            tracked.append(attended.topk(k=self.track_num, dim=0, largest=True).indices)

            x = x + att
            x = x + a.mlp(layer, x)

        st["track"] = torch.cat(tracked).unique(sorted=False)
        st["refreshed_from"].append(restart if restart < num_layers else None)
        req.stats.record(computed / num_layers, length)
        return Computed(a.head(x), None if is_full else pos)
