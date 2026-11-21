"""dLLM-Cache (Liu et al., ICML 2026).

Behaviour follows the authors' LLaDA hook, which replaces each block's forward:
https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/dllm_cache/hooks/cache_hook_LLaDA.py

The method caches features, not only keys and values. Every layer keeps its
attention output and its MLP output for every position, and the residual stream
is rebuilt from them. The prompt and the response are refreshed on separate
schedules. Between response refreshes, each layer recomputes the response
positions whose value vectors moved the most since they were cached.

Unlike the reference, which keeps one process-wide cache object, the state here
belongs to the request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..positions import Positions, count, put_, take
from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


class DLLMCache(CachePolicy):
    """Feature cache with separate prompt and response schedules.

    ``prompt_interval`` and ``response_interval`` are the refresh periods in
    steps (``Kp`` and ``Kr`` in the paper). ``transfer_ratio`` is the share of
    response positions each layer updates on a step without a response refresh
    (``rho``). The defaults are the authors' demo settings. The first layer is
    recomputed in full on every step, as in the reference.
    """

    name = "dllm-cache"
    summary = (
        "Caches attention and MLP outputs; refreshes prompt and response on separate "
        "schedules and updates the response positions whose values changed most."
    )
    paper = "https://proceedings.mlr.press/v306/liu26fi.html"
    reference = "maomaocun/dLLM-cache@17235bf dllm_cache/hooks/cache_hook_LLaDA.py (Apache-2.0)"

    def __init__(
        self,
        adapter,
        prompt_interval: int = 100,
        response_interval: int = 7,
        transfer_ratio: float = 0.25,
    ):
        super().__init__(adapter)
        if prompt_interval < 1 or response_interval < 1:
            raise ValueError("Refresh intervals must be at least 1.")
        if not 0.0 <= transfer_ratio <= 1.0:
            raise ValueError("transfer_ratio must lie in [0, 1].")
        self.prompt_interval = int(prompt_interval)
        self.response_interval = int(response_interval)
        self.transfer_ratio = float(transfer_ratio)

    def config(self) -> dict[str, Any]:
        return {
            "prompt_interval": self.prompt_interval,
            "response_interval": self.response_interval,
            "transfer_ratio": self.transfer_ratio,
        }

    def start(self, req: Request) -> None:
        req.data["attn"] = [None] * self.adapter.num_layers
        req.data["mlp"] = [None] * self.adapter.num_layers

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        a = self.adapter
        prompt_len, length = req.prompt_length, tokens.size(1)
        refresh_prompt = req.step % self.prompt_interval == 0
        refresh_response = req.step % self.response_interval == 0
        x = a.embed(tokens)
        computed = 0.0
        for layer in range(a.num_layers):
            first = layer == 0 or req.data["attn"][layer] is None
            x, n = self._layer(
                req, layer, x, refresh_prompt or first, refresh_response or first, prompt_len
            )
            computed += n
        req.stats.record(computed / a.num_layers, length)
        pos = slice(prompt_len, length)
        return Computed(a.head(x[:, pos]), pos)

    # ------------------------------------------------------------------ layer

    def _layer(
        self,
        req: Request,
        layer: int,
        x: torch.Tensor,
        refresh_prompt: bool,
        refresh_response: bool,
        prompt_len: int,
    ) -> tuple[torch.Tensor, int]:
        a, kv = self.adapter, req.kv
        attn_cache, mlp_cache = req.data["attn"], req.data["mlp"]
        length = x.size(1)
        response = slice(prompt_len, length)
        num_select = int((length - prompt_len) * self.transfer_ratio)
        adaptive = not refresh_response and num_select > 0

        # Adaptive update: values are recomputed for the whole response, and the
        # positions whose value moved most are the ones that get new Q, K and outputs.
        selected: torch.Tensor | None = None
        xn_selected: torch.Tensor | None = None
        if adaptive:
            xn_response = a.attn_norm(layer, x[:, response])
            v_new = a.project_v(layer, xn_response)
            _, v_old = kv.get(layer, response)
            similarity = F.cosine_similarity(_merge(v_new), _merge(v_old), dim=-1)
            selected = similarity.topk(num_select, dim=-1, largest=False).indices.sort(-1).values
            kv.values[layer][:, :, response] = v_new  # type: ignore[index]
            xn_selected = take(xn_response, selected, dim=1)
            selected = selected + prompt_len

        pos: Positions
        if refresh_prompt and refresh_response:
            pos = None
        elif refresh_response:
            pos = response
        elif refresh_prompt:
            pos = slice(0, prompt_len)
            if selected is not None:
                prompt_index = torch.arange(prompt_len, device=x.device).expand(x.size(0), -1)
                pos = torch.cat([prompt_index, selected], dim=1)
        elif selected is not None:
            pos = selected
        else:
            # Nothing to recompute in this layer: rebuild the stream from the caches.
            x = x + attn_cache[layer]
            return x + mlp_cache[layer], 0

        if selected is None or xn_selected is None:
            xn = a.attn_norm(layer, take(x, pos))
        elif refresh_prompt:
            xn = torch.cat([a.attn_norm(layer, x[:, :prompt_len]), xn_selected], dim=1)
        else:
            xn = xn_selected
        rot = a.rotary(pos, length, x)
        q = a.project_q(layer, xn, rot)
        k = a.project_k(layer, xn, rot)
        if pos is None:
            kv.set(layer, k, a.project_v(layer, xn))
        elif selected is None:
            kv.put(layer, pos, k, a.project_v(layer, xn))
        else:
            put_(kv.keys[layer], pos, k, dim=2)  # type: ignore[arg-type]
            if refresh_prompt:
                kv.values[layer][:, :, :prompt_len] = a.project_v(  # type: ignore[index]
                    layer, xn[:, :prompt_len]
                )
        keys, values = kv.get(layer)
        att, _ = a.attend(layer, q, keys, values)
        if pos is None:
            attn_cache[layer] = att
        else:
            put_(attn_cache[layer], pos, att, dim=1)
        x = x + attn_cache[layer]

        out = a.mlp(layer, take(x, pos))
        if pos is None:
            mlp_cache[layer] = out
        else:
            put_(mlp_cache[layer], pos, out, dim=1)
        return x + mlp_cache[layer], count(pos, length)


def _merge(t: torch.Tensor) -> torch.Tensor:
    """``(B, H, N, D)`` to ``(B, N, H * D)``: the value vector of each position."""
    b, h, n, d = t.shape
    return t.transpose(1, 2).reshape(b, n, h * d)
