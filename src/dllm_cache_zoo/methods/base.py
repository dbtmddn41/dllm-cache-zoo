"""Base class for cache methods and the forward pass they share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..adapters.base import ModelAdapter
from ..positions import Positions, count, take

if TYPE_CHECKING:
    from ..request import Request


@dataclass
class Computed:
    """Logits for the positions a method ran through the model.

    ``positions`` follows the convention of :mod:`dllm_cache_zoo.positions`:
    ``None`` for the whole sequence, a ``slice``, or a ``(B, N)`` index.
    """

    logits: torch.Tensor
    positions: Positions = None


class CachePolicy(ABC):
    """A cache method: which positions to run at each step and what to store.

    A policy holds configuration only. Everything that belongs to one generation
    lives on the :class:`~dllm_cache_zoo.request.Request` it is handed, so one
    attached policy can serve many requests.
    """

    #: Registry name, e.g. ``"fast-dllm-dual"``.
    name: ClassVar[str]
    #: Model families the method is defined for.
    families: ClassVar[tuple[str, ...]] = ("bidirectional",)
    #: One line for ``zoo.methods()``.
    summary: ClassVar[str] = ""
    paper: ClassVar[str] = ""
    #: Where the behaviour was taken from (repository and revision, or "paper").
    reference: ClassVar[str] = ""

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter

    def config(self) -> dict[str, Any]:
        """Settings that change behaviour, for logs and run manifests."""
        return {}

    def start(self, req: Request) -> None:  # noqa: B027
        """Called once when a request opens."""

    def on_block_start(self, req: Request, tokens: torch.Tensor) -> None:  # noqa: B027
        """Called before the first forward of each active span."""

    @abstractmethod
    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        """Run one denoising step on ``tokens`` of shape ``(B, L)``."""

    # ------------------------------------------------------------ shared pass

    def run(
        self,
        req: Request,
        tokens: torch.Tensor,
        q_pos: Positions = None,
        *,
        store: bool,
        key_pos: Positions = None,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
        on_attention: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor | None], None]
        | None = None,
    ) -> torch.Tensor:
        """Run every layer for the positions ``q_pos`` and return their logits.

        With ``store`` the fresh keys and values of ``q_pos`` are written into
        the request's store and each query reads the stored sequence (or the
        stored entries at ``key_pos``). Without it the queries only read each
        other, which is a plain forward over ``q_pos``.

        ``mask`` is a boolean attention mask, True where a query may read a key.

        ``on_attention(layer, q, k, weights)`` is called after each attention
        with the queries, the keys they read and the attention weights (when
        ``need_weights`` is set).
        """
        a = self.adapter
        length = tokens.size(1)
        x = a.embed(take(tokens, q_pos))
        rot = a.rotary(q_pos, length, x)
        kv = req.kv
        for layer in range(a.num_layers):
            xn = a.attn_norm(layer, x)
            q = a.project_q(layer, xn, rot)
            k = a.project_k(layer, xn, rot)
            v = a.project_v(layer, xn)
            if store:
                if q_pos is None:
                    kv.set(layer, k, v)
                else:
                    if not kv.has(layer):
                        kv.allocate(layer, length, k, v)
                    kv.put(layer, q_pos, k, v)
                if q_pos is not None or key_pos is not None:
                    k, v = kv.get(layer, key_pos)
            att, weights = a.attend(layer, q, k, v, mask=mask, need_weights=need_weights)
            if on_attention is not None:
                on_attention(layer, q, k, weights)
            x = x + att
            x = x + a.mlp(layer, x)
        req.stats.record(count(q_pos, length), length)
        return a.head(x)
