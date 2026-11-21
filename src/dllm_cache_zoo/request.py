"""Attaching a method to a model and running one generation through it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .adapters.base import ModelAdapter
from .kv import KVStore
from .methods.base import CachePolicy, Computed
from .positions import as_index, put_


@dataclass
class Stats:
    """Work done in one request, counted in positions per forward pass."""

    forwards: int = 0
    #: Positions that went through the layers, summed over forwards.
    computed: float = 0.0
    #: Positions a full recomputation would have run, summed over forwards.
    total: float = 0.0
    #: Forward passes that ran the whole sequence.
    full_forwards: int = 0

    def record(self, computed: float, total: int) -> None:
        self.forwards += 1
        self.computed += computed
        self.total += total
        if computed >= total:
            self.full_forwards += 1

    @property
    def recomputed_share(self) -> float:
        """Share of position-passes computed, where full recomputation is 1.0."""
        return self.computed / self.total if self.total else 0.0


Span = tuple[int, int]


def _as_span(active: Any, default: Span) -> Span:
    if active is None:
        return default
    if isinstance(active, slice):
        if active.start is None or active.stop is None or active.step not in (None, 1):
            raise ValueError("`active` as a slice needs explicit start and stop and no step.")
        return int(active.start), int(active.stop)
    if isinstance(active, range):
        if active.step != 1:
            raise ValueError("`active` as a range needs a step of 1.")
        return active.start, active.stop
    start, end = active
    return int(start), int(end)


def _as_mask(positions: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
    if positions is None:
        return None
    if positions.dtype == torch.bool:
        if positions.shape != x.shape:
            raise ValueError("A boolean `reveal` must have the shape of the tokens.")
        return positions
    index = positions.view(x.size(0), -1)
    return torch.zeros_like(x, dtype=torch.bool).scatter_(1, index, True)


class Request:
    """Cache state for one generation. Create it with :meth:`CacheHandle.request`.

    Call :meth:`forward` where the loop used to call the model. ``active`` is the
    span of positions the sampler may reveal at this step, as ``(start, end)``
    absolute positions: the current block of a blockwise loop, or the whole
    response when the loop has no blocks. A new span starts a new block.
    """

    def __init__(
        self,
        policy: CachePolicy,
        adapter: ModelAdapter,
        *,
        prompt_length: int,
        mask_token_id: int,
    ):
        self.policy = policy
        self.adapter = adapter
        self.prompt_length = int(prompt_length)
        self.mask_token_id = int(mask_token_id)
        self.kv = KVStore(adapter.num_layers)
        #: Private state of the method.
        self.data: dict[str, Any] = {}
        self.stats = Stats()
        #: Forward passes completed in this request.
        self.step = 0
        #: Forward passes completed in the current active span.
        self.block_step = 0
        #: Active spans started so far.
        self.block_index = -1
        self.active: Span = (self.prompt_length, self.prompt_length)
        #: Input of the previous forward pass.
        self.prev_tokens: torch.Tensor | None = None
        #: Positions the last forward pass ran, as a boolean ``(B, L)`` mask.
        self.last_computed: torch.Tensor | None = None
        #: Positions the sampler announced it will reveal at the current step, ``(B, L)``.
        self.reveal: torch.Tensor | None = None
        #: Confidence passed to :meth:`observe` after the last forward, if any.
        self.confidence: torch.Tensor | None = None
        self._closed = False
        policy.start(self)

    # ------------------------------------------------------------------ steps

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        active: Any = None,
        *,
        output: str = "full",
        reveal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits for the current tokens ``x`` of shape ``(B, L)``.

        ``reveal`` names the positions the sampler will reveal at this step, as a
        boolean ``(B, L)`` mask or a ``(B, n)`` index. Only samplers that fix their
        order in advance know it, and only methods that say so need it.

        ``output="full"`` returns ``(B, L, V)``, with zeros at positions the
        method did not run, so an existing sampler can index it as before.
        ``output="active"`` returns only the active span, ``(B, end - start, V)``,
        which avoids building and scoring the rest of the sequence.

        Logits are valid at masked positions of the active span. Other positions
        may hold zeros.
        """
        if self._closed:
            raise RuntimeError("This request is closed. Open a new one with `cache.request(...)`.")
        if x.dim() != 2:
            raise ValueError(f"Expected token ids of shape (B, L), got {tuple(x.shape)}.")
        length = x.size(1)
        span = _as_span(active, (self.prompt_length, length))
        if not 0 <= span[0] < span[1] <= length:
            raise ValueError(f"Active span {span} does not fit a sequence of {length} positions.")
        if self.step == 0 or span != self.active:
            self.active = span
            self.block_step = 0
            self.block_index += 1
            self.policy.on_block_start(self, x)

        self.reveal = _as_mask(reveal, x)

        result = self.policy.forward(self, x)

        self.prev_tokens = x.clone()
        self.confidence = None
        self.step += 1
        self.block_step += 1
        self.last_computed = self._computed_mask(result, x)
        if output == "full":
            return self._full(result, x)
        if output == "active":
            return self._span(result, x, span)
        raise ValueError(f"Unknown output {output!r}. Use 'full' or 'active'.")

    def observe(self, x: torch.Tensor, confidence: torch.Tensor | None = None) -> None:
        """Tell the method what the sampler did. Optional.

        Methods read the revealed tokens from the next ``forward`` call. Pass
        ``confidence`` (``(B, L)``, the sampler's probability of its chosen token
        at each position) when the sampler already has it and the method uses a
        confidence signal, so it is not computed twice.
        """
        self.confidence = confidence

    def newly_revealed(self, x: torch.Tensor) -> torch.Tensor:
        """Positions that were masked in the previous input and hold a token in ``x``."""
        if self.prev_tokens is None:
            return torch.zeros_like(x, dtype=torch.bool)
        return (self.prev_tokens == self.mask_token_id) & (x != self.mask_token_id)

    # ----------------------------------------------------------------- output

    def _computed_mask(self, result: Computed, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x, dtype=torch.bool)
        pos = result.positions
        if pos is None:
            mask[:] = True
        elif isinstance(pos, slice):
            mask[:, pos] = True
        else:
            mask.scatter_(1, pos.expand(x.size(0), -1), True)
        return mask

    def _full(self, result: Computed, x: torch.Tensor) -> torch.Tensor:
        logits, pos = result.logits, result.positions
        if pos is None:
            return logits
        out = logits.new_zeros((x.size(0), x.size(1), logits.size(-1)))
        if not isinstance(pos, slice):
            pos = pos.expand(x.size(0), -1)
        return put_(out, pos, logits, dim=1)

    def _span(self, result: Computed, x: torch.Tensor, span: Span) -> torch.Tensor:
        logits, pos = result.logits, result.positions
        start, end = span
        if pos is None:
            return logits[:, start:end]
        if isinstance(pos, slice):
            lo, hi, _ = pos.indices(x.size(1))
            if lo <= start and end <= hi:
                return logits[:, start - lo : end - lo]
        index = as_index(pos, x.size(1), x.device).expand(x.size(0), -1)
        n = end - start
        local = index - start
        inside = (local >= 0) & (local < n)
        # Positions outside the span are written to a spare slot that is dropped.
        out = logits.new_zeros((x.size(0), n + 1, logits.size(-1)))
        target = torch.where(inside, local, torch.full_like(local, n))
        out.scatter_(1, target.unsqueeze(-1).expand(-1, -1, logits.size(-1)), logits)
        return out[:, :n]

    # ---------------------------------------------------------------- closing

    def close(self) -> None:
        """Drop stored keys, values and features."""
        self.kv.clear()
        self.data.clear()
        self.prev_tokens = None
        self.last_computed = None
        self._closed = True

    def __enter__(self) -> Request:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class CacheHandle:
    """A cache method attached to a model. Create it with :func:`dllm_cache_zoo.attach`."""

    model: torch.nn.Module
    adapter: ModelAdapter
    policy: CachePolicy
    method: str
    settings: dict[str, Any] = field(default_factory=dict)

    def request(self, prompt_length: int, *, mask_token_id: int | None = None) -> Request:
        """Open cache state for one generation.

        ``prompt_length`` is the number of leading positions that hold the prompt.
        ``mask_token_id`` defaults to the model's mask token.
        """
        mask_id = mask_token_id if mask_token_id is not None else self.adapter.mask_token_id
        if mask_id is None:
            raise ValueError(
                "The model does not declare a mask token. Pass `mask_token_id` to `request`."
            )
        return Request(
            self.policy, self.adapter, prompt_length=prompt_length, mask_token_id=mask_id
        )

    def __repr__(self) -> str:
        return (
            f"CacheHandle(method={self.method!r}, adapter={self.adapter.name!r}, "
            f"settings={self.settings!r})"
        )
