"""DPad suffix dropout (Chen et al., ICLR 2026).

Behaviour follows the authors' LLaDA loops and sampler:
https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/generate_dropout.py#L96-L367
https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/sampler.py

DPad stores nothing by itself. Before each block it keeps every position up to
the end of the block and a random subset of the masked suffix inside a window,
chosen with a probability that falls with distance. The rest of the suffix is
left out of the model until the next block: those positions are neither queries
nor keys. The kept positions keep their own rotary positions. It can run with no
cache, or on top of the Fast-dLLM prefix or dual cache.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from ..positions import Positions
from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


def gaussian_keep_probability(window: int, k: float, scale: float, device) -> torch.Tensor:
    """Probability of keeping the suffix position at each distance inside the window.

    The window spans ``k`` standard deviations of a normal curve centred on the
    end of the block, and ``scale`` multiplies the curve.
    """
    x = torch.linspace(0.0, float(k), window, device=device)
    return scale * torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class DPad(CachePolicy):
    """Drop distant masked suffix positions from every forward pass.

    ``cache`` selects what runs underneath: ``"none"``, ``"prefix"`` or
    ``"dual"``. ``window`` is the number of suffix positions that can be kept,
    ``k`` how fast the keep probability falls across the window and ``scale`` its
    height. ``sampler="uniform"`` keeps ``preserved`` positions of the window at
    random instead. The subset is drawn again for each block. Pass ``seed`` to
    make the draws reproducible for a request; otherwise the global generator of
    the model's device is used, as in the reference.
    """

    name = "dpad"
    summary = (
        "Leaves distant masked suffix positions out of each pass; runs with no cache or on "
        "top of the prefix or dual cache."
    )
    paper = "https://arxiv.org/abs/2508.14148"
    reference = "Crys-Chen/DPad@9612888 llada/generate_dropout.py (Apache-2.0)"

    def __init__(
        self,
        adapter,
        cache: str = "prefix",
        window: int = 256,
        k: float = 4.0,
        scale: float = 2.0,
        sampler: str = "gaussian",
        preserved: int = 0,
        seed: int | None = None,
    ):
        super().__init__(adapter)
        if cache not in ("none", "prefix", "dual"):
            raise ValueError("cache must be 'none', 'prefix' or 'dual'.")
        if sampler not in ("gaussian", "uniform"):
            raise ValueError("sampler must be 'gaussian' or 'uniform'.")
        self.cache = cache
        self.window = int(window)
        self.k = float(k)
        self.scale = float(scale)
        self.sampler = sampler
        self.preserved = int(preserved)
        self.seed = seed

    def config(self) -> dict[str, Any]:
        return {
            "cache": self.cache,
            "window": self.window,
            "k": self.k,
            "scale": self.scale,
            "sampler": self.sampler,
            "preserved": self.preserved,
            "seed": self.seed,
        }

    def start(self, req: Request) -> None:
        req.data["generator"] = None

    def _keep(self, req: Request, suffix: torch.Tensor) -> torch.Tensor:
        """The suffix positions that stay in the model for this block."""
        n = suffix.numel()
        device = suffix.device
        generator = req.data["generator"]
        if self.seed is not None and generator is None:
            generator = torch.Generator(device=device).manual_seed(self.seed)
            req.data["generator"] = generator
        if self.sampler == "uniform":
            if self.preserved >= n:
                return suffix
            limit = min(n, self.window)
            order = torch.randperm(limit, generator=generator, device=device)
            return suffix[order[: self.preserved].sort().values]
        prob = gaussian_keep_probability(self.window, self.k, self.scale, device)
        if n > self.window:
            prob = torch.cat([prob, prob.new_zeros(n - self.window)])
        uniform = torch.rand(n, generator=generator, device=device)
        return suffix[uniform < prob[:n]]

    def on_block_start(self, req: Request, tokens: torch.Tensor) -> None:
        end = req.active[1]
        length = tokens.size(1)
        device = tokens.device
        kept = self._keep(req, torch.arange(end, length, device=device))
        req.data["kept"] = kept
        req.data["visible"] = torch.cat([torch.arange(end, device=device), kept]).unsqueeze(0)

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        visible = req.data["visible"]
        if self.cache == "none":
            return Computed(self.run(req, tokens, visible, store=False), visible)
        if req.block_step == 0:
            logits = self.run(req, tokens, visible, store=True, key_pos=visible)
            return Computed(logits, visible)
        start, end = req.active
        pos: Positions = slice(start, end)
        if self.cache == "prefix":
            block = torch.arange(start, end, device=tokens.device)
            pos = torch.cat([block, req.data["kept"]]).unsqueeze(0)
        return Computed(self.run(req, tokens, pos, store=True, key_pos=visible), pos)
