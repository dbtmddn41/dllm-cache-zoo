"""d²Cache (Jiang et al., ICLR 2026).

Ported from the authors' implementation in Research dLLM, including its
selection utilities:
https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/src/cache/d2cache.py
https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/src/utils/common.py

After a first full pass, every step runs a small set of positions and reads the
rest from the store. The set has two parts. Masked positions are ranked by a
certainty prior (their confidence times the local density of decoded tokens)
and the best ``current_k`` are taken, together with the tokens revealed on the
previous step. The other positions are ranked by attention rollout, and the top
share ``rollout_p`` is taken. The method reads attention weights, so its passes
use eager attention.

The paper also proposes decoding guided by the same certainty prior. That is a
sampler choice and stays with the host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..positions import as_index, mask_to_index
from .base import CachePolicy, Computed

if TYPE_CHECKING:
    from ..request import Request


def certainty_density(decoded: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smoothed density of decoded tokens around each position, ``(B, L)``.

    Positions left of the sequence count as decoded. Positions right of it count
    as decoded only when the last position is.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    batch, length = decoded.shape
    device = decoded.device
    padded = F.pad(decoded.float(), (length, length), "constant", 1.0)
    padded[~decoded[:, -1], 2 * length :] = 0.0

    extended = 3 * length
    padded_len = 2 * extended
    dist = torch.cat(
        (torch.arange(extended, device=device), torch.arange(-extended, 0, device=device))
    )
    kernel_fft = torch.fft.fft(torch.exp(-(dist**2) / (2 * sigma**2)), n=padded_len)
    weighted = torch.fft.ifft(
        torch.fft.fft(F.pad(padded, (0, extended)), n=padded_len) * kernel_fft, n=padded_len
    ).real
    kernel_sum = torch.fft.ifft(
        torch.fft.fft(torch.ones(batch, extended * 2, device=device), n=padded_len) * kernel_fft,
        n=padded_len,
    ).real
    return weighted[..., length : 2 * length] / kernel_sum[..., length : 2 * length].clamp_min(1e-8)


def nucleus_select(
    scores: torch.Tensor, top_p: float, min_k: int = 1, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Select the highest-scoring positions whose normalised scores sum to at most ``top_p``.

    At least ``min_k`` positions are selected, and only where ``mask`` is True.
    """
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must lie in [0, 1], got {top_p}.")
    scores = torch.where(mask, scores, 0.0) if mask is not None else scores
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    nucleus = torch.cumsum(sorted_probs, dim=-1) <= top_p
    top_k = torch.arange(nucleus.shape[-1], device=nucleus.device) < min(min_k, scores.shape[-1])
    combined = nucleus | top_k
    if mask is not None:
        combined &= torch.gather(mask, 1, sorted_indices)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, sorted_indices, combined)


class D2Cache(CachePolicy):
    """Dual adaptive cache: certainty prior for masked positions, attention rollout for the rest.

    ``current_k`` is how many masked positions are updated each step, ``sigma``
    the width of the certainty density, and ``rollout_p`` the share of rollout
    mass that selects other positions. ``inflate_w`` also selects positions that
    lie in a gap of at most that width between two selected ones (0 turns it off).

    The reference revision never updates its stored confidence after the first
    pass (``a & b == c`` binds as ``(a & b) == c`` there, and the assignment goes
    to a copy). ``refresh_confidence=True``, the default here, stores the new
    confidence of the masked positions each step ran, as the paper describes.
    Set it to False to reproduce the reference revision.
    """

    name = "d2cache"
    summary = (
        "Updates masked positions picked by a certainty prior and other positions picked by "
        "attention rollout; reuses keys and values elsewhere."
    )
    paper = "https://arxiv.org/abs/2509.23094"
    reference = "Kamichanw/d2Cache@79fb5f6 src/cache/d2cache.py (Apache-2.0)"

    def __init__(
        self,
        adapter,
        rollout_p: float = 0.1,
        current_k: int = 32,
        sigma: float = 10.0,
        inflate_w: int = 0,
        refresh_confidence: bool = True,
    ):
        super().__init__(adapter)
        self.rollout_p = float(rollout_p)
        self.current_k = int(current_k)
        self.sigma = float(sigma)
        self.inflate_w = int(inflate_w)
        self.refresh_confidence = bool(refresh_confidence)

    def config(self) -> dict[str, Any]:
        return {
            "rollout_p": self.rollout_p,
            "current_k": self.current_k,
            "sigma": self.sigma,
            "inflate_w": self.inflate_w,
            "refresh_confidence": self.refresh_confidence,
        }

    def start(self, req: Request) -> None:
        req.data.update(q_mask=None, conf=None, rollout=None, last=None)

    # ---------------------------------------------------------------- forward

    def forward(self, req: Request, tokens: torch.Tensor) -> Computed:
        st = req.data
        if st["last"] is not None:
            self._select(req, tokens)

        if st["q_mask"] is None:
            pos = None
            computed = torch.ones_like(tokens, dtype=torch.bool)
        else:
            scores = None
            if tokens.size(0) > 1:
                # Rows that selected fewer positions are topped up: masked positions
                # by certainty first, then other positions by rollout importance.
                gen = tokens.size(1) - req.prompt_length
                scores = st["importance"].float().clone()
                scores[:, -gen:] += scores.max() + st["density"]
            pos, computed = mask_to_index(st["q_mask"], scores)

        index = as_index(pos, tokens.size(1), tokens.device)

        def rollout(layer: int, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor | None) -> None:
            assert w is not None
            self._accumulate(st, layer, w, index, full=pos is None)

        logits = self.run(req, tokens, pos, store=True, need_weights=True, on_attention=rollout)

        # Confidence of the positions this step ran, for the next selection. It is
        # taken in the dtype of the logits, as the reference sampler computes it.
        probs = F.softmax(logits, dim=-1).amax(dim=-1)
        confidence = torch.full(tokens.shape, -torch.inf, device=tokens.device, dtype=probs.dtype)
        confidence.scatter_(1, index.expand(tokens.size(0), -1), probs)
        st["last"] = {"active": req.active, "computed": computed, "confidence": confidence}
        return Computed(logits, pos)

    def _accumulate(
        self, st: dict[str, Any], layer: int, weights: torch.Tensor, index: torch.Tensor, full: bool
    ) -> None:
        """One layer of attention rollout, where only the rows in ``index`` attend."""
        attn = weights.mean(dim=1)  # (B, N_q, L)
        batch, _, length = attn.shape
        eye = torch.eye(length, device=attn.device, dtype=attn.dtype)
        rows = index.expand(batch, -1)
        # Add the residual connection and renormalise each row. Rows that did not
        # attend this step reduce to the identity and leave the rollout unchanged.
        residual = attn + eye[rows]
        residual = residual / residual.sum(dim=-1, keepdim=True)
        if layer == 0:
            st["rollout"] = eye.expand(batch, -1, -1).clone()
        updated = residual @ st["rollout"]
        if full:
            st["rollout"] = updated
        else:
            st["rollout"].scatter_(1, rows.unsqueeze(-1).expand(-1, -1, length), updated)

    # -------------------------------------------------------------- selection

    def _select(self, req: Request, tokens: torch.Tensor) -> None:
        """Choose the positions to run now from what the previous step produced."""
        st = req.data
        last = st["last"]
        prompt_len = req.prompt_length
        mask_id = req.mask_token_id
        assert req.prev_tokens is not None
        prev_gen = req.prev_tokens[:, prompt_len:]
        gen = tokens[:, prompt_len:]
        batch, gen_len = gen.shape
        length = tokens.size(1)
        device = tokens.device

        confidence = last["confidence"][:, prompt_len:]
        fresh = last["computed"][:, prompt_len:] & (prev_gen == mask_id)
        confidence = torch.where(fresh, confidence, -torch.inf)
        if st["conf"] is None:
            st["conf"] = confidence
        elif self.refresh_confidence:
            st["conf"] = torch.where(fresh, confidence, st["conf"])

        start, end = last["active"]
        block_mask = torch.zeros((batch, gen_len), dtype=torch.bool, device=device)
        block_mask[:, start - prompt_len : end - prompt_len] = True
        remaining = gen == mask_id

        # 1. Masked positions: search from the block start to the first block
        #    boundary after the ``current_k``-th remaining mask.
        block_size = block_mask.sum(dim=1, keepdim=True)
        meets = torch.cumsum(remaining.int(), dim=1) >= self.current_k
        min_end = torch.argmax(meets.int(), dim=1, keepdim=True)
        min_end[~meets.any(dim=1, keepdim=True)] = gen_len - 1
        search_end = (((min_end // block_size) + 1) * block_size) - 1
        block_start = torch.argmax(block_mask.int(), dim=1, keepdim=True)
        cols = torch.arange(gen_len, device=device)
        search = (cols >= block_start) & (cols <= search_end)

        scores = st["conf"] * certainty_density(~remaining, self.sigma)
        # Bias the block so that its masked positions are always among the selected.
        scores[block_mask] += scores.max()
        _, top = torch.topk(
            torch.where(search & remaining, scores, -torch.inf),
            k=min(self.current_k, gen_len),
            dim=-1,
        )
        response = torch.zeros_like(remaining).scatter_(1, top, True) & remaining

        # 2. Tokens revealed on the previous step go through the model once more.
        response |= (prev_gen == mask_id) & ~remaining
        q_mask = F.pad(response, (prompt_len, 0), value=False)

        # 3. Other positions: the ones the sequence attends to most, by rollout.
        importance = st["rollout"].sum(dim=1)
        q_mask |= nucleus_select(importance, self.rollout_p, mask=~q_mask)

        # 4. Fill short gaps between selected positions.
        if self.inflate_w > 0:
            arange = torch.arange(length, device=device).expand(batch, -1)
            nxt = torch.where(q_mask, arange, length)
            nxt = torch.flip(torch.cummin(torch.flip(nxt, dims=[-1]), dim=-1).values, dims=[-1])
            prv = torch.cummax(torch.where(q_mask, arange, -1), dim=-1).values
            gap = (nxt - arange) + (arange - prv)
            q_mask |= (gap <= self.inflate_w) & (prv >= 0) & (nxt < length)

        st["q_mask"] = q_mask
        st["importance"] = importance
        st["density"] = scores
