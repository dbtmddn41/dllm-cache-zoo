"""Selecting and writing sequence positions.

A set of positions is either a ``slice`` over the sequence axis (a contiguous
range, which costs no copy to read) or a ``LongTensor`` of shape ``(B, N)`` or
``(1, N)`` holding absolute positions for each row. ``None`` means every position.
"""

from __future__ import annotations

import torch

Positions = slice | torch.Tensor | None


def as_index(pos: Positions, length: int, device: torch.device) -> torch.Tensor:
    """Absolute positions as a ``(1 or B, N)`` tensor."""
    if pos is None:
        return torch.arange(length, device=device).unsqueeze(0)
    if isinstance(pos, slice):
        start, stop, _ = pos.indices(length)
        return torch.arange(start, stop, device=device).unsqueeze(0)
    return pos


def count(pos: Positions, length: int) -> int:
    if pos is None:
        return length
    if isinstance(pos, slice):
        start, stop, _ = pos.indices(length)
        return max(stop - start, 0)
    return int(pos.size(-1))


def _expand_index(idx: torch.Tensor, like: torch.Tensor, dim: int) -> torch.Tensor:
    """Reshape a ``(B, N)`` index so it can drive gather/scatter along ``dim`` of ``like``."""
    shape = [1] * like.dim()
    shape[0] = idx.size(0)
    shape[dim] = idx.size(1)
    view = idx.view(shape)
    target = list(like.shape)
    target[dim] = idx.size(1)
    return view.expand(target)


def take(t: torch.Tensor, pos: Positions, dim: int = 1) -> torch.Tensor:
    """Read positions along the sequence axis ``dim``. Axis 0 is the batch."""
    if pos is None:
        return t
    if isinstance(pos, slice):
        index = [slice(None)] * t.dim()
        index[dim] = pos
        return t[tuple(index)]
    if pos.size(0) == 1:
        return t.index_select(dim, pos[0])
    return t.gather(dim, _expand_index(pos, t, dim))


def put_(t: torch.Tensor, pos: Positions, src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Write ``src`` at positions along ``dim`` of ``t``, in place."""
    if pos is None:
        t.copy_(src)
        return t
    if isinstance(pos, slice):
        index = [slice(None)] * t.dim()
        index[dim] = pos
        t[tuple(index)] = src
        return t
    if pos.size(0) == 1:
        t.index_copy_(dim, pos[0], src)
        return t
    t.scatter_(dim, _expand_index(pos, src, dim), src)
    return t


def mask_to_index(
    mask: torch.Tensor, scores: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a boolean ``(B, L)`` mask into a sorted ``(B, N)`` index.

    Rows can select different numbers of positions, while one forward pass needs
    the same number in every row. Rows that select fewer than the largest row
    are topped up with unselected positions, highest ``scores`` first (lowest
    position first when ``scores`` is not given). Recomputing an extra position
    only makes the cache fresher. Returns the index and the mask after top-up.
    """
    counts = mask.sum(dim=-1)
    n = int(counts.max())
    if n == 0:
        return mask.new_zeros((mask.size(0), 0), dtype=torch.long), mask
    if bool((counts == n).all()):
        return mask.nonzero(as_tuple=True)[1].view(mask.size(0), n), mask
    length = mask.size(-1)
    if scores is None:
        scores = -torch.arange(length, device=mask.device, dtype=torch.float).expand_as(mask)
    order = torch.where(mask, torch.inf, scores.float()).argsort(
        dim=-1, descending=True, stable=True
    )
    index = order[:, :n].sort(dim=-1).values
    topped = torch.zeros_like(mask).scatter_(1, index, True)
    return index, topped
