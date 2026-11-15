"""Per-request key/value storage, addressed by absolute position."""

from __future__ import annotations

import torch

from .positions import Positions, put_, take


class KVStore:
    """Keys and values for every layer, laid out as ``(B, H_kv, L, D)``.

    Entry ``[:, :, p]`` belongs to sequence position ``p``. Keys are stored with
    rotary embeddings applied, so they can be read back without the positions.
    """

    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers

    def clear(self) -> None:
        for i in range(len(self.keys)):
            self.keys[i] = None
            self.values[i] = None

    def has(self, layer: int) -> bool:
        return self.keys[layer] is not None

    def set(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Replace a layer's entries with tensors that cover the whole sequence."""
        self.keys[layer] = k
        self.values[layer] = v

    def allocate(self, layer: int, length: int, like_k: torch.Tensor, like_v: torch.Tensor) -> None:
        b, h, _, d = like_k.shape
        self.keys[layer] = like_k.new_zeros((b, h, length, d))
        self.values[layer] = like_v.new_zeros((b, like_v.size(1), length, like_v.size(-1)))

    def put(self, layer: int, pos: Positions, k: torch.Tensor, v: torch.Tensor) -> None:
        """Overwrite the entries at ``pos`` in place."""
        keys, values = self.keys[layer], self.values[layer]
        if keys is None or values is None:
            raise RuntimeError(f"Layer {layer} has no stored keys and values to update.")
        put_(keys, pos, k, dim=2)
        put_(values, pos, v, dim=2)

    def get(self, layer: int, pos: Positions = None) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self.keys[layer], self.values[layer]
        if keys is None or values is None:
            raise RuntimeError(f"Layer {layer} has no stored keys and values.")
        return take(keys, pos, dim=2), take(values, pos, dim=2)

    def grow(self, layer: int, length: int) -> None:
        """Extend a layer's storage with zeros up to ``length`` positions."""
        keys, values = self.keys[layer], self.values[layer]
        if keys is None or values is None:
            raise RuntimeError(f"Layer {layer} has no stored keys and values to extend.")
        extra = length - keys.size(2)
        if extra > 0:
            self.keys[layer] = torch.cat(
                [keys, keys.new_zeros((*keys.shape[:2], extra, keys.size(-1)))], dim=2
            )
            self.values[layer] = torch.cat(
                [values, values.new_zeros((*values.shape[:2], extra, values.size(-1)))], dim=2
            )

    def length(self, layer: int = 0) -> int:
        keys = self.keys[layer]
        return 0 if keys is None else int(keys.size(2))

    def nbytes(self) -> int:
        total = 0
        for t in (*self.keys, *self.values):
            if t is not None:
                total += t.numel() * t.element_size()
        return total
