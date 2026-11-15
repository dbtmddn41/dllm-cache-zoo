"""Looking up cache methods and model adapters by name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .adapters.base import ModelAdapter
from .methods.base import CachePolicy
from .request import CacheHandle

_METHODS: dict[str, type[CachePolicy]] = {}
_ADAPTERS: list[type[ModelAdapter]] = []


def register_method(cls: type[CachePolicy]) -> type[CachePolicy]:
    """Add a cache method to the Zoo. Usable as a class decorator."""
    name = cls.name
    if name in _METHODS and _METHODS[name] is not cls:
        raise ValueError(f"A cache method named {name!r} is already registered.")
    _METHODS[name] = cls
    return cls


def register_adapter(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Add a model adapter. Adapters registered later are tried first."""
    if cls not in _ADAPTERS:
        _ADAPTERS.insert(0, cls)
    return cls


@dataclass(frozen=True)
class MethodInfo:
    name: str
    families: tuple[str, ...]
    summary: str
    paper: str
    reference: str


def methods(family: str | None = None) -> list[MethodInfo]:
    """The cache methods in the Zoo, optionally only those for one model family."""
    infos = [
        MethodInfo(cls.name, cls.families, cls.summary, cls.paper, cls.reference)
        for cls in _METHODS.values()
    ]
    if family is not None:
        infos = [info for info in infos if family in info.families]
    return infos


def find_adapter(model: torch.nn.Module) -> ModelAdapter:
    """The adapter for ``model``, chosen from its module layout."""
    for cls in _ADAPTERS:
        if cls.matches(model):
            return cls(model)  # type: ignore[call-arg]
    known = ", ".join(cls.name for cls in _ADAPTERS)
    raise TypeError(
        f"No adapter recognises {type(model).__name__}. Supported model layouts: {known}. "
        "Pass `adapter=` to `attach` to use your own."
    )


def attach(
    model: torch.nn.Module,
    method: str = "none",
    *,
    adapter: ModelAdapter | None = None,
    **settings: Any,
) -> CacheHandle:
    """Attach a cache method to a model that is already loaded.

    The model is left untouched: calling it directly still runs its own forward.
    ``settings`` are the method's keyword arguments, for example
    ``attach(model, "dkv-cache-decode", refresh_interval=4)``.
    """
    try:
        cls = _METHODS[method]
    except KeyError:
        known = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown cache method {method!r}. Available: {known}.") from None
    if adapter is None:
        adapter = find_adapter(model)
    if adapter.family not in cls.families:
        fitting = ", ".join(info.name for info in methods(adapter.family))
        raise ValueError(
            f"{method!r} is defined for {' and '.join(cls.families)} models, and "
            f"{type(model).__name__} is {adapter.family}. Methods for this model: {fitting}."
        )
    policy = cls(adapter, **settings)
    return CacheHandle(
        model=model, adapter=adapter, policy=policy, method=method, settings=policy.config()
    )
