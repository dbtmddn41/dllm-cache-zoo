"""dLLM Cache Zoo: KV cache methods for diffusion language models, selectable by name.

import dllm_cache_zoo as zoo

cache = zoo.attach(model, method="fast-dllm-dual")
with cache.request(prompt_length=prompt_length) as state:
    logits = state.forward(x, active=(block_start, block_end))
"""

from . import adapters
from . import methods as _methods
from .adapters.base import ModelAdapter
from .kv import KVStore
from .methods.base import CachePolicy, Computed
from .registry import (
    MethodInfo,
    attach,
    find_adapter,
    methods,
    register_adapter,
    register_method,
)
from .request import CacheHandle, Request, Stats

for _adapter in reversed(adapters.BUILTIN):
    register_adapter(_adapter)
for _method in _methods.BUILTIN:
    register_method(_method)

__version__ = "0.1.0.dev0"

__all__ = [
    "CacheHandle",
    "CachePolicy",
    "Computed",
    "KVStore",
    "MethodInfo",
    "ModelAdapter",
    "Request",
    "Stats",
    "attach",
    "find_adapter",
    "methods",
    "register_adapter",
    "register_method",
]
