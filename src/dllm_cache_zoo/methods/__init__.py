"""Cache methods."""

from .base import CachePolicy, Computed
from .block_cache import BlockCache
from .d2cache import D2Cache
from .dkv_cache import DKVCacheDecode, DKVCacheGreedy
from .dllm_cache import DLLMCache
from .dpad import DPad
from .elastic_cache import ElasticCache
from .fast_dllm import DualCache, PrefixCache
from .fast_dllm_v2 import SubBlockCache
from .none import NoCache
from .sparse_dllm import SparseDLLM

BUILTIN = (
    NoCache,
    PrefixCache,
    DualCache,
    DKVCacheDecode,
    DKVCacheGreedy,
    DLLMCache,
    D2Cache,
    ElasticCache,
    DPad,
    SparseDLLM,
    BlockCache,
    SubBlockCache,
)

__all__ = [
    "BUILTIN",
    "BlockCache",
    "CachePolicy",
    "Computed",
    "D2Cache",
    "DKVCacheDecode",
    "DKVCacheGreedy",
    "DLLMCache",
    "DPad",
    "DualCache",
    "ElasticCache",
    "NoCache",
    "PrefixCache",
    "SparseDLLM",
    "SubBlockCache",
]
