"""Model adapters."""

from .base import ModelAdapter
from .hf_decoder import FastDLLMv2Adapter, HFDecoderAdapter, SDARAdapter
from .llada import LLaDAAdapter

BUILTIN = (LLaDAAdapter, SDARAdapter, FastDLLMv2Adapter)

__all__ = [
    "BUILTIN",
    "FastDLLMv2Adapter",
    "HFDecoderAdapter",
    "LLaDAAdapter",
    "ModelAdapter",
    "SDARAdapter",
]
