# Block diffusion caches

A block diffusion model generates one block at a time and limits attention so that a block sees itself and earlier blocks. Once a block is final, later steps leave it unchanged, so its keys and values can be stored once and reused. Approximation only enters inside the active block. This differs from the bidirectional models on the other method pages, where every reuse is an approximation. [Block Diffusion paper](https://arxiv.org/abs/2503.09573) · [SDAR paper](https://arxiv.org/abs/2510.06303)

## Block cache

SDAR's generation script writes the prompt blocks into the cache first. Each denoising step then runs only the active block against the cache. When a block has no masks left, one more pass stores its keys and values. Research dLLM implements the same idea as `BlockdCache`, which its documentation calls an exact KV cache. It stores states by absolute position and refreshes a finished block together with the next block, so it needs no separate storing pass. dInfer's LLaDA2 recipes select its prefix cache together with block diffusion decoding. [SDAR generation loop](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/generate.py#L58-L202) · [Research dLLM `BlockdCache`](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L136-L159) · [dInfer LLaDA2 recipes](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md#L116-L150)

In the Zoo, the block cache is the baseline for this family. Check it against a run with no cache on the same checkpoint before measuring anything else, and record the tolerance used. [Comparison protocol](../benchmarking.md#pair-the-runs)

## Fast-dLLM v2 sub-block cache

Fast-dLLM v2 is a block diffusion model adapted from an autoregressive model, and its authors describe the cache as hierarchical. The block-level cache holds completed blocks. The sub-block cache splits the active block into smaller blocks. With `use_block_cache` set, a pass over the whole active block stores its keys and values, and later steps send only the current sub-block through the model. This second level is an approximation inside the active block. [Paper](https://arxiv.org/abs/2509.26328) · [Cache description](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/README.md#L24-L26) · [Sampling loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/generation_functions.py#L17-L168)

## Select a variant upstream

| Checkpoint in the example | Cache | Settings in the example | Entry point |
| --- | --- | --- | --- |
| `JetLM/SDAR-1.7B-Chat` | Block cache | Script defaults of block length 4 and four denoising steps | [SDAR inference script](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/README.md#L59-L65) |
| SDAR-8B-Chat, as `sdar-8b-chat` | Research dLLM `BlockdCache` | `cache=blockd` with block length 32 | [Research dLLM example](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L144-L159) |
| `Efficient-Large-Model/Fast_dLLM_v2_7B` | Block-level and sub-block cache | `use_block_cache` with a `small_block_size` | [Fast-dLLM v2 sampling loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/generation_functions.py#L17-L168) |
| `inclusionAI/LLaDA2.0-mini` and `inclusionAI/LLaDA2.0-flash` | dInfer prefix cache | `--cache prefix` and `--use_bd`, block length 32, threshold 0.9, SGLang backend | [dInfer LLaDA2 recipes](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md#L116-L150) |

SDAR's README also points to JetEngine and LMDeploy for batched inference. The LLaDA2 README describes an engine built on dInfer and SGLang with KV cache reuse. Treat those engines as separate hosts from the reference script when comparing timings. [SDAR inference options](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/README.md#L54-L127) · [LLaDA2 deployment note](https://github.com/inclusionAI/LLaDA2.X/blob/a79832985307bb8ac4a43ceeea54168bfb80f0cf/README.md#L61-L63)

The SDAR repository carries an MIT license. Fast-dLLM, Research dLLM and dInfer carry Apache-2.0 licenses. [SDAR license](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/LICENSE) · [Fast-dLLM license](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/LICENSE) · [All methods](../../README.md#method-zoo)
