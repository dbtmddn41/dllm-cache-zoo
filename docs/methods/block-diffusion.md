# Block diffusion caches

A block diffusion model generates one block at a time and limits attention so that a block sees itself and earlier blocks. Once a block is final, later steps leave it unchanged, so its keys and values can be stored once and reused. Approximation only enters inside the active block. This differs from the bidirectional models on the other method pages, where every reuse is an approximation. [Block Diffusion paper](https://arxiv.org/abs/2503.09573) · [SDAR paper](https://arxiv.org/abs/2510.06303)

## Block cache

SDAR's generation script writes the prompt blocks into the cache first. Each denoising step then runs only the active block against the cache. When a block has no masks left, one more pass stores its keys and values. Research dLLM implements the same idea as `BlockdCache`, which its documentation calls an exact KV cache. It stores states by absolute position and refreshes a finished block together with the next block, so it needs no separate storing pass. dInfer's LLaDA2 recipes select its prefix cache together with block diffusion decoding. [SDAR generation loop](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/generate.py#L58-L202) · [Research dLLM `BlockdCache`](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L136-L159) · [dInfer LLaDA2 recipes](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md#L116-L150)

In the Zoo, the block cache is the baseline for this family. It was checked against a run with no cache on the same checkpoint, and the tolerance is recorded in the check report. [Comparison protocol](../design/benchmarking.md#pair-the-runs)

### Use it from the Zoo

```python
cache = zoo.attach(model, "block-cache")
with cache.request(prompt_length=prompt_length, mask_token_id=mask_id) as state:
    logits = state.forward(x, active=(block_start, block_end), output="active")
```

Pass the block being decoded as `active`. Blocks are aligned to multiples of the block length from position 0, as in the SDAR script, so the last prompt tokens can share a block with the first generated ones. When the loop moves to a new block, one pass stores everything between the stored prefix and that block, which covers the prompt on the first call and the completed block afterwards. The store is preallocated and read as a view, so a step copies only the active block. `block_length` defaults to the length of the first span. SDAR's configuration does not name its mask token, so `request` takes it from the tokenizer. `method="none"` runs the same model without a cache, over the whole prefix with the block attention mask. [Policy](../../src/dllm_cache_zoo/methods/block_cache.py) · [SDAR adapter](../../src/dllm_cache_zoo/adapters/hf_decoder.py) · [Example](../../examples/sdar_generate.py)

The released SDAR model code imports flash-attn and runs its unmasked attention through it. The adapter uses PyTorch's attention kernel for every pass. On SDAR-1.7B-Chat in float32, the cached logits agree with the uncached pass and with the model's own masked forward to within 0.00012 on every checked prompt. In bfloat16 the three differ by rounding. The released model's bfloat16 forward is itself not reproducible across processes, so the comparison with the SDAR script's tokens varies with the run: greedy decoding matched on one or two of three prompts. The check report has the numbers. [Check report](../checks.md#block-cache-on-sdar)

## Fast-dLLM v2 sub-block cache

Fast-dLLM v2 is a block diffusion model adapted from an autoregressive model, and its authors describe the cache as hierarchical. The block-level cache holds completed blocks. The sub-block cache splits the active block into smaller blocks. With `use_block_cache` set, a pass over the whole active block stores its keys and values, and later steps send only the current sub-block through the model. This second level is an approximation inside the active block. [Paper](https://arxiv.org/abs/2509.26328) · [Cache description](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/README.md#L24-L26) · [Sampling loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/generation_functions.py#L17-L168)

### Use the sub-block cache from the Zoo

```python
from dllm_cache_zoo.loops import sub_block_generate

cache = zoo.attach(model, "fast-dllm-v2-sub-block")
out = sub_block_generate(cache, prompt_ids, max_new_tokens=256, block_length=32, sub_block_length=8, threshold=0.9)
```

In your own loop, pass the sub-block being refined as `active`. A step runs the whole block when the block has not been run yet or the first position of the sub-block is still masked, and the sub-block alone otherwise, which is the rule of the checkpoint's generation function. `block_length` defaults to the checkpoint's `bd_size`. `method="block-cache"` on the same model is the exact block level alone, and `method="none"` recomputes everything. [Policy](../../src/dllm_cache_zoo/methods/fast_dllm_v2.py) · [Reference loop](../../src/dllm_cache_zoo/loops.py) · [Example](../../examples/fast_dllm_v2_generate.py)

Fast-dLLM v2 keeps the next-token head of the autoregressive model it was adapted from: the hidden state at a position predicts the token after it. The Zoo returns logits aligned with the position they predict, so a sampler indexes them as for any other model. The first position of a block is predicted from the last position of the block before it. While that position is masked, a step returns its logits alone, and the loop reveals it before it refines the block. [Fast-dLLM v2 adapter](../../src/dllm_cache_zoo/adapters/hf_decoder.py)

On `Efficient-Large-Model/Fast_dLLM_v2_7B` in float32, the sub-block cache, the block cache and full recomputation each reveal the same tokens as the checkpoint's `generate` on every checked prompt, with one token per step and with a threshold of 0.9. In bfloat16, 9 of 15 runs give the same tokens and the others part at a near tie. The reference skips storing the prompt when the prompt is exactly one block long and then decodes without it. The Zoo stores it. On a model without a next-token head, such as SDAR, the policy runs the whole block on the first step of each sub-block. [Check report](../checks.md#fast-dllm-v2)

## Select a variant upstream

| Checkpoint in the example | Cache | Settings in the example | Entry point |
| --- | --- | --- | --- |
| `JetLM/SDAR-1.7B-Chat` | Block cache | Script defaults of block length 4 and four denoising steps | [SDAR inference script](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/README.md#L59-L65) |
| SDAR-8B-Chat, as `sdar-8b-chat` | Research dLLM `BlockdCache` | `cache=blockd` with block length 32 | [Research dLLM example](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L144-L159) |
| `Efficient-Large-Model/Fast_dLLM_v2_7B` | Block-level and sub-block cache | `use_block_cache` with a `small_block_size` | [Fast-dLLM v2 sampling loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/generation_functions.py#L17-L168) |
| `inclusionAI/LLaDA2.0-mini` and `inclusionAI/LLaDA2.0-flash` | dInfer prefix cache | `--cache prefix` and `--use_bd`, block length 32, threshold 0.9, SGLang backend | [dInfer LLaDA2 recipes](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md#L116-L150) |

SDAR's README also points to JetEngine and LMDeploy for batched inference. The LLaDA2 README describes an engine built on dInfer and SGLang with KV cache reuse. Treat those engines as separate hosts from the reference script when comparing timings. [SDAR inference options](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/README.md#L54-L127) · [LLaDA2 deployment note](https://github.com/inclusionAI/LLaDA2.X/blob/a79832985307bb8ac4a43ceeea54168bfb80f0cf/README.md#L61-L63)

The SDAR repository carries an MIT license. Fast-dLLM, Research dLLM and dInfer carry Apache-2.0 licenses. [SDAR license](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/LICENSE) · [Fast-dLLM license](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/LICENSE) · [All methods](../../README.md#method-zoo)
