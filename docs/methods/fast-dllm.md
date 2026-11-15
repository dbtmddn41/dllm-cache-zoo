# Fast-dLLM v1

Fast-dLLM adds a cache to blockwise diffusion decoding. Each block starts with a full forward pass. Its two variants differ in which positions the model computes again during the remaining denoising steps of that block. Wu et al. also introduce confidence-aware parallel decoding, which can be used alongside the cache. [Paper](https://arxiv.org/abs/2505.22618) · [Authors’ setup guide](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/README.md#L73-L95)

## Prefix cache

Keep keys and values for the prompt and all completed blocks. During refinement, recompute the active block and the suffix. For example, when filling the second of three response blocks, the cache holds the prompt and block one. Blocks two and three still go through the model. [LLaDA implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L132-L207)

## Dual cache

Keep the prefix and the suffix from the block's first pass. In the same three-block example, only block two goes through the model on subsequent steps. A masked suffix token can change its hidden state when another position is revealed, so freezing the suffix can affect the final answer. Use multiple blocks when comparing the two variants so that at least one active block has a suffix. [LLaDA implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L210-L272)

## Use it from the Zoo

```python
cache = zoo.attach(model, "fast-dllm-prefix")  # or "fast-dllm-dual"
with cache.request(prompt_length=prompt_length) as state:
    logits = state.forward(x, active=(block_start, block_end))
```

Neither variant has settings. Pass the block being decoded as `active`: the first call with a new span runs the whole sequence and rebuilds the store, and later calls run the block and suffix (Prefix) or the block alone (Dual). A loop with a single block caches only the prompt under Prefix. Confidence-aware parallel decoding is a sampler setting, `threshold` in the Zoo's reference loop. [Policies](../../src/dllm_cache_zoo/methods/fast_dllm.py) · [Reference loop](../../src/dllm_cache_zoo/loops.py)

On LLaDA-8B-Instruct both variants reveal the same tokens as the authors' loops on every checked prompt, with a fixed step budget and with a confidence threshold of 0.9. The store keeps keys with rotary embeddings applied, where the reference stores them without and applies the embeddings at every step. The values that reach the attention are the same. [Check report](../checks.md)

## Select a variant upstream

| Checkpoint in the authors’ example | Prefix | Dual | Entry point |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct | Set `--use_cache` | Set both `--use_cache` and `--if_cache_position` | [Interactive chat](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L21-L75) |
| Dream-v0-Base-7B | Set the model argument `use_cache=true` | Also set `dual_cache=true` | [Evaluation recipes](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L18-L84) |

The LLaDA chat script defaults to full recomputation. Hold the checkpoint, block size, step budget, and token-transfer rule fixed when switching among the three settings. On Dream, compare the authors’ *Prefix Cache + Parallel* and *Dual Cache + Parallel* recipes to keep the confidence threshold and denoising budget matched. [Dream comparison settings](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L57-L72) · [Authors’ LLaDA results](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/README.md#L104-L116)

Fast-dLLM v2 is a separate block diffusion model with its own cache. [Block diffusion caches](block-diffusion.md#fast-dllm-v2-sub-block-cache)

The linked LLaDA implementation carries an Apache-2.0 header. [Upstream license](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/LICENSE) · [All methods](../../README.md#method-zoo)
