# Fast-dLLM v1

Fast-dLLM adds a cache to blockwise diffusion decoding. Each block starts with a full forward pass. Its two variants differ in which positions the model computes again during the remaining denoising steps of that block. Wu et al. also introduce confidence-aware parallel decoding, which can be used alongside the cache. [Paper](https://arxiv.org/abs/2505.22618) · [Authors’ setup guide](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/README.md#L73-L95)

## Prefix cache

Keep keys and values for the prompt and all completed blocks. During refinement, recompute the active block and the suffix. For example, when filling the second of three response blocks, the cache holds the prompt and block one. Blocks two and three still go through the model. [LLaDA implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L132-L207)

## Dual cache

Keep the prefix and the suffix from the block's first pass. In the same three-block example, only block two goes through the model on subsequent steps. A masked suffix token can change its hidden state when another position is revealed, so freezing the suffix can affect the final answer. Use multiple blocks when comparing the two variants so that at least one active block has a suffix. [LLaDA implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L210-L272)

## Select a variant upstream

| Checkpoint in the authors’ example | Prefix | Dual | Entry point |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct | Set `--use_cache` | Set both `--use_cache` and `--if_cache_position` | [Interactive chat](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L21-L75) |
| Dream-v0-Base-7B | Set the model argument `use_cache=true` | Also set `dual_cache=true` | [Evaluation recipes](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L18-L84) |

The LLaDA chat script defaults to full recomputation. Hold the checkpoint, block size, step budget, and token-transfer rule fixed when switching among the three settings. On Dream, compare the authors’ *Prefix Cache + Parallel* and *Dual Cache + Parallel* recipes to keep the confidence threshold and denoising budget matched. [Dream comparison settings](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L57-L72) · [Authors’ LLaDA results](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/README.md#L104-L116)

The linked LLaDA implementation carries an Apache-2.0 header. [Upstream license](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/LICENSE) · [All methods](../../README.md#method-zoo)
