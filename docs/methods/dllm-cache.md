# dLLM-Cache

Liu et al. use different refresh schedules for the prompt and the generated response. Between full response updates, the method can refresh the positions whose value features have changed the most. The LLaDA implementation also reuses attention and MLP outputs, which matters when interpreting its memory use and speed. [Paper](https://proceedings.mlr.press/v306/liu26fi.html) · [LLaDA implementation](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/dllm_cache/hooks/cache_hook_LLaDA.py)

## Use the upstream demos

| Checkpoint in the authors’ example | How the method attaches | Entry point |
| --- | --- | --- |
| LLaDA-8B-Instruct | Registers cache hooks on `model.transformer.blocks` | [LLaDA demo](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_LLaDA.py) |
| Dream-v0-Instruct-7B | Registers cache hooks on `model.layers` | [Dream demo](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_Dream.py) |

Both demos set the prompt refresh interval to **100 steps**, the response refresh interval to **7 steps**, and the transfer ratio to **0.25**. These are the authors’ demo settings. When changing the ratio, hold the two intervals fixed so you can attribute any change to the fraction of response positions updated. [LLaDA demo settings](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_LLaDA.py#L10-L43) · [Configuration fields](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/dllm_cache/cache/Config.py)

The authors provide installation instructions and separate evaluation recipes. Their repository carries an Apache-2.0 license. [Setup and evaluation](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/README.md#L25-L62) · [Upstream license](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/LICENSE) · [All methods](../../README.md#method-zoo)
