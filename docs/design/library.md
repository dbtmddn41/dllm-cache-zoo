# Library design

The planned library separates **which positions to recompute** from **how the model recomputes them**. A cache method is a policy that names the positions to refresh at a denoising step. A model adapter runs the model on those positions and serves the rest from stored states. The host keeps its own generation loop: it loads the checkpoint, selects tokens, streams output if needed, and decides when to stop. A benchmark uses the same integration as a downstream application. Until the first milestone lands, the method pages link to the authors’ implementations. [Roadmap](roadmap.md) · [All design notes](README.md)

## Who it is for

The Zoo is for anyone who runs a diffusion language model and wants to reuse computation between denoising steps. Published implementations each carry their own copy of the model code. Fast-dLLM, Elastic-Cache and Flash-dLLM all ship a `modeling_llada.py`, and dLLM's Fast-dLLM pipeline adds another. Adopting a method today therefore means adopting its fork of the model and its sampler. The Zoo aims to offer each method as a named entry that attaches to the model you already load, the way a model zoo offers checkpoints by name.

| Use | What the Zoo provides |
| --- | --- |
| Speed up generation in your own script or application | A method selected by name, the checkpoints it supports, and a switch back to the baseline. |
| Compare methods or develop a new one | Every method on the same adapter, and a protocol for matched measurements. A new method is one policy. |
| Add a method to an inference engine | The policy as a checked reference for the engine's own implementation, and criteria for deciding whether to enable it. |

[Fast-dLLM](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/model/modeling_llada.py) · [Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/model/modeling_llada.py) · [Flash-dLLM](https://github.com/VILA-Lab/Flash-dLLM/blob/a667345f5746bb74f060c87bae5997fc3de9710c/llada/model/modeling_llada.py) · [dLLM Fast-dLLM pipeline](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/models/modeling_llada.py) · [Comparison protocol](benchmarking.md) · [Serving integration](serving.md)

## Policies, adapters and hosts

| Component | Responsibility |
| --- | --- |
| Cache policy | Defines the reusable rule and its configuration. At each step it names the positions to recompute and decides when to rebuild the cache. It declares the signals it reads and the kind of state it stores. A policy holds no model code. |
| Model adapter | Runs the model on the positions the policy names and serves the others from stored states. It preserves masks and position IDs, reads and writes the runtime's cache layout, supplies the signals a policy asks for, and owns cache state for the lifetime of each request. |
| Host generation loop | Supplies step and block information and retains ownership of token selection, sampling, batching, and stopping. A custom loop can call the adapter directly. |

## How published methods fit

| Method | Recomputed between refreshes | Refresh | Signal the policy reads | Stored state |
| --- | --- | --- | --- | --- |
| [Fast-dLLM Prefix](../methods/fast-dllm.md#prefix-cache) | Active block and suffix | At each new block | Block bounds | Keys and values |
| [Fast-dLLM Dual](../methods/fast-dllm.md#dual-cache) | Active block | At each new block | Block bounds | Keys and values, written by position |
| [dKV-Cache Decode](../methods/dkv-cache.md#decode) | Masked positions and those decoded on the previous step | Fixed interval | Mask state | Keys and values |
| [dKV-Cache Greedy](../methods/dkv-cache.md#greedy) | Positions just decoded, due next, and optionally nearby masks | Fixed interval | Unmasking order sampled up front | Keys and values |
| [dLLM-Cache](../methods/dllm-cache.md) | Response positions whose value features changed most | Separate fixed intervals for prompt and response | Value-feature similarity | Layer features, including attention and MLP outputs |
| [d²Cache](../methods/d2cache.md) | Masked positions picked by certainty, plus positions picked by attention rollout | Chosen at every step | Attention weights | Keys and values |
| [Elastic-Cache](../methods/elastic-cache.md) | Sliding window of masked positions | When attention on the most-attended tokens drifts past a threshold, starting from deeper layers | Attention weights | Keys and values |
| [Block cache](../methods/block-diffusion.md#block-cache) | Active block | When a block completes | Block bounds | Keys and values of completed blocks |
| [Fast-dLLM v2 Sub-block](../methods/block-diffusion.md#fast-dllm-v2-sub-block-cache) | Active sub-block | While the sub-block's first position is still masked | Sub-block bounds | Keys and values for completed blocks and for the active block |

This table is the working hypothesis for the interface. It is drawn from the papers and pinned code, and the first milestone tests it by running the first five rows on one adapter. Four points need a decision during that work:

- **Stored state varies.** Most methods store keys and values. dLLM-Cache also reuses attention and MLP outputs, so the adapter needs a second storage point per layer.
- **Signals have a cost.** A policy that reads attention weights can rule out a fused attention kernel. Research dLLM requires eager attention for d²Cache. A policy declares its signals, and an adapter rejects a combination it cannot serve. [d²Cache attention requirement](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L133-L134)
- **Some methods bundle a decoding rule.** Fast-dLLM adds confidence-aware parallel decoding, d²Cache adds certainty-prior decoding, and Elastic-Cache decodes inside a sliding window. The Zoo implements the cache. The decoding rule stays in the host, and each example and result names the one used. [Decoding conditions](benchmarking.md#hold-decoding-fixed)
- **Some methods choose the context.** The table covers nine variants, and the catalog lists many more. Eviction and sparse attention methods decide which stored entries a query reads. Suffix pruning methods decide which masked positions enter the forward pass. A policy for those methods names positions to keep and positions to drop, beside positions to recompute. Design the interface so this can be added without changing the policies of the first two milestones. [Method catalog](../catalog.md)

Research dLLM already structures its caches as hooks around the model forward, attention and feed-forward calls, with callbacks at step and block boundaries. Keep the Zoo's policy interface close enough that a method written for one can be ported to the other. [Research dLLM cache base class](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/src/cache/base.py#L165-L394)

## What changes in a LLaDA script

LLaDA's example loads its checkpoint with Transformers, then calls a standalone PyTorch generation function. Inside that function, each denoising step knows the active block and mask before it calls the model. This is a concrete place to integrate a cache. [Checkpoint loading and generation call](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L233-L261) · [Denoising step](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L138-L159)

In the proposed integration, the application attaches a method to its model once. The loop opens cache state for the request and passes the block bounds and token positions at the model call. The adapter returns logits aligned to the positions the host sampler expects. The sampler chooses which tokens to reveal. Switching between compatible methods changes one setting after this initial integration.

```python
import dllm_cache_zoo as zoo  # working name

cache = zoo.attach(model, method="fast-dllm-dual")  # adapter chosen from the model class

with cache.request(prompt_length=prompt_length) as state:
    for block in blocks:
        for step in range(steps_per_block):
            logits = state.forward(x, active=block)  # recomputes what the policy asks for
            x = reveal(x, logits)                    # host sampler, unchanged
            state.observe(x)                         # the policy sees which positions changed
```

This sketch shows the intended shape. Names and signatures are fixed once three methods run through it.

For **Fast-dLLM Dual**, the first step of a block runs on the whole sequence. Subsequent steps send only the active block through the model and replace the corresponding positions in each layer's cache. Transformers exposes KV storage through `past_key_values` on compatible models. To reproduce Dual, the adapter also needs the active block bounds, a shorter forward pass, and indexed KV writes. [Fast-dLLM generation loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L232-L272) · [Attention update](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/model/modeling_llada.py#L730-L754) · [Transformers cache API](https://huggingface.co/docs/transformers/en/kv_cache)

## Hosts

The Zoo ships no models and no samplers. The first adapters target the model classes in the dLLM library, which loads LLaDA, Dream and LLaDA2 behind one sampler interface. Its Fast-dLLM pipeline shows the cost an adapter is meant to remove: a separate modeling file for each model, and a sampler whose `use_cache` setting selects prefix or dual. Keep the dLLM dependency inside the adapter. A host that loads the same checkpoint through Transformers should be able to use the same policies. [Sampler interface](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/base.py#L21-L51) · [Fast-dLLM sampler settings](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L156-L177) · [LLaDA2 sampler](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/sampler.py#L95-L230)

An adapter may use a native hook when it exposes the required information in time. For example, Dream's published `diffusion_generate` loop calls the model **before** its logits hook. Skipping computation on that step requires an integration at or before the forward call. [Dream sampling loop](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/05334cb9faaf763692dcf9d8737c642be2b2a6ae/generation_utils.py#L408-L415)
