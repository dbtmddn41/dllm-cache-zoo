# Library design

The library separates **which positions to recompute** from **how the model recomputes them**. A cache method is a policy that names the positions to refresh at a denoising step and decides what to store. A model adapter runs the model's own modules on those positions, and the rest is read from stored states. The host keeps its own generation loop: it loads the checkpoint, selects tokens, streams output if needed, and decides when to stop. A benchmark uses the same integration as a downstream application. [Roadmap](roadmap.md) · [All design notes](README.md) · [Package source](../../src/dllm_cache_zoo/__init__.py)

## Who it is for

The Zoo is for anyone who runs a diffusion language model and wants to reuse computation between denoising steps. Published implementations each carry their own copy of the model code. Fast-dLLM, Elastic-Cache and Flash-dLLM all ship a `modeling_llada.py`, and dLLM's Fast-dLLM pipeline adds another. Adopting a method from its own repository therefore means adopting its fork of the model and its sampler. The Zoo offers each method as a named entry that attaches to the model you already load, the way a model zoo offers checkpoints by name.

| Use | What the Zoo provides |
| --- | --- |
| Speed up generation in your own script or application | A method selected by name, the checkpoints it supports, and a switch back to the baseline. |
| Compare methods or develop a new one | Every method on the same adapter, and a protocol for matched measurements. A new method is one policy. |
| Add a method to an inference engine | The policy as a checked reference for the engine's own implementation, and criteria for deciding whether to enable it. |

[Fast-dLLM](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/model/modeling_llada.py) · [Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/model/modeling_llada.py) · [Flash-dLLM](https://github.com/VILA-Lab/Flash-dLLM/blob/a667345f5746bb74f060c87bae5997fc3de9710c/llada/model/modeling_llada.py) · [dLLM Fast-dLLM pipeline](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/models/modeling_llada.py) · [Comparison protocol](benchmarking.md) · [Serving integration](serving.md)

## Policies, adapters and hosts

| Component | Responsibility |
| --- | --- |
| Cache policy | Defines the reusable rule and its configuration. At each step it names the positions to recompute and decides when to rebuild the cache. A policy holds no model code and no state of its own: everything that belongs to one generation lives on the request. [Policy base class](../../src/dllm_cache_zoo/methods/base.py) |
| Model adapter | Runs the model's modules for the positions the policy names: embedding, query, key and value projections with rotary embeddings at absolute positions, attention, MLP and head. It supplies attention weights when a policy asks for them. One adapter covers a model layout, and it is chosen from the modules of the loaded model. [Adapter interface](../../src/dllm_cache_zoo/adapters/base.py) |
| Host generation loop | Passes the current tokens and the active span at each step and retains ownership of token selection, sampling, batching, and stopping. The request it opens owns the cache state for one generation. [Request](../../src/dllm_cache_zoo/request.py) |

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
| [DPad](../methods/dpad.md) | What the cache underneath recomputes, minus the dropped suffix | At each new block, with a new draw of kept suffix positions | Block bounds | Nothing of its own |
| [Sparse-dLLM](../methods/sparse-dllm.md) | Active block | At each new block, after one or more full passes | Queries and keys | The kept share of keys and values outside the block, for each layer |

Every row runs from the Zoo. The published methods raised four design questions, and the implementation answers them as follows. [Method selection](method-selection.md) · [Check report](../checks.md)

- **Stored state varies.** Most methods store keys and values, which live in a per-request store addressed by absolute position. Keys are stored with rotary embeddings applied, so a stored key stays valid while its token does. dLLM-Cache also keeps each layer's attention and MLP outputs, and Elastic-Cache keeps each layer's input and queries. Such extra state sits in the request's private data, next to the store. [Key and value store](../../src/dllm_cache_zoo/kv.py) · [dLLM-Cache policy](../../src/dllm_cache_zoo/methods/dllm_cache.py)
- **Signals have a cost.** A policy that reads attention weights asks the adapter for them, and that pass runs eager attention in place of the fused kernel. d²Cache and Elastic-Cache do so. Sparse-dLLM scores entries from queries and keys and keeps the fused kernel. [d²Cache attention requirement](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L133-L134)
- **Some methods bundle a decoding rule.** Fast-dLLM adds confidence-aware parallel decoding, d²Cache adds certainty-prior decoding, and Elastic-Cache decodes inside a sliding window. The Zoo implements the cache. The decoding rule stays in the host, and each example and result names the one used. dKV-Cache Greedy is the one method that needs something from the sampler: the positions it is about to reveal, passed as `reveal`. [Decoding conditions](benchmarking.md#hold-decoding-fixed) · [Reference loops](../../src/dllm_cache_zoo/loops.py)
- **Some methods choose the context.** Suffix pruning decides which masked positions enter the forward pass, and eviction decides which stored entries a query reads. The shared forward pass takes the positions to run and, separately, the positions whose keys may be read. DPad passes the same kept set as both. Sparse-dLLM keeps a different set of entries in each layer, so it runs its own layer loop on the adapter. [Shared forward pass](../../src/dllm_cache_zoo/methods/base.py) · [Third integrations](method-selection.md#third-integrations)

Research dLLM structures its caches as hooks around the model forward, attention and feed-forward calls, with callbacks at step and block boundaries. The Zoo's policies make the same decisions at the same points, so a method written for one ports to the other: the d²Cache policy here is a port of theirs. [Research dLLM cache base class](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/src/cache/base.py#L165-L394)

## What changes in a LLaDA script

LLaDA's example loads its checkpoint with Transformers, then calls a standalone PyTorch generation function. Inside that function, each denoising step knows the active block and mask before it calls the model. This is where a cache goes in. [Checkpoint loading and generation call](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L233-L261) · [Denoising step](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L138-L159)

The application attaches a method to its model once. The loop opens cache state for the request and passes the tokens and the active span where it used to call the model. The sampler chooses which tokens to reveal, as before. Switching between methods changes one setting.

```python
import dllm_cache_zoo as zoo

cache = zoo.attach(model, method="fast-dllm-dual")  # adapter chosen from the model's layout

with cache.request(prompt_length=prompt_length) as state:
    for start, end in blocks:
        for step in range(steps_per_block):
            logits = state.forward(x, active=(start, end))  # recomputes what the policy asks for
            x = reveal(x, logits)                           # host sampler, unchanged
```

`active` is the span of positions the sampler may reveal at this step. A new span starts a new block, which is when Prefix and Dual rebuild their store. A loop without blocks leaves it out, and the whole response is one span. A sliding-window loop passes its window.

`state.forward` returns logits of shape `(B, L, V)`, so an existing sampler indexes them as before. They are valid at the masked positions of the active span, and positions the method did not run hold zeros. A sampler that only scores the active span should ask for `output="active"`, which returns `(B, end - start, V)` and skips building and scoring the rest. With a cache in place, a softmax over the whole sequence costs more than the model call it follows. [Example with the LLaDA loop](../../examples/own_loop.py)

The method reads which tokens were revealed from the next call, so the loop reports nothing back. `state.stats` counts forward passes and the share of positions that went through the layers, and `state.last_computed` marks the positions of the last pass. `cache.request` gives each generation its own state, so one attached method can serve several requests.

For **Fast-dLLM Dual**, the first step of a block runs on the whole sequence. Subsequent steps send only the active block through the layers and overwrite its entries in each layer's store, as the reference does with indexed writes. The stock LLaDA model has no cache path of its own: its forward asserts that no keys and values are passed. The adapter therefore runs the layer loop itself, from the model's own modules. [Fast-dLLM generation loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L232-L272) · [Attention update](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/model/modeling_llada.py#L730-L754) · [LLaDA adapter](../../src/dllm_cache_zoo/adapters/llada.py)

## Hosts

The Zoo ships no models. An adapter is chosen from the module layout of the loaded model, so one adapter serves every class with that layout. The LLaDA adapter reads `model.model.transformer.blocks` and drives the checkpoint's own Transformers code as well as copies of it, such as the model class in the dLLM library. The SDAR and Fast-dLLM v2 adapters read the Hugging Face decoder layout, `model.model.layers` with `self_attn` and `mlp`. Fast-dLLM v2 has a next-token head, and its adapter says so, which lets the block policies return logits aligned with the position they predict. A model the Zoo does not recognize can be attached with an adapter of your own. [Adapter interface](../../src/dllm_cache_zoo/adapters/base.py) · [dLLM model class](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/models/modeling_llada.py) · [SDAR adapter](../../src/dllm_cache_zoo/adapters/hf_decoder.py)

The Zoo's reference loops exist for the examples and the checks. They are the published samplers of LLaDA, Elastic-Cache and SDAR with the model call replaced, so every method can run under one decoding rule. A host with its own sampler does not need them. [Reference loops](../../src/dllm_cache_zoo/loops.py) · [Sampler interface of the dLLM library](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/base.py#L21-L51)

A host loop that cannot be edited needs a different entry point. Dream's published `diffusion_generate` loop calls the model **before** its logits hook, so skipping computation on that step requires an integration at or before the forward call. That is open work, together with the Dream adapter. [Dream sampling loop](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/05334cb9faaf763692dcf9d8737c642be2b2a6ae/generation_utils.py#L408-L415)

## Add a method

A method is one class. It subclasses `CachePolicy`, sets its registry name, the model families it is defined for, its paper and the revision its behavior follows, and implements `forward(req, tokens)`. Most methods call the shared `run` with the positions to recompute. `on_block_start` is called before the first pass of each active span. Per-generation state goes in `req.data` and keys and values in `req.kv`. [Policy base class](../../src/dllm_cache_zoo/methods/base.py) · [Fast-dLLM policies, the shortest example](../../src/dllm_cache_zoo/methods/fast_dllm.py)

```python
import dllm_cache_zoo as zoo
from dllm_cache_zoo import CachePolicy, Computed


@zoo.register_method
class EveryOtherStep(CachePolicy):
    name = "every-other-step"
    families = ("bidirectional",)
    summary = "Runs the whole sequence on even steps and the active span on odd ones."

    def forward(self, req, tokens):
        if req.block_step % 2 == 0:
            return Computed(self.run(req, tokens, None, store=True))
        span = slice(*req.active)
        return Computed(self.run(req, tokens, span, store=True), span)
```

A new method needs a test on the toy models and a comparison with its reference on a released checkpoint. [Contribution guide](../../CONTRIBUTING.md#add-a-method-to-the-library)
