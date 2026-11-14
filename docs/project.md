# Library design

The planned library exposes cache methods by name. A team plugs a method into the denoising loop it already uses. The host loads the checkpoint, selects tokens, streams output if needed, and decides when to stop. A benchmark can use the same integration as a downstream application. The current method pages link to the authors’ implementations.

## What changes in a LLaDA script

LLaDA's example loads its checkpoint with Transformers, then calls a standalone PyTorch generation function. Inside that function, each denoising step knows the active block and mask before it calls the model. This is a concrete place to integrate a cache. [Checkpoint loading and generation call](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L233-L261) · [Denoising step](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L138-L159)

In the proposed integration, the application selects a method for its LLaDA adapter. The loop opens cache state for the request and passes the block bounds and token positions to the adapter at the model call. The adapter returns logits aligned to the positions the host sampler expects. The sampler chooses which tokens to reveal. Switching between compatible methods changes one setting after this initial integration.

For **Fast-dLLM Dual**, the first step of a block runs on the whole sequence. Subsequent steps send only the active block through the model and replace the corresponding positions in each layer's cache. Transformers exposes KV storage through `past_key_values` on compatible models. To reproduce Dual, the adapter also needs the active block bounds, a shorter forward pass, and indexed KV writes. [Fast-dLLM generation loop](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L232-L272) · [Attention update](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/model/modeling_llada.py#L730-L754) · [Transformers cache API](https://huggingface.co/docs/transformers/en/kv_cache)

## What each integration provides

| Component | Responsibility |
| --- | --- |
| Cache method | Defines the reusable rule, its configuration, and the step or layer information needed to choose which states to refresh. A custom engine can call a reference implementation or implement the same decisions using its native kernels. |
| Model and runtime adapter | Resolves the method when configuring the worker and batches the required model operations. It preserves masks and position IDs, reads and writes the runtime's cache layout, and owns cache state for the lifetime of each request. |
| Host generation loop | Supplies step or layer information where the method needs it and retains ownership of token selection, sampling, batching, and stopping. A custom loop can call the adapter directly. |

An adapter may use a native hook when it exposes the required information in time. For example, Dream's published `diffusion_generate` loop calls the model **before** its logits hook. Skipping computation on that step requires an integration at or before the forward call. A serving runtime such as vLLM would need its own integration with the runner's per-request block tables and slot mappings. These are candidate integration points for future work. [Dream sampling loop](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/05334cb9faaf763692dcf9d8737c642be2b2a6ae/generation_utils.py#L408-L415) · [vLLM attention backend](https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backend/)

Advertise support for a **method, model revision, and runtime integration** together. Each supported combination should link the method's paper and implementation revision, applicable license notices, a minimal host-side example, a reusable configuration, and the checks run on that path. Report sampler restrictions before inference and link matched measurements when available. [Example of a sampler restriction](methods/dkv-cache.md#greedy)

## Proposed first release

Keep the LLaDA-8B-Instruct PyTorch loop as the correctness reference for full recomputation, Fast-dLLM Prefix, and Fast-dLLM Dual. Compare each cached variant to the authors’ corresponding implementation. For the first serving integration, take **Prefix** through one single-GPU BF16 HTTP worker. Its contiguous reusable region makes it a simpler first implementation. The first product milestone is a measured improvement on a declared workload with a working cache-off rollback. Dual is the next candidate once that path is established. [Upstream variants](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) · [Serving path and release criteria](serving.md)
