# Browse by checkpoint

Start with the checkpoint in your pipeline. Each row lists methods with a linked example for those exact weights in an upstream repository. The two model families differ in what a cache can rely on, so they are listed separately. [Model families](../README.md#model-families)

## Bidirectional masked diffusion

| Checkpoint in the example | Methods with upstream examples |
| --- | --- |
| `GSAI-ML/LLaDA-8B-Instruct` | [Fast-dLLM v1 Prefix and Dual](methods/fast-dllm.md#select-a-variant-upstream) · [dKV-Cache Decode and Greedy](methods/dkv-cache.md#select-a-variant-upstream) · [dLLM-Cache](methods/dllm-cache.md#use-the-upstream-demos) · [d²Cache](methods/d2cache.md#select-the-method-upstream) |
| `GSAI-ML/LLaDA-1.5` | [Elastic-Cache](methods/elastic-cache.md#select-the-method-upstream) |
| `Dream-org/Dream-v0-Base-7B` | [Fast-dLLM v1 Prefix and Dual](methods/fast-dllm.md#select-a-variant-upstream) |
| `Dream-org/Dream-v0-Instruct-7B` | [dKV-Cache Decode](methods/dkv-cache.md#decode) · [dLLM-Cache](methods/dllm-cache.md#use-the-upstream-demos) · [Elastic-Cache](methods/elastic-cache.md#select-the-method-upstream) |

The Dream recipes listed here span **Base** and **Instruct** checkpoints. For a cache comparison, choose one checkpoint and run both methods on that revision with the same prompt format. [Fast-dLLM Dream recipe](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L18-L36) · [dKV-Cache Dream example](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L26-L42) · [dLLM-Cache Dream demo](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_Dream.py)

## Block diffusion

| Checkpoint in the example | Caches with upstream examples |
| --- | --- |
| `JetLM/SDAR-1.7B-Chat` | [Block cache in the SDAR generation script](methods/block-diffusion.md#block-cache) |
| SDAR-8B-Chat, as `sdar-8b-chat` in Research dLLM | [Block cache as `BlockdCache`](methods/block-diffusion.md#block-cache) |
| `Efficient-Large-Model/Fast_dLLM_v2_7B` | [Fast-dLLM v2 block-level and sub-block cache](methods/block-diffusion.md#fast-dllm-v2-sub-block-cache) |
| `inclusionAI/LLaDA2.0-mini` and `inclusionAI/LLaDA2.0-flash` | [Prefix cache with block diffusion decoding in dInfer](methods/block-diffusion.md#block-cache) |

SDAR publishes 1.7B, 4B, 8B and 30B checkpoints. LLaDA2.0-mini has 16B parameters and LLaDA2.0-flash has 100B, and the same repository now lists LLaDA2.1 and LLaDA2.2 models. Choose the checkpoint by the memory you have before choosing a method. [SDAR checkpoints](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/README.md#L255-L261) · [LLaDA2 model variants](https://github.com/inclusionAI/LLaDA2.X/blob/a79832985307bb8ac4a43ceeea54168bfb80f0cf/README.md#L23-L48)

## Methods without a page

Published methods that have no method page yet are listed in the catalog with their papers, code revisions and licenses. A method moves to the tables above once its page links an example for exact weights. [Method catalog](catalog.md)

[Browse all cache methods](../README.md#method-zoo)
