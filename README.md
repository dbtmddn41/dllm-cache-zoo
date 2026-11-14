# dLLM Cache Zoo

A collection of **KV cache methods for diffusion language models**, organized like a model zoo. Pick a method by name and turn it on for the model you already run. The planned library can be used to speed up generation in your own script, to compare methods under the same settings, or as a reference when adding a method to an inference engine.

In the planned library, a method is a small policy that decides which positions to recompute at each denoising step. A model adapter carries out that decision, and your sampler still chooses which tokens to reveal. Each entry comes with a method page, the checkpoints it supports, and results measured against a matched baseline.

**The Zoo is in the design stage.** The examples linked below run in the authors’ repositories, and no method runs from the Zoo yet. The first milestone runs full recomputation and four published cache variants through one interface on LLaDA-8B-Instruct, then checks each against its authors’ implementation. [Library design](docs/project.md) · [Roadmap](docs/project.md#roadmap) · [Comparison protocol](docs/benchmarking.md)

## Method zoo

| Method | What it reuses between denoising steps | Upstream examples | Zoo milestone |
| --- | --- | --- | --- |
| [Fast-dLLM v1 · Prefix](docs/methods/fast-dllm.md#prefix-cache) | Prompt and completed blocks before the active block | [LLaDA](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) · [Dream](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L38-L47) | 1 |
| [Fast-dLLM v1 · Dual](docs/methods/fast-dllm.md#dual-cache) | Both sides of the active block, including the masked suffix | [LLaDA](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) · [Dream](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L65-L75) | 1 |
| [dKV-Cache · Decode](docs/methods/dkv-cache.md#decode) | Previously decoded positions after a one-step delay | [LLaDA](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L87-L128) · [Dream](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L26-L75) | 1 |
| [dKV-Cache · Greedy](docs/methods/dkv-cache.md#greedy) | Positions outside a small update set chosen from the decoding order | [LLaDA](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L130-L135) | Later |
| [dLLM-Cache](docs/methods/dllm-cache.md) | Prompt and response states on separate schedules, with selective response updates | [LLaDA](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_LLaDA.py) · [Dream](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_Dream.py) | 1 |
| [d²Cache](docs/methods/d2cache.md) | Positions outside a per-step update set chosen by certainty and attention rollout | [LLaDA](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L94-L111) | 2 |
| [Elastic-Cache](docs/methods/elastic-cache.md) | Positions outside a sliding window, until attention on the most-attended tokens drifts | [LLaDA](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/eval_gsm8k.sh#L5-L19) · [Dream](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/dream/eval_gsm8k.sh#L5-L22) | 2 |
| [Block cache](docs/methods/block-diffusion.md#block-cache) | Completed blocks of a block diffusion model, stored once each block is final | [SDAR](https://github.com/JetAstra/SDAR/blob/4c2749ba103448f45520e8411533710a1e66574d/generate.py#L58-L202) · [LLaDA2 in dInfer](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md#L116-L150) | 2 |
| [Fast-dLLM v2 · Sub-block](docs/methods/block-diffusion.md#fast-dllm-v2-sub-block-cache) | Completed blocks, plus the rest of the active block while one sub-block is refined | [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v2/generation_functions.py#L17-L168) | Later |

Each example link opens a fixed upstream revision. The milestone column says when the Zoo plans to run that method itself. For the exact model IDs, [browse by checkpoint](docs/methods.md).

### More methods in the catalog

The table above holds the methods that already have a page. The catalog tracks 32 more published methods with their papers, code revisions and licenses. The aim is broad coverage, and the catalog is the list the Zoo works through. [Method catalog](docs/catalog.md)

| Group | Methods |
| --- | --- |
| [Selective recomputation on bidirectional models](docs/catalog.md#selective-recomputation-on-bidirectional-models) | Flash-dLLM · Dynamic-dLLM · FlashDLM · dInfer cache options · EntropyCache · SPA-Cache · DyLLM · ES-dLLM · Polestar · DARE · Window-Diffusion · Archer · LaCache |
| [Suffix pruning](docs/catalog.md#suffix-pruning) | DPad · Streaming-dLLM · Structured Suffix Modeling |
| [Eviction and sparse attention for long contexts](docs/catalog.md#eviction-and-sparse-attention-for-long-contexts) | Sparse-dLLM · MaskKV · SparseD · Focus-dLLM · Prefilling-dLLM · PulseCol · WaveFilter · GroupKV |
| [Block diffusion](docs/catalog.md#block-diffusion) | MAGE · LoSA · FlashBlock · HERALD · MaskAhead · FOCUS |
| [Caches shared across requests](docs/catalog.md#caches-shared-across-requests) | BiCache · Affix Cache |

## Model families

| Family | Checkpoints in the linked examples | What a cache can rely on |
| --- | --- | --- |
| Bidirectional masked diffusion | LLaDA-8B, LLaDA-1.5, Dream-v0 | Every position attends to every other position, so any reuse between steps is an approximation. |
| Block diffusion | SDAR, Fast-dLLM v2, LLaDA2.0 | A block attends to itself and earlier blocks. The SDAR and Fast-dLLM v2 examples store a completed block once and refine only the active block. |

## Try a method with your model

For **LLaDA-8B-Instruct**, the Fast-dLLM chat example already offers a useful three-way switch. The default runs full recomputation. In the same script, `--use_cache` selects prefix caching, and adding `--if_cache_position` selects dual caching. Use the same block size and decoding threshold across the three runs. The method page links the authors’ setup instructions and explains the corresponding Dream settings. [Fast-dLLM method page](docs/methods/fast-dllm.md)

## Related projects

| Project | What it offers | How the Zoo relates |
| --- | --- | --- |
| [Research dLLM](https://github.com/Kamichanw/d2Cache/tree/79fb5f64864389ba093c29122a21ef0391080312) | Config-selectable Prefix and Dual, dLLM-Cache, d²Cache and a block diffusion cache, with several decoding strategies and lm-eval tasks. Maintained by the d²Cache authors. | Closest in purpose. The Zoo packages only the caches, for use inside a generation loop you already have, and includes methods outside that codebase such as dKV-Cache and Elastic-Cache. |
| [dLLM](https://github.com/ZHZisZZ/dllm/tree/ca176752fbceec49c6b4777a2c18ae88e4eb10ed) | Training, inference and evaluation pipelines for LLaDA, Dream and LLaDA2, including a Fast-dLLM pipeline. | Planned first host for the Zoo's model adapters. |
| [dInfer](https://github.com/inclusionAI/dInfer/tree/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f) | A batched inference engine with Prefix and Dual cache options and vLLM and SGLang backends. | Candidate engine for running Zoo methods in serving. |

## Roadmap

1. Run full recomputation, Fast-dLLM Prefix and Dual, dKV-Cache Decode and dLLM-Cache on LLaDA-8B-Instruct through one policy interface.
2. Add a block diffusion checkpoint with its block cache, and bring d²Cache and Elastic-Cache to the same interface.
3. Publish an installable release with a support table, where each method and checkpoint pair has an example, a configuration and measured results. From here, methods move from the catalog into the Zoo one at a time.
4. Carry one method into an existing serving engine and validate it under load.

Developing a cache method? The contribution guide describes how a method and its model configuration fit into the collection. [Contribute a method](CONTRIBUTING.md) · [Library design](docs/project.md) · [Serving integration](docs/serving.md)

The Zoo's own materials are licensed under Apache-2.0. Each method page credits its authors and records the upstream license. [License](LICENSE) · [Cite the Zoo](CITATION.cff)
