# dLLM Cache Zoo

A collection of **KV cache methods for diffusion language models**. The goal is to select a method by name inside the generation loop you already use. A model- and runtime-specific adapter will handle the cache updates while your sampler chooses which tokens to reveal.

**The collection is in development.** The examples below run in the authors’ repositories. The first proposed serving target is Fast-dLLM Prefix on LLaDA-8B-Instruct, checked against a PyTorch reference and evaluated under concurrent HTTP traffic. [Integration design](docs/project.md) · [Serving validation plan](docs/serving.md)

## Method zoo

| Method | What it reuses between denoising steps | LLaDA example | Dream example |
| --- | --- | --- | --- |
| [Fast-dLLM v1 · Prefix](docs/methods/fast-dllm.md#prefix-cache) | Prompt and completed blocks before the active block | [Chat](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) | [Evaluation recipe](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L38-L47) |
| [Fast-dLLM v1 · Dual](docs/methods/fast-dllm.md#dual-cache) | Both sides of the active block, including the masked suffix | [Chat](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) | [Parallel decoding recipe](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/eval.md#L65-L75) |
| [dKV-Cache · Decode](docs/methods/dkv-cache.md#decode) | Previously decoded positions after a one-step delay | [Generation example](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L87-L128) | [Generation example](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L26-L75) |
| [dKV-Cache · Greedy](docs/methods/dkv-cache.md#greedy) | Positions outside a small update set chosen from the decoding order | [Generation example](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L130-L135) | — |
| [dLLM-Cache](docs/methods/dllm-cache.md) | Prompt and response states on separate schedules, with selective response updates | [Demo](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_LLaDA.py) | [Demo](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/demo_Dream.py) |

Each model link opens an example at a fixed upstream revision. A dash means we have yet to identify an example for that combination. For the exact model IDs, [browse by checkpoint](docs/methods.md).

## Try a method with your model

For **LLaDA-8B-Instruct**, the Fast-dLLM chat example already offers a useful three-way switch. The default runs full recomputation. In the same script, `--use_cache` selects prefix caching, and adding `--if_cache_position` selects dual caching. Use the same block size and decoding threshold across the three runs. The method page links the authors’ setup instructions and explains the corresponding Dream settings. [Fast-dLLM method page](docs/methods/fast-dllm.md)

Developing a cache method? The contribution guide describes how a method and its model configuration fit into the collection. [Contribute a method](CONTRIBUTING.md) · [Library design](docs/project.md) · [Comparison protocol](docs/benchmarking.md)

The Zoo's own materials are licensed under Apache-2.0. Each method page credits its authors and records the upstream license. [License](LICENSE) · [Cite the Zoo](CITATION.cff)
