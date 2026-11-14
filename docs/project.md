# Project scope and roadmap

## The researcher's task

A team already evaluates a masked diffusion language model on its own prompts. It wants to try a cache policy, preserve its sampler and scoring code, and learn whether saved computation is worth any change in answer quality. The intended library will let that team select a policy in Python and export enough information to reproduce the comparison. A paper author should also be able to contribute one policy and reuse the same evaluation path.

The initial scope is inference-time reuse of attention keys and values across denoising steps in text diffusion models. Methods that also reuse other activations should identify them. The first model target is LLaDA-8B-Instruct. Dream is a candidate for testing which parts of the interface transfer. Training, multimodal generation, autoregressive serving, and a hosted leaderboard are outside the first milestone. [LLaDA model card](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) · [First reference policy](methods.md#fast-dllm-v1-prefix)

## Where the integration boundaries belong

| Proposed component | Responsibility | An example decision it owns |
| --- | --- | --- |
| Model adapter | Run the backbone, maintain position and attention-mask semantics, and reset request-local state | Map a refresh of positions 8–11 to the correct key and value slots |
| Cache policy | Choose what to retain or refresh for this request and step | Refresh every position when entering a new block |
| Sampler | Choose tokens and positions to unmask and decide when to stop | Reveal a fixed number of high-confidence tokens |
| Evaluation runner | Pair inputs and seeds, time both arms, score outputs, and export the manifest | Compute accuracy and latency from the same list of example IDs |

A team using only the adapter and a policy should be able to keep its own runner. The first baseline and cache policy will exercise these boundaries before the public API is settled.

## How to read a support claim

| Status | What a reader can verify |
| --- | --- |
| Reference | A description, paper, and specific upstream code revision appear in the catalog. |
| Integrated | The Zoo has a runnable example for a named model revision, configuration, and a correctness check. |
| Benchmarked | A paired result links the configuration, per-example records, scoring rule, and environment. |

Integration and benchmark status apply to a **policy and model pair**. An upstream Dream example alone establishes a reference for Dream. Each measured claim should identify the pair it covers.

## First milestones

| Order | Deliverable | Completion evidence |
| --- | --- | --- |
| 1 | LLaDA full-recomputation reference | A pinned model and tokenizer, an offline or small fixture for routine checks, and a deterministic run matching the upstream baseline on named prompts. |
| 2 | Fast-dLLM v1 prefix on the same LLaDA path | A research script can select either policy. Tests cover block boundaries, successive requests, and a configuration that refreshes all states. An upstream comparison records any intentional differences. |
| 3 | First public paired comparison | A GSM8K pilot with fixed example IDs, one configuration, raw records, and a table of quality, latency, throughput, and peak memory. A fresh environment can regenerate the table. |

After the first comparison, candidate additions are dual caching on LLaDA and the prefix policy on Dream. The first isolates a different reuse region. The second tests portability. A downstream use case or reproduction report can guide the order. [Benchmark protocol](benchmarking.md) · [Maintainer workflow](maintaining.md)
