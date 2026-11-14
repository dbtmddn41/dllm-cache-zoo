# Library design

The Zoo will ship implementations of named KV cache methods for diffusion language models. A team should be able to discover the methods available for its checkpoint, load one into its existing generation script, and switch to a second method by changing the selection. Today's directory points to the authors’ implementations. The behaviors below describe the intended library.

## What a researcher should be able to do

Suppose a team already runs prompts through LLaDA-8B-Instruct. It first selects full recomputation, then Fast-dLLM prefix caching, then dual caching. Across all three runs it keeps its model, prompt formatting, generation settings, and scoring code. The method selection determines which positions are recomputed. The team can save that choice and its parameters with its experiment, or import the same method into a different project.

| Operation | What the library should provide |
| --- | --- |
| Find a method | A list of available variants for the selected backbone and the model revisions on which each was checked. |
| Load a method | A named implementation and an example configuration with its cache-specific settings, such as block size or refresh interval. |
| Generate | A model adapter that applies the method during denoising, preserves compatible sampler settings, and starts each request with fresh cache state. |
| Switch or compare | The same generation entry point for the baseline and other compatible methods. If a method needs a particular sampler, expose that requirement before the run. |

For example, dKV-Cache-Greedy's LLaDA implementation requires a random unmasking order. That capability belongs in the method entry so a team using confidence-based unmasking can see which choices fit its pipeline. [Greedy method page](methods/dkv-cache.md#greedy)

## What ships with an integrated method

- One method name per selectable variant, with the authors’ paper, implementation revision, and applicable license notices.
- An implementation reachable through the shared entry point, plus the compatible backbone, checkpoint revision, and sampler requirements.
- A minimal example and a reusable configuration stating the method parameters. A team should be able to start with its own prompts.
- A correctness check for the supported model and a link to matched measurements when available. In the method table, label upstream examples, Zoo examples, and measured results by their origin.

## Proposed first release

Start with **one backbone and two switchable cache methods**. LLaDA-8B-Instruct and the prefix and dual variants of Fast-dLLM v1 are the initial candidates because the authors already expose full, prefix, and dual generation for the same checkpoint. The release milestone is a fresh user running all three settings from one example with the same prompt. Publishing a paired result would show what the switch costs in throughput, memory, and answer quality. [Upstream three-way selection](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) · [Comparison protocol](benchmarking.md)

One subsequent integration on LLaDA can test whether the interface accommodates a different refresh rule. Porting an existing method to Dream can test whether a second backbone can share the same method selection. [Current method collection](../README.md#method-zoo) · [Maintainer workflow](maintaining.md)
