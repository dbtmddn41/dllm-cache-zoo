# dLLM Cache Zoo

A collection of **KV cache methods for diffusion language models**, organized like a model zoo. Pick a method by name and turn it on for the model you already run. Your generation loop and sampler stay as they are.

> [!NOTE]
> The Zoo is a pre-release (`0.1.0.dev0`), installed from source. Nine cache settings run on LLaDA checkpoints, the completed-block cache runs on SDAR and Fast-dLLM v2, and the sub-block cache runs on Fast-dLLM v2. Each was checked against its authors' implementation on a released checkpoint. Measured speed and accuracy results are not published yet. [What was checked](docs/checks.md) · [Roadmap](#roadmap)

## Install

```sh
pip install "dllm-cache-zoo[hf] @ git+https://github.com/dbtmddn41/dllm-cache-zoo.git"
```

The package itself needs only PyTorch. The `hf` extra adds Transformers for loading checkpoints.

## Usage

```python
import torch
from transformers import AutoModel
import dllm_cache_zoo as zoo

model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda().eval()

cache = zoo.attach(model, method="fast-dllm-dual")

with cache.request(prompt_length=prompt_length) as state:
    for start, end in blocks:
        for step in range(steps_per_block):
            logits = state.forward(x, active=(start, end))  # was: model(x).logits
            x = reveal(x, logits)                           # your sampler, unchanged
```

`attach` leaves the model untouched and picks an adapter from its module layout. `request` holds the cache for one generation. `state.forward` takes the place of the model call: it runs the positions the method asks for and reads the others from the store. `active` is the span the sampler may reveal at this step, which is the current block of a blockwise loop. Switching methods changes the `method` name. [Library design](docs/design/library.md) · [The same change in the LLaDA reference loop](examples/own_loop.py)

To try a method without a loop of your own, use a reference loop from the Zoo:

```python
from dllm_cache_zoo.loops import llada_generate

out = llada_generate(cache, prompt_ids, gen_length=128, steps=128, block_length=32)
print(out.steps, out.stats.recomputed_share)  # forward passes, share of positions recomputed
```

```sh
python examples/llada_generate.py --method dkv-cache-decode --set refresh_interval=8
python examples/sdar_generate.py --method block-cache
python examples/fast_dllm_v2_generate.py --method fast-dllm-v2-sub-block
```

`zoo.methods()` lists every method with its paper and the revision its behavior follows. Method settings are keyword arguments of `attach`, for example `zoo.attach(model, "dllm-cache", prompt_interval=100, response_interval=7, transfer_ratio=0.25)`. Each method page lists them.

## Method zoo

| Method | Name in the Zoo | Paper | What it reuses between denoising steps | Checkpoints checked |
| --- | --- | --- | --- | --- |
| Full recomputation | `none` | | Nothing. The control for every comparison | LLaDA-8B-Instruct · SDAR-1.7B-Chat · Fast_dLLM_v2_7B |
| [Fast-dLLM v1 · Prefix](docs/methods/fast-dllm.md#prefix-cache) | `fast-dllm-prefix` | [Paper](https://arxiv.org/abs/2505.22618) | Prompt and completed blocks before the active block | LLaDA-8B-Instruct |
| [Fast-dLLM v1 · Dual](docs/methods/fast-dllm.md#dual-cache) | `fast-dllm-dual` | [Paper](https://arxiv.org/abs/2505.22618) | Both sides of the active block, including the masked suffix | LLaDA-8B-Instruct |
| [dKV-Cache · Decode](docs/methods/dkv-cache.md#decode) | `dkv-cache-decode` | [Paper](https://arxiv.org/abs/2505.15781) | Previously decoded positions after a one-step delay | LLaDA-8B-Instruct |
| [dKV-Cache · Greedy](docs/methods/dkv-cache.md#greedy) | `dkv-cache-greedy` | [Paper](https://arxiv.org/abs/2505.15781) | Positions outside a small update set chosen from the decoding order | LLaDA-8B-Instruct |
| [dLLM-Cache](docs/methods/dllm-cache.md) | `dllm-cache` | [Paper](https://proceedings.mlr.press/v306/liu26fi.html) | Prompt and response states on separate schedules, with selective response updates | LLaDA-8B-Instruct |
| [d²Cache](docs/methods/d2cache.md) | `d2cache` | [Paper](https://arxiv.org/abs/2509.23094) | Positions outside a per-step update set chosen by certainty and attention rollout | LLaDA-8B-Instruct |
| [Elastic-Cache](docs/methods/elastic-cache.md) | `elastic-cache` | [Paper](https://arxiv.org/abs/2510.14973) | Positions outside a sliding window, until attention on the most-attended tokens drifts | LLaDA-8B-Instruct |
| [DPad](docs/methods/dpad.md) | `dpad` | [Paper](https://arxiv.org/abs/2508.14148) | Nothing by itself. Distant masked suffix positions are left out of each forward pass, under Prefix, Dual or no cache | LLaDA-8B-Instruct |
| [Sparse-dLLM](docs/methods/sparse-dllm.md) | `sparse-dllm` | [Paper](https://arxiv.org/abs/2508.02558) | The top-scoring share of keys and values outside the active block, chosen once for each block | LLaDA-8B-Instruct |
| [Block cache](docs/methods/block-diffusion.md#block-cache) | `block-cache` | [Block Diffusion](https://arxiv.org/abs/2503.09573) · [SDAR](https://arxiv.org/abs/2510.06303) | Completed blocks of a block diffusion model, stored once each block is final | SDAR-1.7B-Chat · Fast_dLLM_v2_7B |
| [Fast-dLLM v2 · Sub-block](docs/methods/block-diffusion.md#fast-dllm-v2-sub-block-cache) | `fast-dllm-v2-sub-block` | [Paper](https://arxiv.org/abs/2509.26328) | Completed blocks, plus the rest of the active block while one sub-block is refined | Fast_dLLM_v2_7B |

A checkpoint in the last column means the Zoo's method was run against the authors' implementation on those weights and the two were compared. The check report gives the outcome for each method and the settings used. LLaDA-8B-Base and LLaDA-1.5 share the LLaDA model code and load through the same adapter, and Dream has no adapter yet. Each method page links the authors' examples and records what the Zoo does differently. [Check report](docs/checks.md) · [Browse by checkpoint](docs/methods.md)

### More methods in the catalog

The catalog tracks 32 more methods with their papers, available code and licenses. The selection review records which methods are planned, where other teams use them, and what it would take to attach them to an existing model. [Method selection](docs/design/method-selection.md) · [Full catalog](docs/catalog.md)

| Group | Methods |
| --- | --- |
| [Selective recomputation on bidirectional models](docs/catalog.md#selective-recomputation-on-bidirectional-models) | Flash-dLLM · Dynamic-dLLM · FlashDLM · dInfer cache options · EntropyCache · SPA-Cache · DyLLM · ES-dLLM · Polestar · DARE · Window-Diffusion · Archer · LaCache · SureLock |
| [Suffix pruning](docs/catalog.md#suffix-pruning) | Streaming-dLLM · Structured Suffix Modeling |
| [Eviction and sparse attention for long contexts](docs/catalog.md#eviction-and-sparse-attention-for-long-contexts) | MaskKV · SparseD · Focus-dLLM · Prefilling-dLLM · PulseCol · WaveFilter · GroupKV · Step-dLLM |
| [Block diffusion](docs/catalog.md#block-diffusion) | MAGE · LoSA · FlashBlock · HERALD · MaskAhead · FOCUS |
| [Caches shared across requests](docs/catalog.md#caches-shared-across-requests) | BiCache · Affix Cache |

## Model families

| Family | Checkpoints | What a cache can rely on |
| --- | --- | --- |
| Bidirectional masked diffusion | LLaDA-8B, LLaDA-1.5 in the Zoo. Dream-v0 in upstream examples | Every position attends to every other position, so any reuse between steps is an approximation. |
| Block diffusion | SDAR and Fast-dLLM v2 in the Zoo. LLaDA2.0 in upstream examples | A block attends to itself and earlier blocks, so a completed block is stored once and reused exactly. |

## Roadmap

1. **Done.** One adapter for LLaDA with full recomputation, Fast-dLLM Prefix and Dual, dLLM-Cache and dKV-Cache Decode, with no method-specific model code.
2. **Done.** Block diffusion checkpoints (SDAR, Fast-dLLM v2) with their block cache, and d²Cache and Elastic-Cache on the LLaDA adapter. dKV-Cache Greedy, Fast-dLLM v2 Sub-block, DPad and Sparse-dLLM are in as well, which completes the eleven selected settings.
3. **Next.** Measured results for each method and checkpoint pair under the comparison protocol, a Dream adapter, and an installable release with a support table. Further methods are selected by conference acceptance, external use and compatibility with a host.
4. Carry one method into an existing serving engine and validate it under load.

The design notes describe each step and the interface. [Roadmap in detail](docs/design/roadmap.md) · [Design notes](docs/design/README.md)

## Related projects

| Project | What it offers | How the Zoo relates |
| --- | --- | --- |
| [Research dLLM](https://github.com/Kamichanw/d2Cache/tree/79fb5f64864389ba093c29122a21ef0391080312) | Config-selectable Prefix and Dual, dLLM-Cache, d²Cache and a block diffusion cache, with several decoding strategies and lm-eval tasks. Maintained by the d²Cache authors. | Closest in purpose. The Zoo packages only the caches, for use inside a generation loop you already have and on the model class you already load, and includes methods outside that codebase such as dKV-Cache, Elastic-Cache, DPad and Sparse-dLLM. |
| [dLLM](https://github.com/ZHZisZZ/dllm/tree/ca176752fbceec49c6b4777a2c18ae88e4eb10ed) | Training, inference and evaluation pipelines for LLaDA, Dream and LLaDA2, including a Fast-dLLM pipeline. | A host for the Zoo's adapters: its LLaDA model class has the layout the LLaDA adapter reads. |
| [dInfer](https://github.com/inclusionAI/dInfer/tree/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f) | A batched inference engine with Prefix and Dual cache options and vLLM and SGLang backends. | Candidate engine for running Zoo methods in serving. |

## Contributing

Developing a cache method, or missing one from the list? The smallest contribution is one row in the catalog, and the guide also covers method pages and the checks to run. [Contribution guide](CONTRIBUTING.md)

## License and citation

The Zoo is licensed under Apache-2.0. Each method module names the paper it implements and the upstream revision it follows, and each method page records the upstream license. Methods whose reference code carries no license are written from the paper. [License](LICENSE) · [Third-party notices](NOTICE) · [Cite the Zoo](CITATION.cff)
