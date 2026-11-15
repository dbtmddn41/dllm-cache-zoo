# Method catalog

The Zoo surveys a broad range of published methods for diffusion language models. This index lists candidates that have no method page yet. The selection review records planned integrations and the reasons for deferring other candidates. [Method selection](design/method-selection.md) · [Methods with a page](../README.md#method-zoo) · [Browse by checkpoint](methods.md)

Each row links the paper version that was read and, where the authors published code, the revision that was checked. The descriptions come from the abstracts and repository READMEs. A method page replaces a row once the method's settings and examples have been read in full. The models column names models from the abstract or the README.

Last survey: 14 November 2026. A dash in the code column means the abstract links no repository and a GitHub search on that date found none.

## Selective recomputation on bidirectional models

These methods decide which positions to recompute at a denoising step, or when to refresh, on LLaDA and Dream style models. They are the closest fit to the policy interface. [How published methods fit](design/library.md#how-published-methods-fit)

| Method | Cache component | Models | Code and license |
| --- | --- | --- | --- |
| [Flash-dLLM](https://arxiv.org/abs/2609.26796v1) | Flash-Cache fuses the QKV projection, rotary embedding and cache write, and refreshes newly decoded tokens plus a fixed budget of highly attended decoded tokens. It ships with Flash-Verify, a draft-and-verify decoding rule. | LLaDA-1.5 | [VILA-Lab/Flash-dLLM](https://github.com/VILA-Lab/Flash-dLLM/blob/a667345f5746bb74f060c87bae5997fc3de9710c/README.md#L41-L62) · Apache-2.0 |
| [Dynamic-dLLM](https://arxiv.org/abs/2606.26120v1) | Dynamic Cache Updating reallocates the update budget for each layer at each denoising step. It ships with an adaptive parallel decoding rule. | LLaDA-8B-Instruct | [TianyiWu233/DYNAMIC-DLLM](https://github.com/TianyiWu233/DYNAMIC-DLLM/blob/43cb93b7805842d6017e1ee4be4e34345f1ae032/README.md#L5-L64) · no license file |
| [FlashDLM](https://arxiv.org/abs/2505.21467v2) | Sliding-window KV caching, used with decoding guided by a small autoregressive model. The authors describe the code as experimental. | Dream-7B Instruct | [ZhanqiuHu/flash-dlm-experimental](https://github.com/ZhanqiuHu/flash-dlm-experimental/blob/552eaaed0f58061414f5bf2d8e974744e8fbc5e0/README.md#L14-L23) · MIT |
| [dInfer cache options](https://arxiv.org/abs/2510.08666) | Prefix and Dual with `prefix_look`, `after_look` and `warmup_times` settings. The dInfer paper describes a vicinity refresh for states near the active tokens. | LLaDA-MoE-7B-A1B-Instruct | [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md#L152-L173) · Apache-2.0 |
| [EntropyCache](https://arxiv.org/abs/2603.18489v3) | Uses the maximum entropy of the newly decoded token distributions to decide whether a step recomputes, and recomputes the most recently decoded tokens. | LLaDA-8B-Instruct, Dream-7B-Instruct | [mscheong01/EntropyCache](https://github.com/mscheong01/EntropyCache/tree/31d05de9d6517277fbaf6316a6cb079807605e36) · code release pending |
| [SPA-Cache](https://proceedings.mlr.press/v306/sun26ao.html) | Finds the tokens that need an update through a low-dimensional singular proxy, and gives stable layers a smaller update budget. | LLaDA, Dream-v0-Instruct-7B | [wenhao728/spa-cache](https://github.com/wenhao728/spa-cache/tree/f26dd1f6114068043eabf995e8ebf9a086f53aac) · no license file, MIT badge in the README |
| [DyLLM](https://proceedings.mlr.press/v306/lee26i.html) | Compares attention contexts between adjacent steps, recomputes attention and feed-forward layers only for the tokens that changed, and reuses cached activations for the rest. | LLaDA, Dream | [scale-snu/DyLLM](https://github.com/scale-snu/DyLLM/tree/79c684184d79a6b64a36b10cd7e74a52fd4af45c) · MIT |
| [ES-dLLM](https://arxiv.org/abs/2603.10088v1) | Skips tokens in early layers. Importance is estimated from the change in intermediate tensors and from confidence scores of earlier iterations. | LLaDA-8B, Dream-7B | [zhuzj19/ES-dLLM](https://github.com/zhuzj19/ES-dLLM/tree/3c5151d44d88354925a2f1b4a75745aee3444f36) · MIT |
| [Polestar](https://arxiv.org/abs/2607.14107v1) | Polestar-Cache finds stale cache positions from the drift of token representations and refreshes them sparsely. It ships with Polestar-Commit, a drift-based rule for committing tokens. | LLaDA-8B-Instruct, LLaDA-1.5, Dream-7B, LLaDA-V | [gt-synergy-lab/Polestar](https://github.com/gt-synergy-lab/Polestar/tree/0044144771ad05f02d6befd13d1f7fbfb7d6abe7) · MIT |
| [DARE](https://arxiv.org/abs/2605.08134v1) | DARE-KV reuses cached key and value activations and DARE-O reuses attention outputs. Reuse is predicted from the change in query representations over time. | Not named in the abstract | The repository linked in the abstract returned 404 on the survey date |
| [Window-Diffusion](https://arxiv.org/abs/2601.20332v3) | Slides a computation window over the sequence. Active tokens are computed, buffer tokens use cached keys and values with periodic refresh, and tokens far outside the window are pruned. | LLaDA, Dream | [vhicrgit/Window-Diffusion](https://github.com/vhicrgit/Window-Diffusion/tree/d8bb349e81b74741900049f307184e848006b330) · no license file |
| [Archer](https://arxiv.org/abs/2608.08086v3) | Targets decoders that can roll back earlier tokens. It recomputes the response at every update and reuses prompt keys and values within a bounded neighborhood of response states. | LLaDA-8B-Instruct | [Hxnng/Archer](https://github.com/Hxnng/Archer/tree/ab3e4cce51ae72cb04ac36b824e4ff1bc5016585) · no license file |
| [LaCache](https://arxiv.org/abs/2607.16339v2) | Exact caches for tokens that stay unchanged within a block: embedding outputs, pre-attention states and FlashAttention softmax statistics. It ships with FP8 quantization of feed-forward layers. | Not named in the abstract | — |
| [SureLock](https://arxiv.org/abs/2602.06412v3) | Locks an unmasked position once its predicted distribution has stabilized across steps. A locked position skips its query projection and feed-forward sublayers, and its cached keys and values stay available to the other positions. | LLaDA-8B | — · the [project page](https://daioba.github.io/surelock) in the abstract links no repository |

## Suffix pruning

These methods shorten the masked suffix that enters each forward pass. DPad belongs to this group and has a method page. The Structured Suffix Modeling abstract describes the method as compatible with a KV cache. [DPad](methods/dpad.md)

| Method | What it prunes | Models | Code and license |
| --- | --- | --- | --- |
| [Streaming-dLLM](https://arxiv.org/abs/2601.17917v3) | Prunes redundant mask tokens in the suffix. It ships with a confidence-aware decoding rule that exits early for converged tokens. | LLaDA-1.5, LLaDA-8B-Instruct | [xiaoshideta/Streaming-dLLM](https://github.com/xiaoshideta/Streaming-dLLM/tree/c4320c2d42f049f4610d8273217f5d168146d016) · no license file |
| [Structured Suffix Modeling](https://arxiv.org/abs/2608.23167v1) | Splits the suffix into local, middle and tail regions and keeps a different number of tokens in each. It carries the previous step's decoding results into the suffix representations. | Three models in the paper, LLaDA in the README | [zifengcheng/SSM](https://github.com/zifengcheng/SSM/tree/c049852ff1dbc7b47bbb88a6b4e6e469f0b6a1f0) · no license file, Apache-2.0 headers in source files |

## Eviction and sparse attention for long contexts

These methods choose which cached entries to keep, or which keys a query attends to. Their main target is the memory and attention cost of long prompts. Sparse-dLLM belongs to this group and has a method page. [Sparse-dLLM](methods/sparse-dllm.md)

| Method | What it selects | Models | Code and license |
| --- | --- | --- | --- |
| [MaskKV](https://arxiv.org/abs/2510.09309v1) | Scores prompt tokens for each head by the attention of mask queries and evicts the low-scoring ones, with a cache budget that varies by layer. | LLaDA-8B, Dream-7B | [jianuo-huang/MaskKV](https://github.com/jianuo-huang/MaskKV/tree/428b009ff83d451d60c080937b99952afe175d83) · code release pending |
| [SparseD](https://arxiv.org/abs/2509.24014v1) | Computes a sparse attention pattern for each head once and reuses it across steps, after early steps that run full attention. | LLaDA-1.5, Dream-v0-Instruct-7B | [INV-WZQ/SparseD](https://github.com/INV-WZQ/SparseD/tree/52155b4aa78368695f96744cade06d2e0865d085) · no license file |
| [Focus-dLLM](https://arxiv.org/abs/2602.02159v1) | Predicts the regions about to be unmasked from the previous step's confidence, and prunes attention outside them while keeping attention sinks. | LLaDA, Dream-7B | [Longxmas/Focus-dLLM](https://github.com/Longxmas/Focus-dLLM/tree/3e446a3c17351619de5e4bca0f1227372570e27f) · no license file |
| [Prefilling-dLLM](https://arxiv.org/abs/2606.10537v2) | Splits the prefix into chunks, caches their keys and values once, and decodes against the most relevant chunks. | LLaDA, Dream-v0-Base-7B | [menik1126/Prefilling-dLLM](https://github.com/menik1126/Prefilling-dLLM/tree/7ae0c9272bce3564acc3ab69ed72d099eb5631d0) · MIT |
| [PulseCol](https://arxiv.org/abs/2605.20813v1) | Column-sparse attention. The pattern is found at an early step, reused across steps and refreshed at a few intermediate steps. | Not named in the abstract | — |
| [WaveFilter](https://arxiv.org/abs/2606.00724v1) | Picks key tokens in a long context with a wavelet decomposition and builds a sparse KV cache from them. | Not named in the abstract | — |
| [GroupKV](https://arxiv.org/abs/2609.17573v1) | Selects context in contiguous groups, prefetches offloaded keys and values, and corrects stale entries after updates. | Not named in the abstract | — |
| [Step-dLLM](https://arxiv.org/abs/2611.01320v1) | Refreshes a block-sparse attention mask at anchor steps along the denoising trajectory and reuses it between them, with a sparsity budget that varies by head. It relies on a fused Triton kernel for block-sparse attention. | LLaDA-1.5, Dream-7B-Instruct | — |

## Block diffusion

These methods build on the block cache of a block diffusion model. [Block diffusion caches](methods/block-diffusion.md)

| Method | Cache component | Models | Code and license |
| --- | --- | --- | --- |
| [MAGE](https://arxiv.org/abs/2602.14209v2) | Runs one exact attention pass on the fully masked block at its first step, and reuses the selected KV index sets for the remaining steps of that block. | Three block diffusion families, not named in the abstract | — |
| [LoSA](https://arxiv.org/abs/2604.12056v1) | Reuses cached prefix-attention results for tokens whose hidden states are stable between steps, and applies sparse attention only to active tokens. | Not named in the abstract | — |
| [FlashBlock](https://arxiv.org/abs/2602.05305v3) | Caches the attention output that comes from tokens outside the current block and reuses it across the block's steps. | Diffusion language models and video generation models | [Caesarhhh/Flash_Block](https://github.com/Caesarhhh/Flash_Block/tree/a322fb80f29df0d4cc842fbc011929bc85ba8f36) · no license file |
| [HERALD](https://arxiv.org/abs/2606.21633v3) | Keeps the full KV cache in CPU memory, selects a KV subset once per block, and prepares the next block's subset while the current block denoises. | Four block diffusion models, not named in the abstract | — |
| [MaskAhead](https://arxiv.org/abs/2610.06996v1) | Ranks KV entries with mask queries. Masks of the current block select what to attend to, and probes of upcoming blocks decide what to evict. | Fast-dLLM v2, DreamReasoner, LLaDA2.0-mini | [brain-lab-research/MaskAhead](https://github.com/brain-lab-research/MaskAhead/tree/8df6f1ce4c0b4fec053f156bbb191ad9a23069e0) · Apache-2.0 |
| [FOCUS](https://arxiv.org/abs/2601.23278v2) | Predicts from early-layer attention which tokens of the active block can be decoded, and evicts the others from the forward pass. It is built on LMDeploy and includes an SGLang implementation. | SDAR-8B-Chat, LLaDA2.0 | [sands-lab/FOCUS](https://github.com/sands-lab/FOCUS/tree/277f1d158b977867425a50ed17a6218466139c9e) · Apache-2.0 |

## Caches shared across requests

These methods reuse states between requests that share a span of text. They matter most in serving. [Serving integration](design/serving.md)

| Method | Cache component | Models | Code and license |
| --- | --- | --- | --- |
| [BiCache](https://arxiv.org/abs/2606.07571v4) | Reuses the keys and values of a shared prefix across requests in shallow layers, and picks the safe layer depth from the share of shared-prefix tokens in each request. | LLaDA-8B-Instruct | [OSSS-KU/BiCache](https://github.com/OSSS-KU/BiCache/tree/6820c1f2e640490b7575dfb5aebac19e8c08b5c1) · AGPL-3.0 |
| [Affix Cache](https://arxiv.org/abs/2608.26140v2) | Reuses a cached span shared across requests at any position, and recomputes keys and values only for a request-specific set of anchor tokens. | LLaDA-8B-Instruct, Dream | [sands-lab/ACache](https://github.com/sands-lab/ACache/tree/b70ad2be24ae1b77c124dd637e803b7877e0a36b) · Apache-2.0 |

## What the catalog leaves out

The catalog lists methods that store, select or skip model states at inference time without training. The same survey found related work that falls outside that line. Decoding rules and speculative decoding stay with the host, for example Fast-dLLM++. Models built on causal attention so that a standard prefix cache applies are checkpoints, for example WeDLM. Cache quantization and memory planning are separate topics, for example dLLM-KVQ and Mosaic. So are visual token pruning and early stopping for multimodal models, for example DRAMA, whose abstract states that it uses no attention caching. [Scope](design/roadmap.md#scope) · [Fast-dLLM++](https://arxiv.org/abs/2606.02955v3) · [WeDLM](https://arxiv.org/abs/2512.22737v1) · [dLLM-KVQ](https://arxiv.org/abs/2610.39853v1) · [Mosaic](https://arxiv.org/abs/2601.06562v2) · [DRAMA](https://arxiv.org/abs/2611.16345v1)

## Keep the catalog current

The survey used two sources: an arXiv search for diffusion language model papers with cache, KV, eviction, reuse, skipping or pruning in the title, and the efficiency section of the Awesome-Diffusion-LLM list. Check a paper's abstract and its repository before adding a row, since a list entry can be out of date. To add a row, give the paper version, a description of what the method stores or skips, the models it ran on, and the code revision with its license. [arXiv API](https://info.arxiv.org/help/api/index.html) · [Awesome-Diffusion-LLM](https://github.com/AIDASLab/Awesome-Diffusion-LLM/tree/9836c49fe40b58d9b0d3640311a8001219790c05) · [Contribution guide](../CONTRIBUTING.md#add-a-method-to-the-catalog)
