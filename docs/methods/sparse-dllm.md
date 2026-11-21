# Sparse-dLLM

Song et al. cache the keys and values outside the active block, then evict part of that cache. Each block starts with two passes over the whole sequence. On the second pass, every layer stores its keys and values, scores each entry outside the active block against the block's queries, and keeps the top-scoring share. From the third step on, only the active block goes through the model, and it attends to itself and to the kept entries. The cache is cleared when the next block starts. The paper reports peak memory close to that of the uncached model. [Paper](https://arxiv.org/abs/2508.02558) · [Authors’ summary](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/README.md#L10-L16) · [LLaDA generation loop](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/llada_generate.py#L74-L126) · [Cache and eviction](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/modeling_llada.py#L526-L583)

## How entries are scored

Entries on both sides of the block compete in one ranking: the prompt, the completed blocks and the masked suffix. In each layer, the score of an entry is the product of its key and the mean query of the active block, averaged over heads. A max pool over neighboring positions then lifts the score of an entry that sits next to a high-scoring one. The layer keeps the top share and drops the other keys and values, and each layer ranks its own entries. [Eviction](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/modeling_llada.py#L542-L580)

The score is computed from the query and key tensors beside the attention call. The method reads no attention weights, and the attention kernel runs unchanged on a shorter key sequence. [Where a layer stores, evicts and reads](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/modeling_llada.py#L772-L807)

The paper compares delays of zero to five steps before the cache is stored and chooses one. The released loops fix the delay at one step. [Paper, ablations](https://arxiv.org/html/2508.02558v2) · [Step states in the loop](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/llada_generate.py#L84-L96)

## Settings that change behavior

`keep_ratio` is the share of entries outside the active block that each layer keeps. `kernel_size` is the width of the max pool. The authors use 0.5 and 3 for every model in the paper. `block_length` sets the decoding block and also how often the cache is rebuilt. Each block spends two full passes before reuse starts. With a block length of 32 and one step for each position, 30 of a block's 32 steps run on the active block alone. A schedule with few steps in a block leaves little to reuse. [Paper, setup and ablations](https://arxiv.org/html/2508.02558v2) · [Settings passed to the model](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/llada_wrapper.py#L187-L190)

The method adds no decoding rule of its own. The LLaDA loop keeps low-confidence remasking with a fixed number of tokens at each step. The Dream loop decodes block by block, its wrapper defaults to the `entropy` rule, and the first step of a block reveals only the block's first position. Give the baseline the same block schedule when timing either model. [Dream loop](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/dream/generation_utils.py#L414-L447) · [Dream wrapper defaults](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/opencompass/models/sparse_dllm/dream_wrapper.py#L126-L156)

The shipped configurations use a batch size of one, and the eviction code assumes a single sequence.

## Select the method upstream

The code installs into an OpenCompass checkout. Copy the repository's `opencompass/` and `myeval/` directories into it and register the three model classes. The README lists the versions the authors used: `opencompass==0.4.2`, `torch==2.6.0` and `transformers==4.46.3`. Each configuration takes a local path to the checkpoint. [Installation](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/README.md#L22-L50) · [Evaluation commands](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/README.md#L52-L81)

| Checkpoint in the authors’ example | Settings in the example | Entry point |
| --- | --- | --- |
| LLaDA-8B-Instruct | 256 steps, block length 32, output length 256, `kernel_size=3`, `keep_ratio=0.5`, seed 2025 | [OpenCompass configuration](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_performance/eval_sparse_dllm_llada_chat.py#L43-L55) · [Speed script](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_speed/eval_speed_llada_example.sh#L1-L6) |
| LLaDA-1.5 | The same settings | [OpenCompass configuration](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_performance/eval_sparse_dllm_llada_1.5.py#L43-L55) · [Speed script](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_speed/eval_speed_llada_example.sh#L8-L10) |
| Dream-v0-Instruct-7B | 256 steps, output length 256, `kernel_size=3`, `keep_ratio=0.5`, seed 2025, and the wrapper's default block length of 32 | [OpenCompass configuration](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_performance/eval_sparse_dllm_dream_chat.py#L42-L54) · [Speed script](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_speed/eval_speed_dream_example.sh#L1-L6) |
| Dream-v0-Base-7B | The same settings | [OpenCompass configuration](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_performance/eval_sparse_dllm_dream_base.py#L42-L54) · [Speed script](https://github.com/OpenMOSS/Sparse-dLLM/blob/3fd8986bee4ddd68e70ee8041da3a8c9de44f405/myeval/eval_speed/eval_speed_dream_example.sh#L8-L10) |

The speed scripts report tokens per second and peak GPU memory.

## Use it from the Zoo

```python
cache = zoo.attach(model, "sparse-dllm", keep_ratio=0.5, kernel_size=3, delay=1)
```

`keep_ratio` and `kernel_size` are the authors' settings. `delay` is the number of full passes in a block before the pass that stores and evicts. The released loops fix it at 1, and the paper's ablation varies it. Pass the block being decoded as `active`. Batches are supported: each row keeps its own entries. [Policy](../../src/dllm_cache_zoo/methods/sparse_dllm.py)

Sparse-dLLM is the first eviction method in the Zoo. Each layer keeps a different set of entries, so the policy runs its own layer loop on the adapter. It is written from the paper, because the authors' repository has no license file. On LLaDA-8B-Instruct it reveals the same tokens as the authors' loop on every checked prompt. Kept entries are stored in sequence order, where the reference stores them in score order, which changes nothing but the order of the attention sum. [Check report](../checks.md) · [Third integrations](../design/method-selection.md#third-integrations)

The pinned revision has no repository license file. The examples above point directly to that revision. [Upstream repository](https://github.com/OpenMOSS/Sparse-dLLM/tree/3fd8986bee4ddd68e70ee8041da3a8c9de44f405) · [All methods](../../README.md#method-zoo)
