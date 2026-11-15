# dKV-Cache

Ma et al. use the order in which positions become decoded to choose which states to reuse. The authors provide two variants with different update sets. [Paper](https://arxiv.org/abs/2505.15781) · [Authors’ setup and examples](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L23-L35)

## Decode

A position that has just changed from a mask to a token gets one more update before its state is reused. Between full refreshes, the model recomputes the still-masked and newly decoded positions. The refresh interval controls how often cached states are rebuilt. [LLaDA implementation](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/generation_utils/llada_dkv_cache_decode.py)

## Greedy

The LLaDA implementation samples an unmasking order up front. Between full refreshes it updates positions just decoded, positions due next, and optionally nearby masked positions. Its example uses random unmasking. A cache-only comparison needs a baseline with the same unmasking order. [LLaDA implementation](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/generation_utils/llada_dkv_cache_greedy.py#L122-L215)

## Use it from the Zoo

```python
cache = zoo.attach(model, "dkv-cache-decode", refresh_interval=8)
cache = zoo.attach(model, "dkv-cache-greedy", refresh_interval=2, window=4)
```

`refresh_interval` counts steps inside a block. For Decode, the first step of a block runs the whole sequence without storing, the second and every `refresh_interval`-th step run it and store it, and the other steps run the positions that were still masked one step earlier. The default of 8 is the authors' LLaDA example. [Policies](../../src/dllm_cache_zoo/methods/dkv_cache.py)

Greedy needs the positions the sampler is about to reveal, before the forward pass: `state.forward(x, active=block, reveal=positions)`. Only a sampler that fixes its order in advance knows them, such as the random order of the authors' example (`remasking="random_order"` in the Zoo's reference loop). `window` is the number of masked neighbors added around each of those positions. With several positions per step, the reference visits them in unmasking order and the Zoo in sequence order, which can change which neighbors are taken. [Reference loop](../../src/dllm_cache_zoo/loops.py)

Both policies are written from the paper, because the authors' repository has no license file. On LLaDA-8B-Instruct, Decode reveals the same tokens as the authors' driver on every checked prompt at an interval of 8, and all but one of 384 tokens at an interval of 4. Greedy reveals the same tokens on every checked prompt. The store is laid out by position, where the reference concatenates fresh and stored entries, so the attention sums in a different order and results can differ in the last bit. Batches whose rows have different numbers of masked positions are padded with extra positions to recompute, which the reference does not support. [Check report](../checks.md)

## Select a variant upstream

| Checkpoint in the authors’ example | Variant | Settings in the example | Entry point |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct | Decode | `--decode` with an eight-step refresh interval | [LLaDA driver](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L130-L135) |
| LLaDA-8B-Instruct | Greedy | `--greedy`, random unmasking, a two-step refresh interval, and a four-position window setting | [LLaDA driver](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L130-L135) |
| Dream-v0-Instruct-7B | Decode | `--decode` with a four-step refresh interval in the batch example | [Dream driver and settings](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L26-L83) |

The authors document a batch size of one for their LLaDA driver. The pinned revision has no repository license file. The examples above point directly to that revision. [Upstream repository](https://github.com/horseee/dKV-Cache/tree/49a76fcc43b744ec2d960137f216e419317138b1) · [All methods](../../README.md#method-zoo)
