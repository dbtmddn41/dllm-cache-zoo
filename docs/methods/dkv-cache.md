# dKV-Cache

Ma et al. use the order in which positions become decoded to choose which states to reuse. The authors provide two variants with different update sets. [Paper](https://arxiv.org/abs/2505.15781) · [Authors’ setup and examples](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L23-L35)

## Decode

A position that has just changed from a mask to a token gets one more update before its state is reused. Between full refreshes, the model recomputes the still-masked and newly decoded positions. The refresh interval controls how often cached states are rebuilt. [LLaDA implementation](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/generation_utils/llada_dkv_cache_decode.py)

## Greedy

The LLaDA implementation samples an unmasking order up front. Between full refreshes it updates positions just decoded, positions due next, and optionally nearby masked positions. Its example uses random unmasking. A cache-only comparison needs a baseline with the same unmasking order. [LLaDA implementation](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/generation_utils/llada_dkv_cache_greedy.py#L122-L215)

## Select a variant upstream

| Checkpoint in the authors’ example | Variant | Settings in the example | Entry point |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct | Decode | `--decode` with an eight-step refresh interval | [LLaDA driver](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L130-L135) |
| LLaDA-8B-Instruct | Greedy | `--greedy`, random unmasking, a two-step refresh interval, and a four-position window setting | [LLaDA driver](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L130-L135) |
| Dream-v0-Instruct-7B | Decode | `--decode` with a four-step refresh interval in the batch example | [Dream driver and settings](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/README.md#L26-L83) |

The authors document a batch size of one for their LLaDA driver. The pinned revision has no repository license file. The examples above point directly to that revision. [Upstream repository](https://github.com/horseee/dKV-Cache/tree/49a76fcc43b744ec2d960137f216e419317138b1) · [All methods](../../README.md#method-zoo)
