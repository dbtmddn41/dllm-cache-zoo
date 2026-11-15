# Check report

Every method in the Zoo was compared with its authors' implementation on a released checkpoint. This page records what was run on 15 November 2026 and what came out. The checks establish that a method behaves like its reference. They take no timings and score no benchmark, so they say nothing yet about speed or accuracy. [Comparison protocol for measured results](design/benchmarking.md) · [Method table](../README.md#method-zoo)

## How a method is checked

1. **CPU tests.** Small random models with the module layout of LLaDA, SDAR and Fast-dLLM v2 run every method to completion. Each method must match full recomputation under the settings where its definition says it recomputes everything, and the block cache must match the masked full pass. These run on every pull request. [Toy models](../tests/toy.py) · [Method tests](../tests/test_methods.py) · [Block cache tests](../tests/test_block_cache.py) · [Sub-block cache tests](../tests/test_fast_dllm_v2.py)
2. **Adapter against the model.** On the released checkpoint, the adapter's full pass is compared with the model's own forward on the same tokens.
3. **Method against its reference.** The authors' generation loop runs twice on the same prompts: once as published, with their model class and cache, and once with the model call answered by the Zoo on the stock checkpoint. The sampler is theirs both times, so only the cache differs. The check passes when both runs reveal the same tokens. [Comparison script](../checks/compare_upstream.py)

Some references needed a different arrangement. Elastic-Cache, DPad and Fast-dLLM v2 build their cache into the loop, so the Zoo side uses its own copy of their decoding loop. d²Cache is replayed step by step. [Step-by-step replay](#step-by-step-replay)

## LLaDA-8B-Instruct

The adapter's full pass returns the same logits as the model's forward, bit for bit, on all three prompts. [Result file](../checks/results/adapter.json)

Each row below covers three prompts and 128 response positions per prompt. Unless a threshold is given, a run takes 128 steps in blocks of 32, with low-confidence remasking at temperature 0. The last column is the share of position-passes the Zoo ran, where full recomputation is 1.0. It shows that the cache was active during the check. It is a count of work and no measure of speed.

| Method | Settings | Tokens equal to the reference | Steps, reference and Zoo | Positions run |
| --- | --- | --- | --- | --- |
| `none` | | 384 of 384 | 128 each | 1.00 |
| `fast-dllm-prefix` | | 384 of 384 | 128 each | 0.52 |
| `fast-dllm-prefix` | threshold 0.9 | 384 of 384 | 56, 79, 61 on both sides | 0.57 |
| `fast-dllm-dual` | | 384 of 384 | 128 each | 0.23 |
| `fast-dllm-dual` | threshold 0.9 | 384 of 384 | 52, 80, 66 on both sides | 0.25 |
| `dkv-cache-decode` | `refresh_interval=8` | 384 of 384 | 128 | 0.50 |
| `dkv-cache-decode` | `refresh_interval=4` | 383 of 384 | 128 | 0.57 |
| `dkv-cache-greedy` | `refresh_interval=2`, `window=4`, random order, seed 1234 | 384 of 384 | 128 | 0.52 |
| `dllm-cache` | `prompt_interval=100`, `response_interval=7`, `transfer_ratio=0.25` | 384 of 384 | 128 | 0.32 |
| `dllm-cache` | intervals 5 and 3, `transfer_ratio=0.25` | 384 of 384 | 128 | 0.46 |
| `dllm-cache` | intervals 5 and 3, `transfer_ratio=0` | 384 of 384 | 128 | 0.33 |
| `elastic-cache` | `gamma=0.9`, `track_num=1`, window 16, threshold 0.9 | 384 of 384 | 49, 116, 56 on both sides | 0.23 |
| `dpad` | `cache="none"`, `window=128`, `k=4`, `scale=2`, seed 1234 | 384 of 384 | 128 each | 0.85 |
| `dpad` | `cache="prefix"`, same sampler settings | 384 of 384 | 128 each | 0.37 |
| `dpad` | `cache="prefix"`, threshold 0.9 | 384 of 384 | 44, 74, 21 on both sides | 0.47 |
| `dpad` | `cache="dual"`, same sampler settings | 384 of 384 | 128 each | 0.22 |
| `sparse-dllm` | `keep_ratio=0.5`, `kernel_size=3` | 384 of 384 | 128 | 0.25 |

The one differing token is in the dKV-Cache Decode run with an interval of 4. That policy keeps its store laid out by position, where the reference concatenates fresh and stored entries, so the attention sums in a different order and a near-tie can fall the other way in bfloat16. The settings of every run and the per-prompt counts are in the result files. [Result files](../checks/results/fast-dllm-prefix.json)

### Step-by-step replay

The reference for d²Cache is the authors' Research dLLM codebase, which drives its caches from its own model class and decoding loop. The check runs that loop once and records, at every step, the tokens that went in and the positions the cache ran. The Zoo then replays the same tokens on the stock checkpoint. Both sides see identical inputs, and the check compares the positions they chose to run and whether the Zoo's logits pick the token the reference revealed. [Comparison script](../checks/compare_d2cache.py)

| Precision | Steps with the same positions | Mean overlap of the positions | Revealed tokens the Zoo's logits pick |
| --- | --- | --- | --- |
| float32 | 384 of 384 | 1.000 | 381 of 381 |
| bfloat16 | 252 of 384 | 0.964 | 380 of 381 |

In float32 the two implementations run the same positions at every step. In bfloat16 the positions differ on a third of the steps by a few entries. The selection ranks positions by attention rollout and by confidence, both computed in half precision, and many of those values are near ties. Settings: `rollout_p=0.1`, `current_k=32`, `sigma=10`, `inflate_w=0`, one token per step. [float32 result](../checks/results/d2cache-float32.json) · [bfloat16 result](../checks/results/d2cache-bfloat16.json)

Two things were needed to run the reference at its pinned revision. Its cache switches to the decode state before it checks for a query mask and asserts on the first step, so the check gives it an empty mask there, which has the effect of the guard in the previous revision. The reference also never refreshes its stored confidence, so the Zoo ran with `refresh_confidence=False`. The Zoo's default refreshes it. [d²Cache page](methods/d2cache.md#use-it-from-the-zoo)

## Block cache on SDAR

A block-causal model makes the completed-block cache exact, so the check compares logits. For each active block of a greedy generation on `JetLM/SDAR-1.7B-Chat`, with blocks of 4, it takes the logits of the block from the Zoo's block cache, from the Zoo's uncached pass over the whole prefix with the block attention mask, and from the model's own forward with the same mask. [Comparison script](../checks/compare_sdar.py)

| Precision | Prompt | Cache and uncached pass | Cache and the model's forward | Uncached pass and the model's forward | Largest logit |
| --- | --- | --- | --- | --- | --- |
| float32 | 1 | 0.00011 | 0.00011 | 0.00007 | 38.0 |
| float32 | 2 | 0.00006 | 0.00008 | 0.00006 | 23.5 |
| float32 | 3 | 0.00010 | 0.00009 | 0.00005 | 57.3 |
| bfloat16 | 1 | 0.50 | 0.50 | 0.50 | 66.5 |
| bfloat16 | 2 | 13.00 | 9.50 | 9.50 | 50.8 |
| bfloat16 | 3 | 0.50 | 0.75 | 0.50 | 57.2 |

Each cell is the largest absolute logit difference over all blocks and steps. In float32 the cache, the uncached pass and the model's own forward agree to within 0.00011 on every prompt, and they predict the same token at all 392 positions. In bfloat16 the three computations use different attention kernels and differ by rounding. One step of bfloat16 at these magnitudes is 0.5. On the second prompt the uncached pass and the model's forward differ from each other about as much as the cache differs from either, so the difference comes from the kernels and not from caching. The predicted token is the same for 390 of 392 positions. [float32 result](../checks/results/block-cache-sdar-float32.json) · [bfloat16 result](../checks/results/block-cache-sdar-bfloat16.json)

The bfloat16 numbers change from run to run, and the float32 numbers repeat. The released model's own bfloat16 forward is not reproducible across processes: on one fixed input, two processes returned logits that differ by up to 0.5 and disagree on one of 24 predicted tokens, while two calls inside one process returned identical logits. Across five runs of this check, the largest difference between the cache and the model's forward on the second prompt ranged from 3.0 to 9.5, and the uncached pass was as far from the model's forward each time.

The SDAR generation script, with its own cache, was also compared with the Zoo's reference loop on 64 response positions in bfloat16. The script decodes through flash-attn, which takes half precision only, so this comparison has no float32 run, and its outcome varies with the run for the reason above. Over four runs, greedy decoding gave the same tokens on the third prompt every time, on the first prompt in three runs, and never on the second, where a differing token early in the response sends the two generations apart. In the run stored here the counts are 17 of 64, 7 of 65 and 64 of 64 with greedy decoding, and 52, 12 and 64 with sampling at temperature 1 and the same seed.

## Fast-dLLM v2

The reference is the generation function that ships with `Efficient-Large-Model/Fast_dLLM_v2_7B`, which has both cache levels built in. The Zoo side runs the same decoding rule through its reference loop: blocks of 32, sub-blocks of 8, greedy, up to 128 new tokens. Each cell counts the prompts, out of three, on which the Zoo reveals the same tokens as the reference. [Comparison script](../checks/compare_fast_dllm_v2.py)

| Zoo method | Reference setting | Threshold | float32 | bfloat16 |
| --- | --- | --- | --- | --- |
| `none` | `use_block_cache=False` | 1, one token per step | 3 of 3 | 1 of 3 |
| `block-cache` | `use_block_cache=False` | 1 | 3 of 3 | 2 of 3 |
| `block-cache` | `use_block_cache=False` | 0.9 | 3 of 3 | 2 of 3 |
| `fast-dllm-v2-sub-block` | `use_block_cache=True` | 1 | 3 of 3 | 2 of 3 |
| `fast-dllm-v2-sub-block` | `use_block_cache=True` | 0.9 | 3 of 3 | 2 of 3 |

In float32 every run reveals the same tokens as the reference: 291 of 291 over the three prompts, in each row. The logits of a block also agree with the model's own forward to within 0.0001. In bfloat16 the logits differ from the model's forward by at most 0.63 on logits up to 35, one or two rounding steps, and in six of the fifteen runs a near tie falls the other way and the two generations part. Those runs agree on 81 of 104 tokens (`none`, first prompt), 96 of 97 (`none` and `block-cache`, third prompt) and 44 of 82 (the sub-block cache, second prompt). The adapter and the checkpoint call the attention kernel differently (with or without an all-true mask, with repeated or grouped key heads), which accounts for the rounding. [float32 result](../checks/results/fast-dllm-v2-float32.json) · [bfloat16 result](../checks/results/fast-dllm-v2-bfloat16.json)

The reference's `none` row is its exact block level too, because its generation function always stores completed blocks. The Zoo's `none` recomputes the whole prefix at every step, and matching tokens show that the block cache changes nothing.

## Environment

| Item | Value |
| --- | --- |
| GPU | One NVIDIA H100 NVL |
| LLaDA and Fast-dLLM v2 checks | Docker image `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`, PyTorch 2.9.1, Transformers 4.53.1 |
| d²Cache check | The same image, with Transformers 4.57.0 for Research dLLM |
| SDAR check | Docker image `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`, PyTorch 2.8.0, flash-attn 2.8.3, Transformers 4.53.1 |
| Checkpoints | `GSAI-ML/LLaDA-8B-Instruct` at `08b83a6`, `JetLM/SDAR-1.7B-Chat` at `14fcc4b`, `Efficient-Large-Model/Fast_dLLM_v2_7B` at `0661abf` |
| References | The revisions pinned on each method page |

The SDAR model code compiles a normalization kernel on first use, which needs a C compiler in the image. The prebuilt flash-attn wheels stop at PyTorch 2.8, which is why that check runs on 2.8.

## Reproduce a check

```sh
git clone https://github.com/NVlabs/Fast-dLLM.git upstream/fast-dllm
git -C upstream/fast-dllm checkout a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b
python checks/compare_upstream.py --case fast-dllm-dual --root upstream/fast-dllm --threshold 0.9
```

To run every case, use the runner script with a directory of reference checkouts. `--case` selects the method and `--root` is a checkout of its reference. One upstream is imported per process, because several of them ship modules with the same names. The other cases are `adapter`, `fast-dllm-none`, `fast-dllm-prefix`, `dkv-cache-decode`, `dkv-cache-greedy`, `dllm-cache`, `elastic-cache`, `dpad-none`, `dpad-prefix`, `dpad-dual` and `sparse-dllm`. d²Cache, SDAR and Fast-dLLM v2 have their own scripts. [Cases and options](../checks/compare_upstream.py) · [Runner for all cases](../checks/run_all.sh)

## What the checks do not cover

- **Speed, memory and task accuracy.** None was measured. The comparison protocol describes how they will be. [Comparison protocol](design/benchmarking.md)
- **Other checkpoints.** LLaDA-8B-Base and LLaDA-1.5 use the model code of LLaDA-8B-Instruct and load through the same adapter, and larger SDAR checkpoints use the code of SDAR-1.7B-Chat. They were not run. Dream has no adapter.
- **Batches.** The references decode one sequence at a time, and so did the checks. The CPU tests run batches of two for the methods on LLaDA except Elastic-Cache, which takes one sequence. The sub-block loop takes one sequence as well.
- **Long prompts and long responses.** The prompts are short questions and the responses 128 positions, or 64 on SDAR.
