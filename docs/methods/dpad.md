# DPad

Chen et al. shorten the masked suffix that goes through the model. DPad stores no states of its own. Before a block is decoded, it keeps every position up to the end of that block and a random subset of the masked positions in a window right after it. The rest of the suffix stays out of the model input until the next block. A position is less likely to be kept the farther it lies from the block, and the subset is drawn again for each block. [Paper](https://arxiv.org/abs/2508.14148) · [Authors’ description](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/README.md#L72-L97) · [Suffix sampling](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/generate_dropout.py#L96-L104)

## Use with a cache

The authors’ code builds on Fast-dLLM and keeps its three loops: no cache, Prefix and Dual. Without a cache, every pass runs on the pruned sequence. With a cache, the first pass of a block does. Under Prefix, later passes send the active block and the kept suffix positions. Under Dual, later passes send only the active block, and the kept suffix positions hold their keys and values from the first pass. The paper's cached runs use Prefix, and the authors call Dual unnecessary once the suffix is pruned. [Prefix loop](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/generate_dropout.py#L182-L279) · [Dual loop](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/generate_dropout.py#L282-L367) · [Paper, §4.1](https://arxiv.org/html/2508.14148v2)

A kept position keeps its original rotary position. The model's forward call therefore takes the kept indices and the full sequence length, and an integration has to carry that change into the attention module. [Rotary embedding with kept indices](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/model/modeling_llada.py#L442-L479)

## Settings that change behavior

`-d=gaussian` turns the pruning on. `-w` is the window length, and masked positions beyond the window are always dropped. `-k` sets how fast the chance of keeping a position falls: the end of the window lies `k` standard deviations out on a normal curve centered on the block's end. `-sc` scales that curve, which raises or lowers the number of kept positions. For LLaDA-8B-Instruct on GSM8K the authors use `-k=4 -sc=2 -w=256`. By the sampler's formula, the position right after the block is then kept with a probability of about 0.8. The paper gives the kept share of the window as 25%. The authors tune the three values for each model and benchmark on a small subset. [Flags](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/params.py#L30-L39) · [Gaussian sampler](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/sampler.py#L40-L62) · [Paper, Appendix A.1](https://arxiv.org/html/2508.14148v2)

The subset is random, and the evaluation script sets a fixed seed before it runs. [Seed](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/eval_llada.py#L442-L444)

Two flags in the authors’ recipes belong to decoding. `-th` selects Fast-dLLM's confidence-aware parallel decoding with that threshold. `-e` ends generation once a decoded block contains an end-of-sequence token, and fills the rest of the response with that token. Every DPad recipe sets `-e`, and most vanilla recipes leave it off. To measure the pruning alone, set `-e` on the baseline too, as one run in the long-sequence script does. [Early termination](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/generate_dropout.py#L171-L177) · [Baseline with early termination](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/scripts/long_seq.sh#L4-L5)

## Select the method upstream

| Checkpoint in the authors’ example | Settings in the example | Entry point |
| --- | --- | --- |
| `GSAI-ML/LLaDA-8B-Instruct`, as `-m=instruct` | GSM8K with four shots, length 256, block size 32, `-d=gaussian -k=4 -sc=2 -w=256 -e` | [LLaDA-8B-Instruct script](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/scripts/main_instruct.sh#L7-L11) |
| `GSAI-ML/LLaDA-1.5`, as `-m=1.5` | The same task with `-k=3 -sc=1.6 -w=256 -e`. The long-sequence script uses one shot and length 1024, and its last run adds `-th=0.9` and `-c` for Prefix | [LLaDA-1.5 script](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/scripts/main_1.5.sh#L7-L11) · [Long-sequence script](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/scripts/long_seq.sh#L7-L20) |
| `Dream-org/Dream-v0-Base-7B`, as `-m=base` under `dream/` | GSM8K with four shots, length 256, block size 32, `-d=gaussian -k=4 -sc=1.6 -w=256 -e` | [Dream script](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/dream/scripts/main_base.sh#L7-L11) |

Each script runs a task four ways: vanilla, DPad, parallel decoding, and the two together. `run.py` turns the flags into an lm-eval launch. Without `-th`, a run takes one step for each response position. `-c` selects Prefix and `-dc` selects Dual. [Run script](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/run.py#L44-L107)

Some limits apply at the pinned revision. The loops build a batch of one sequence. The Dream run script rejects the Instruct checkpoint. `chat.py` imports modules that the repository does not contain, so start from `run.py`. The generation code also has a uniform sampler that keeps a fixed number of window positions. `run.py` forwards the name `random` while the generation function accepts only `uniform`, so that sampler is reachable only by calling the function directly. [Dream checkpoint choice](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/dream/run.py#L35-L41) · [Chat imports](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/chat.py#L21-L23) · [Sampler names](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/llada/generate_dropout.py#L134-L139)

## Use it from the Zoo

```python
cache = zoo.attach(model, "dpad", cache="prefix", window=256, k=4.0, scale=2.0)
```

`cache` selects what runs underneath: `"none"`, `"prefix"` or `"dual"`. `window`, `k` and `scale` are the authors' `-w`, `-k` and `-sc`. `sampler="uniform"` with `preserved=n` keeps a fixed number of window positions. The kept positions are drawn again at each new `active` span. They come from the global generator of the model's device, as in the reference, or from a generator of the request when `seed` is set. Early termination at an end-of-sequence token is a rule of the loop and is not part of the policy. [Policy](../../src/dllm_cache_zoo/methods/dpad.py)

DPad is the first suffix pruning method in the Zoo. The shared forward pass takes the positions to run and the positions whose keys may be read, and DPad passes the kept set as both. On LLaDA-8B-Instruct, with the same seed, the policy reveals the same tokens as the authors' three loops on every checked prompt, and under Prefix with a confidence threshold of 0.9 as well. The check ran with early termination off on both sides and a window of 128, since the reference requires the window to fit in the response length. [Check report](../checks.md) · [Third integrations](../design/method-selection.md#third-integrations)

The repository carries an Apache-2.0 license. [Upstream license](https://github.com/Crys-Chen/DPad/blob/96128883e5cb4e9ab832a2b8d1231f4cdd8ecc04/LICENSE) · [All methods](../../README.md#method-zoo)
