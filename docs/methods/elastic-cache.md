# Elastic-Cache

Nguyen-Tri et al. use the model's own attention to decide when and where to refresh. Decoding moves a sliding window across the masked positions and caches the masks far from it. The method tracks the most-attended tokens. When their attention changes past a threshold, it refreshes the cache starting from the deeper layers, where the authors observe the largest drift. [Paper](https://arxiv.org/abs/2510.14973) · [Method overview](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/README.md#L47-L68) · [LLaDA generation loop](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/generate.py#L44-L147)

## Settings that change behavior

`window_length` is the sliding window size. `gamma` is the threshold that triggers a cache update and `track_num` is the number of most-attended tokens that are watched. `block_caching` caches the masked positions far from the window. `threshold` belongs to confidence-aware decoding and is a decoding setting. Hold it fixed across the baseline and the cached run. [Parameters](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/README.md#L107-L116)

## Select the method upstream

| Checkpoint in the authors’ example | Settings in the example | Entry point |
| --- | --- | --- |
| LLaDA-1.5 | Generation length 512, window 16, `threshold=0.9`, `gamma=0.9`, `track_num=1`, block caching on, GSM8K with five shots | [LLaDA GSM8K script](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/llada/eval_gsm8k.sh#L5-L19) |
| Dream-v0-Instruct-7B | Generation length 512, window 32, `threshold=0.9`, `gamma=0.9`, `track_num=1`, block caching on, GSM8K with five shots | [Dream GSM8K script](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/dream/eval_gsm8k.sh#L5-L22) |

Both scripts keep a second checkpoint as a commented alternative: LLaDA-8B-Instruct for LLaDA and Dream-v0-Base-7B for Dream. The README notes that a corrected evaluation gave slightly lower throughput and higher accuracy on GSM8K and MATH than first reported. Measure a local baseline before quoting a speedup. [Authors’ correction note](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/README.md#L72-L74)

The repository carries an Apache-2.0 license. [Upstream license](https://github.com/VILA-Lab/Elastic-Cache/blob/1960d8fc6231205a1ae4ebba3898d475e339f7e1/LICENSE) · [All methods](../../README.md#method-zoo)
