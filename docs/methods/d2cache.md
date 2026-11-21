# d²Cache

Jiang et al. choose, at every decoding step, a small set of positions whose keys and values are updated, and reuse cached states for all other positions. The choice has two stages. Masked positions are ranked by a certainty prior, and the remaining positions are ranked by attention rollout. The authors also describe a decoding rule guided by the same certainty prior, which their codebase enables by default with the cache. [Paper](https://arxiv.org/abs/2509.23094) · [Authors’ usage notes](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L73-L134) · [Implementation](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/src/cache/d2cache.py)

## Settings that change behavior

`current_k` sets how many masked positions are updated and `sigma` shapes the certainty density used to rank them. `rollout_p` sets the top-p ratio for the attention rollout selection. `inflate_w` also selects unselected positions that lie between two selected ones. The usage notes list its default as 4, while the shipped configuration sets 0 and reserves it for parallel or semi-autoregressive decoding. [Hyper-parameters](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L83-L90) · [Shipped configuration](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/configs/cache/d2cache.yaml)

The method reads attention weights, so the authors require the eager attention implementation. Include that requirement in the baseline when timing the method, because the attention implementation also changes the cost of full recomputation. [Attention requirement](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L133-L134)

## Select the method upstream

| Checkpoint in the authors’ example | Decoding | Settings in the example | Entry point |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct, as `llada-inst` | Certainty-prior decoding, on by default | `cache=d2cache`, `rollout_p=0.1`, `current_k=32`, `sigma=10`, eager attention | [Usage example](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L94-L111) |
| Model left at the configured default | Certainty-prior decoding disabled with `generation.sigma=0` | The settings above and `inflate_w=4` | [Parallel decoding example](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/docs/kv_caching.md#L112-L130) |

For a cache-only comparison, use the second form so the baseline and the cached run share one decoding rule. The same codebase also selects Prefix and Dual, dLLM-Cache and a block diffusion cache through its `cache` setting, and its model configurations cover LLaDA-8B, LLaDA-1.5, Dream and SDAR. [Supported models and caches](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/README.md#L47-L64)

The repository carries an Apache-2.0 license. [Upstream license](https://github.com/Kamichanw/d2Cache/blob/79fb5f64864389ba093c29122a21ef0391080312/LICENSE) · [All methods](../../README.md#method-zoo)
