# Method catalog

Each entry describes one cache policy. A paper can contain several policies with different refresh rules. All four entries currently have **Reference** status. Model names here describe text backbones in the upstream work. For the planned criteria for integration and benchmarking, see the [project scope](project.md).

| Policy | Where reuse happens | What triggers recomputation | Text backbones in upstream work |
| --- | --- | --- | --- |
| Fast-dLLM v1 prefix | Before the active block | Block boundary, then the active block and suffix on each step | LLaDA, Dream |
| Fast-dLLM v1 dual | On both sides of the active block | Block boundary, then only the active block on each step | LLaDA, Dream |
| dKV-Cache-Decode | At previously decoded token positions | Newly decoded and still-masked positions, with periodic full refreshes | LLaDA, Dream |
| dLLM-Cache | In the prompt and at selected response positions | Separate refresh intervals and changes in response features | LLaDA, Dream |

## Fast-dLLM v1 prefix

A full pass initializes each block. The implementation keeps keys and values before the block, then evaluates the remaining sequence on subsequent steps. Block length determines how often the prefix is rebuilt. Fast-dLLM also introduces confidence-aware parallel decoding. Keep the token-transfer rule fixed to measure the effect of its cache. The linked LLaDA file carries an Apache-2.0 header. [Paper](https://arxiv.org/abs/2505.22618) · [LLaDA prefix implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L132-L207) · [Dream implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/model/generation_utils_block.py) · [License](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/LICENSE)

## Fast-dLLM v1 dual

The same block-initialization pass also supplies keys and values for the masked suffix. During refinement, the model replaces states in the current block. A comparison using a single final block cannot demonstrate a saving from suffix reuse, because there are no subsequent blocks. For an experiment isolating this difference, include a case with multiple generation blocks. [Paper](https://arxiv.org/abs/2505.22618) · [LLaDA dual implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L210-L272) · [Dream implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/dream/model/generation_utils_block.py) · [Apache-2.0 license](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/LICENSE)

## dKV-Cache-Decode

This policy tracks token positions as they leave the masked set. A newly decoded position receives another update before it becomes eligible for reuse. Periodic full passes refresh accumulated states. Record the refresh interval when comparing runs. The paper also describes a Greedy variant with a different position-selection rule. No license file was found in the linked repository revision, so this entry points readers to the authors’ code. [Paper](https://arxiv.org/abs/2505.15781) · [LLaDA Decode implementation](https://github.com/horseee/dKV-Cache/blob/49a76fcc43b744ec2d960137f216e419317138b1/generation_utils/llada_dkv_cache_decode.py) · [Upstream tree and model examples](https://github.com/horseee/dKV-Cache/tree/49a76fcc43b744ec2d960137f216e419317138b1)

## dLLM-Cache

The method uses separate refresh intervals for the prompt and response. Its adaptive response update ranks positions by feature similarity and refreshes a configurable fraction. Record both intervals and the transfer ratio. The scope of this entry is the text models in the paper. The upstream repository includes an Apache-2.0 license. [Paper](https://proceedings.mlr.press/v306/liu26fi.html) · [LLaDA hook](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/dllm_cache/hooks/cache_hook_LLaDA.py) · [Configuration fields](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/dllm_cache/cache/Config.py) · [License](https://github.com/maomaocun/dLLM-cache/blob/17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a/LICENSE)
