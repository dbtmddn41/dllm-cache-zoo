# Roadmap

The Zoo is in the design stage, and no method runs from it yet. These milestones take it to an installable library and then into a serving engine. [Library design](library.md)

The methods named in the first two milestones were chosen to prove the interface on both model families. The Zoo is meant to hold many more. The catalog lists the published methods found so far, and after the release they move into the Zoo one at a time. [Method catalog](../catalog.md)

1. **One model, one interface.** Keep the LLaDA-8B-Instruct PyTorch loop as the correctness reference for full recomputation, Fast-dLLM Prefix, Fast-dLLM Dual, dKV-Cache Decode and dLLM-Cache. Compare each cached variant to the authors’ implementation at its pinned revision. The milestone is met when these policies share one adapter with no method-specific model code, and each has a result measured under the comparison protocol. The pinned dKV-Cache revision has no license file, so its policy is written from the paper. [Upstream variants](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/chat.py#L49-L56) · [dKV-Cache license note](../methods/dkv-cache.md#select-a-variant-upstream)
2. **A second family and newer methods.** Add a block diffusion checkpoint with its block cache as that family's baseline. SDAR-8B-Chat is the first candidate, with LLaDA2.0-mini next. Bring d²Cache and Elastic-Cache to the LLaDA adapter. Then test which approximate policies still save work inside an active block. [Block diffusion caches](../methods/block-diffusion.md) · [Checkpoints](../methods.md#block-diffusion)
3. **An installable release.** Publish the package with a support table of method and checkpoint pairs. Each pair has a host-side example, a configuration and measured results with their raw records. Ask each method's authors to review their entry. From here a new method is one policy file, one configuration and one measured entry. Work through the catalog in that form, starting with methods whose code carries a license. A technical report on the policy formulation can follow so the Zoo can be cited as a whole. [Maintaining the Zoo](maintaining.md)
4. **Serving integration.** Carry one method into an engine that teams already deploy, and validate it under load. [Serving integration](serving.md)

## Scope

| In or out | What |
| --- | --- |
| Implemented | Caches that need no training and act at inference time, on bidirectional masked diffusion models and on block diffusion models. |
| Catalogued for later | Cache eviction, sparse attention, suffix pruning, and caches shared across requests. They enter the Zoo once the policy interface covers context selection. |
| Left to the host | Decoding strategies such as parallel decoding, remasking rules and unmasking orders. The host provides them, and each example and result names the one used. |
| Out of scope | Methods that require training or fine-tuning, quantization, and a serving engine of the Zoo's own. Block diffusion checkpoints are used as released. |

Advertise support for a **method, model revision, and host** together. Each supported combination should link the method's paper and implementation revision, applicable license notices, a minimal host-side example, a reusable configuration, and the checks run on that path. Report sampler restrictions before inference and link matched measurements when available. [Example of a sampler restriction](../methods/dkv-cache.md#greedy)
