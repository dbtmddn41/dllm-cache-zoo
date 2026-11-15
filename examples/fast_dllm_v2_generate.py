"""Generate with a Fast-dLLM v2 checkpoint and its hierarchical cache.

    python examples/fast_dllm_v2_generate.py --method fast-dllm-v2-sub-block
    python examples/fast_dllm_v2_generate.py --method block-cache   # exact, no sub-block cache
    python examples/fast_dllm_v2_generate.py --method none          # full recomputation

The loop is the checkpoint's own decoding rule: blocks of 32 refined in
sub-blocks of 8, revealing every position above a confidence threshold.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import sub_block_generate


def main() -> None:
    names = [info.name for info in zoo.methods("block-causal")]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--method", default="fast-dllm-v2-sub-block", choices=names)
    parser.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sub-block-length", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(args.device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer([text], return_tensors="pt")["input_ids"].to(args.device)

    cache = zoo.attach(model, args.method)
    print(cache)
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out = sub_block_generate(
        cache,
        prompt,
        max_new_tokens=args.max_new_tokens,
        block_length=cache.adapter.block_length,
        sub_block_length=args.sub_block_length,
        threshold=args.threshold,
        stop_token_id=stop_id,
    )
    answer = out.tokens[0, prompt.size(1) :]
    stops = (answer == stop_id).nonzero()
    if stops.numel():
        answer = answer[: int(stops[0])]
    print(tokenizer.decode(answer, skip_special_tokens=True))
    print(
        f"\n{out.steps} denoising steps; {out.stats.recomputed_share:.0%} of the positions "
        "a full recomputation runs"
    )


if __name__ == "__main__":
    main()
