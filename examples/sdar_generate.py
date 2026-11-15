"""Generate with an SDAR block diffusion checkpoint and the completed-block cache.

    python examples/sdar_generate.py --method block-cache
    python examples/sdar_generate.py --method none        # full recomputation, same tokens

The released SDAR model code imports flash-attn, so it has to be installed.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import block_diffusion_generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="JetLM/SDAR-1.7B-Chat")
    parser.add_argument("--method", default="block-cache", choices=["block-cache", "none"])
    parser.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    # SDAR's configuration does not name the mask token, so it comes from the tokenizer.
    mask_id = tokenizer(tokenizer.mask_token)["input_ids"][0]

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer([text], return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt = prompt.to(args.device)

    settings = {"block_length": args.block_length}
    cache = zoo.attach(model, args.method, **settings)
    out = block_diffusion_generate(
        cache,
        prompt,
        mask_token_id=mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        stop_token_ids=(tokenizer.eos_token_id,),
    )
    answer = tokenizer.decode(out.tokens[0, prompt.size(1) :], skip_special_tokens=False)
    print(answer.replace(tokenizer.mask_token, ""))
    print(
        f"\n{out.steps} denoising steps; {out.stats.recomputed_share:.0%} of the positions "
        "a full recomputation runs"
    )


if __name__ == "__main__":
    main()
