"""Generate with LLaDA and a cache method from the Zoo.

    python examples/llada_generate.py --method fast-dllm-dual
    python examples/llada_generate.py --method dkv-cache-decode --set refresh_interval=4
    python examples/llada_generate.py --method elastic-cache --loop window

The decoding rule is the same for every method, so the outputs can be compared
with ``--method none``. Needs ``pip install "dllm-cache-zoo[hf]"`` and a GPU.
"""

from __future__ import annotations

import argparse
import ast

import torch
from transformers import AutoModel, AutoTokenizer

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import llada_generate, sliding_window_generate


def parse_settings(pairs: list[str]) -> dict:
    settings = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            settings[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            settings[key] = value
    return settings


def main() -> None:
    names = [info.name for info in zoo.methods("bidirectional")]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--method", default="fast-dllm-dual", choices=names)
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="settings of the method, e.g. refresh_interval=4",
    )
    parser.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--threshold", type=float, default=None, help="confidence threshold for parallel decoding"
    )
    parser.add_argument(
        "--loop",
        default="block",
        choices=["block", "window"],
        help="blockwise sampler, or the sliding window of the Elastic-Cache recipe",
    )
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(args.device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = torch.tensor(tokenizer(text)["input_ids"], device=args.device).unsqueeze(0)

    cache = zoo.attach(model, args.method, **parse_settings(args.set))
    print(cache)
    if args.loop == "window":
        out = sliding_window_generate(
            cache,
            prompt,
            gen_length=args.gen_length,
            window_length=args.window_length,
            threshold=args.threshold if args.threshold is not None else 0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        remasking = "random_order" if args.method == "dkv-cache-greedy" else "low_confidence"
        out = llada_generate(
            cache,
            prompt,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            threshold=args.threshold,
            remasking=remasking,
        )

    answer = tokenizer.batch_decode(out.tokens[:, prompt.size(1) :], skip_special_tokens=True)[0]
    print(answer)
    print(
        f"\n{out.steps} forward passes, {out.stats.full_forwards} of them over the whole "
        f"sequence; {out.stats.recomputed_share:.0%} of the positions a full recomputation runs"
    )


if __name__ == "__main__":
    main()
