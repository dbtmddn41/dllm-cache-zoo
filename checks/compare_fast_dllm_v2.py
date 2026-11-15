"""Check the block cache and the sub-block cache on a released Fast-dLLM v2 checkpoint.

The reference is the generation function that ships with the checkpoint, which
has both cache levels built in. Two comparisons, on a handful of prompts:

1. Logits: the Zoo's passes against the model's own forward on the same tokens.
   The model predicts each token from the position before it, and the Zoo
   returns logits aligned with the position they predict.
2. Tokens: ``model.generate`` with and without ``use_block_cache`` against the
   Zoo's reference loop with ``fast-dllm-v2-sub-block``, ``block-cache`` and
   ``none``, greedy.

    python checks/compare_fast_dllm_v2.py --dtype float32

Correctness only: no timings are taken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import sub_block_generate

PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per "
    "hour. How many kilometers can she run in 8 hours?",
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
]

# (name, threshold, use_block_cache in the reference, method in the Zoo)
RUNS = [
    ("none", 1.0, False, "none"),
    ("block-cache", 1.0, False, "block-cache"),
    ("block-cache-th", 0.9, False, "block-cache"),
    ("sub-block", 1.0, True, "fast-dllm-v2-sub-block"),
    ("sub-block-th", 0.9, True, "fast-dllm-v2-sub-block"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--sub-block-length", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
    )
    model = model.to("cuda").eval()
    block, sub = args.block_length, args.sub_block_length
    mask_id = zoo.attach(model, "none").adapter.mask_token_id
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    results = []
    for i, text in enumerate(PROMPTS):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        prompt = tokenizer([chat], return_tensors="pt")["input_ids"].to("cuda")
        prompt_len = prompt.size(1)

        # ---- 1. logits of the block that holds the end of the prompt
        end = (prompt_len // block + 1) * block
        x = torch.full((1, end), mask_id, dtype=torch.long, device="cuda")
        x[:, :prompt_len] = prompt
        first = max(prompt_len, end - block + 1)
        diffs = {}
        with torch.no_grad():
            ref = model(input_ids=x, block_size=block).logits[:, first - 1 : end - 1].float()
            for method in ("none", "block-cache", "fast-dllm-v2-sub-block"):
                with zoo.attach(model, method).request(prompt_len) as state:
                    out = state.forward(x, active=(end - sub, end)).float()
                diffs[method] = (out[:, first:end] - ref).abs().max().item()
        scale = ref.abs().max().item()

        # ---- 2. tokens of the checkpoint's generate against the Zoo loop
        row = dict(prompt=i, dtype=args.dtype, logit_scale=scale, max_abs_logit_diff=diffs, runs={})
        for name, threshold, use_block_cache, method in RUNS:
            y_up = model.generate(
                prompt,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                block_size=block,
                small_block_size=sub,
                threshold=threshold,
                use_block_cache=use_block_cache,
            )[0, prompt_len:]
            gen = sub_block_generate(
                zoo.attach(model, method),
                prompt,
                max_new_tokens=args.max_new_tokens,
                block_length=block,
                sub_block_length=sub,
                threshold=threshold,
                stop_token_id=stop_id,
            )
            y_zoo = gen.tokens[0, prompt_len:]
            stops = (y_zoo == stop_id).nonzero()
            if stops.numel():
                y_zoo = y_zoo[: int(stops[0]) + 1]
            n = min(y_up.numel(), y_zoo.numel())
            same = int((y_up[:n] == y_zoo[:n]).sum())
            row["runs"][name] = dict(
                method=method,
                threshold=threshold,
                reference_tokens=y_up.numel(),
                zoo_tokens=y_zoo.numel(),
                compared=n,
                same=same,
                zoo_steps=gen.steps,
                recomputed_share=gen.stats.recomputed_share,
            )
        results.append(row)
        logits = ", ".join(f"{k} {v:.5f}" for k, v in diffs.items())
        tokens = ", ".join(f"{k} {r['same']}/{r['compared']}" for k, r in row["runs"].items())
        print(
            f"[fast-dllm-v2 {args.dtype}] prompt {i}: max |logit diff| vs model forward: "
            f"{logits} (logits up to {scale:.1f}); tokens equal to model.generate: {tokens}"
        )

    ok = all(r["same"] == r["compared"] for row in results for r in row["runs"].values())
    verdict = "IDENTICAL" if ok else "DIFFERENT"
    print(f"[fast-dllm-v2 {args.dtype}] {verdict} on {len(results)} prompts")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                dict(
                    case="fast-dllm-v2",
                    model=args.model,
                    dtype=args.dtype,
                    torch=torch.__version__,
                    zoo=zoo.__version__,
                    results=results,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
