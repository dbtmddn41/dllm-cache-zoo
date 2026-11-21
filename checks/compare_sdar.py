"""Check the completed-block cache on a released SDAR checkpoint.

Three comparisons, on a handful of prompts:

1. Logits of the active block: the Zoo's block cache against a pass of the
   stock model over the whole prefix with the block attention mask, and against
   the Zoo's own uncached pass. A block-causal model makes the cache exact, so
   the differences should be at the level of bfloat16 rounding.
2. Tokens: the SDAR generation script (its own cache) against the Zoo's
   reference loop with ``block-cache``, greedy and seeded.

    python checks/compare_sdar.py --root /work/upstream/sdar

Correctness only: no timings are taken. The stock SDAR model code needs
flash-attn to import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import block_diffusion_generate

PROMPTS = [
    "If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, "
    "find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n",
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="checkout of JetAstra/SDAR")
    parser.add_argument("--model", default="JetLM/SDAR-1.7B-Chat")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
        )
        .to("cuda")
        .eval()
    )
    mask_id = tokenizer(tokenizer.mask_token)["input_ids"][0]
    block = args.block_length
    # The released model runs its unmasked attention through flash-attn, which takes
    # half precision only. Its masked forward works in float32, so the logits are compared
    # in both precisions and the SDAR script, which decodes unmasked, in bfloat16 only.
    use_script = args.dtype != "float32"

    sys.path.insert(0, str(Path(args.root)))
    import generate as up  # type: ignore[import-not-found]

    results = []
    for i, text in enumerate(PROMPTS):
        chat = tokenizer.apply_chat_template(
            [dict(role="user", content=text)], add_generation_prompt=True, tokenize=False
        )
        enc = tokenizer([chat], return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        prompt = enc["input_ids"]
        prompt_len = prompt.size(1)

        # ---- 1. logits of each active block during a greedy generation
        total = ((prompt_len + args.gen_length + block - 1) // block) * block
        x = torch.full((1, total), mask_id, dtype=torch.long, device="cuda")
        x[:, :prompt_len] = prompt
        blocks = torch.arange(total, device="cuda") // block
        full_mask = (blocks[None, :] <= blocks[:, None]).unsqueeze(0)
        cached = zoo.attach(model, "block-cache").request(prompt_len, mask_token_id=mask_id)
        plain = zoo.attach(model, "none", block_length=block).request(
            prompt_len, mask_token_id=mask_id
        )
        d_stock = d_plain = d_base = 0.0
        argmax_same = argmax_total = 0
        with torch.no_grad():
            for start in range((prompt_len // block) * block, total, block):
                end = start + block
                for _ in range(2):
                    out = cached.forward(x, active=(start, end), output="active").float()
                    base = plain.forward(x, active=(start, end), output="active").float()
                    ref = model(
                        x[:, :end],
                        attention_mask=full_mask[:, :end, :end],
                        position_ids=torch.arange(end, device="cuda").unsqueeze(0),
                    ).logits
                    ref = ref[:, start:end].float()
                    d_stock = max(d_stock, (out - ref).abs().max().item())
                    d_plain = max(d_plain, (out - base).abs().max().item())
                    d_base = max(d_base, (base - ref).abs().max().item())
                    argmax_same += int((out.argmax(-1) == ref.argmax(-1)).sum())
                    argmax_total += out.shape[1]
                    masked = (x[0, start:end] == mask_id).nonzero().flatten()
                    if masked.numel():
                        x[0, start + masked[0]] = ref[0, masked[0]].argmax()
                still = x[0, start:end] == mask_id
                x[0, start:end][still] = ref[0].argmax(-1)[still]
        scale = ref.abs().max().item()
        cached.close()
        plain.close()

        # ---- 2. tokens of the upstream script against the Zoo loop
        tokens = {"greedy": (0, 0), "sampled": (0, 0)}
        for strategy, top_k, temperature in (
            ("low_confidence_static", 1, 1.0),
            ("low_confidence_dynamic", 0, 1.0),
        ):
            if not use_script:
                break
            kw = dict(
                gen_length=args.gen_length,
                block_length=block,
                denoising_steps=args.denoising_steps,
                temperature=temperature,
                top_k=top_k,
            )
            torch.manual_seed(1234)
            y_up = up.block_diffusion_generate(
                model, enc, mask_id=mask_id, remasking_strategy=strategy, **kw
            )
            torch.manual_seed(1234)
            y_zoo = block_diffusion_generate(
                zoo.attach(model, "block-cache"),
                prompt,
                mask_token_id=mask_id,
                remasking=strategy,
                **kw,
            ).tokens
            name = "greedy" if top_k == 1 else "sampled"
            tokens[name] = (int((y_up == y_zoo).sum()) - prompt_len, y_up.numel() - prompt_len)

        row = dict(
            prompt=i,
            dtype=args.dtype,
            max_abs_diff_cache_vs_stock=d_stock,
            max_abs_diff_cache_vs_uncached=d_plain,
            max_abs_diff_uncached_vs_stock=d_base,
            logit_scale=scale,
            argmax_same=argmax_same,
            argmax_total=argmax_total,
            greedy_tokens_same=tokens["greedy"][0],
            greedy_tokens_total=tokens["greedy"][1],
            sampled_tokens_same=tokens["sampled"][0],
            sampled_tokens_total=tokens["sampled"][1],
        )
        results.append(row)
        print(
            f"[block-cache {args.dtype}] prompt {i}: max |logit diff| cache vs stock "
            f"{d_stock:.4f}, cache vs uncached {d_plain:.4f}, uncached vs stock {d_base:.4f} "
            f"(logits up to {scale:.1f}); argmax equal {argmax_same}/{argmax_total}; tokens "
            f"equal to the SDAR script: greedy {tokens['greedy'][0]}/{tokens['greedy'][1]}, "
            f"sampled {tokens['sampled'][0]}/{tokens['sampled'][1]}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                dict(
                    case="block-cache",
                    model=args.model,
                    torch=torch.__version__,
                    zoo=zoo.__version__,
                    results=results,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
