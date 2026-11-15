"""Put a cache method into a generation loop you already have.

This is the blockwise sampler from the LLaDA repository, kept as it is. The two
places that changed are marked: the request that holds the cache state, and the
call that used to be ``model(x).logits``.

    python examples/own_loop.py --method fast-dllm-dual
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import dllm_cache_zoo as zoo


@torch.no_grad()
def generate(cache, prompt, steps=128, gen_length=128, block_length=32, mask_id=126336):
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
    )
    x[:, : prompt.shape[1]] = prompt
    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    with cache.request(prompt_length=prompt.shape[1]) as state:  # <- added
        for num_block in range(num_blocks):
            start = prompt.shape[1] + num_block * block_length
            end = start + block_length
            for _ in range(steps):
                mask_index = x == mask_id
                logits = state.forward(x, active=(start, end))  # <- was: model(x).logits
                x0 = torch.argmax(logits, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, end:] = -torch.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                # one token per step here; the reference spreads the block over `steps`
                select = torch.topk(confidence[0], k=max(1, block_length // steps)).indices
                x[0, select] = x0[0, select]
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--method", default="fast-dllm-dual")
    parser.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to("cuda").eval()
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = torch.tensor(tokenizer(text)["input_ids"], device="cuda").unsqueeze(0)

    cache = zoo.attach(model, args.method)  # <- added
    out = generate(cache, prompt)
    print(tokenizer.batch_decode(out[:, prompt.shape[1] :], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
