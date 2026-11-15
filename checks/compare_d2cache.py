"""Compare the Zoo's d2Cache with the authors' implementation in Research dLLM.

The authors' loop is run once with their model class and cache. At every step
it records the tokens that went in and the positions their cache ran. The Zoo
then replays the same tokens through ``state.forward`` on the stock checkpoint,
so both sides see identical inputs and only the cache differs. Reported per
step: whether the two ran the same positions, and whether the Zoo's logits pick
the token the authors' run revealed.

    python checks/compare_d2cache.py --root /work/upstream/d2cache

Correctness only: no timings are taken. The reference revision never refreshes
its stored confidence, so the Zoo runs with ``refresh_confidence=False`` here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import dllm_cache_zoo as zoo

PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per "
    "hour. How many kilometers can she run in 8 hours?",
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
]


def _stub_lm_eval() -> None:
    """Research dLLM imports its lm-eval wrappers with its models. They are not used here,
    so two empty base classes stand in when lm-eval is not installed."""
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        import types

        for name in ("lm_eval", "lm_eval.api", "lm_eval.api.instance", "lm_eval.api.model"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["lm_eval.api.instance"].Instance = object  # type: ignore[attr-defined]
        sys.modules["lm_eval.api.model"].TemplateLM = object  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="checkout of Kamichanw/d2Cache")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--rollout-p", type=float, default=0.1)
    parser.add_argument("--current-k", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--inflate-w", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stock = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
    )
    stock = stock.to("cuda").eval()
    mask_id = zoo.attach(stock, "none").adapter.mask_token_id
    os.environ["MASK_TOKEN_ID"] = str(mask_id)

    sys.path.insert(0, str(Path(args.root)))
    _stub_lm_eval()
    from src.cache import d2Cache  # type: ignore[import-not-found]
    from src.generation.vanilla import vanilla_generate  # type: ignore[import-not-found]
    from src.models.llada.modeling_llada import LLaDAModelLM  # type: ignore[import-not-found]

    up_model = (
        LLaDAModelLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype), attn_implementation="eager"
        )
        .to("cuda")
        .eval()
    )
    settings = dict(
        rollout_p=args.rollout_p,
        current_k=args.current_k,
        sigma=args.sigma,
        inflate_w=args.inflate_w,
    )

    summary = []
    for i, text in enumerate(PROMPTS):
        messages = [{"role": "user", "content": text}]
        chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = torch.tensor(tokenizer(chat)["input_ids"], device="cuda").unsqueeze(0)
        prompt_len = prompt.size(1)

        cache = d2Cache(up_model.config, **settings)
        trace: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        step_end = cache.on_step_end

        def record(block_mask, frame, delta, cache=cache, trace=trace, step_end=step_end):
            x = torch.cat([frame.prompts, frame.generated_tokens], dim=1).clone()
            ran = None if cache.active_q_mask is None else cache.active_q_mask.clone()
            trace.append((x, ran))
            if ran is None:
                # The pinned revision switches to its decode state before it checks for a
                # query mask, and asserts on the first step. The revision before it skipped
                # the update when there was no mask; an empty mask has the same effect.
                cache.active_q_mask = torch.zeros_like(x, dtype=torch.bool)
                try:
                    return step_end(block_mask, frame, delta)
                finally:
                    cache.active_q_mask = None
            return step_end(block_mask, frame, delta)

        cache.on_step_end = record
        with torch.no_grad():
            vanilla_generate(
                up_model,
                prompt,
                max_new_tokens=args.gen_length,
                block_length=args.block_length,
                num_transfer_tokens=1,
                temperature=0.0,
                sigma=None,
                mask_token_id=mask_id,
                cache=cache,
            )

        handle = zoo.attach(stock, "d2cache", refresh_confidence=False, **settings)
        same_set = same_token = revealed = 0
        iou_sum = 0.0
        with handle.request(prompt_length=prompt_len) as state:
            for t, (x, ran) in enumerate(trace):
                first = int((x[0] == mask_id).nonzero()[0])
                block = (first - prompt_len) // args.block_length
                start = prompt_len + block * args.block_length
                logits = state.forward(x, active=(start, start + args.block_length))
                mine = state.last_computed
                theirs = torch.ones_like(mine) if ran is None else ran.bool()
                inter = (mine & theirs).sum().item()
                union = (mine | theirs).sum().item()
                iou_sum += inter / union
                same_set += int(inter == union)
                if t + 1 < len(trace):
                    nxt = trace[t + 1][0]
                    changed = (nxt != x)[0].nonzero().flatten()
                    revealed += changed.numel()
                    same_token += int((logits[0, changed].argmax(-1) == nxt[0, changed]).sum())
        steps = len(trace)
        row = dict(
            prompt=i,
            steps=steps,
            same_positions=same_set,
            mean_iou=iou_sum / steps,
            revealed=revealed,
            same_token=same_token,
            recomputed_share=state.stats.recomputed_share,
        )
        summary.append(row)
        print(
            f"[d2cache] prompt {i}: same positions on {same_set}/{steps} steps "
            f"(mean IoU {row['mean_iou']:.4f}), Zoo logits pick the revealed token for "
            f"{same_token}/{revealed}, recomputed share {row['recomputed_share']:.3f}"
        )

    ok = all(
        r["same_positions"] == r["steps"] and r["same_token"] == r["revealed"] for r in summary
    )
    print(f"[d2cache] {'IDENTICAL' if ok else 'DIFFERENT'} on {len(summary)} prompts")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                dict(
                    case="d2cache",
                    model=args.model,
                    dtype=args.dtype,
                    settings=settings,
                    torch=torch.__version__,
                    zoo=zoo.__version__,
                    results=summary,
                ),
                indent=2,
            )
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
