"""Compare a Zoo method with the authors' implementation on a released checkpoint.

Each comparison runs the authors' own generation loop twice on the same prompts:

1. as published, with the authors' model class and cache code;
2. with the model call replaced by the Zoo (``state.forward``) on the stock
   checkpoint, using the authors' uncached loop, so the sampler is theirs.

The two runs should reveal the same tokens. This is a correctness check on a
handful of prompts, not a benchmark: it reports agreement and no timings.

    python checks/compare_upstream.py --case fast-dllm-prefix --root /work/upstream/fast-dllm

``--root`` is a checkout of the upstream repository at the revision named on
the method's page. Only one upstream is imported per process, because several
of them ship modules with the same names.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import dllm_cache_zoo as zoo

PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per "
    "hour. How many kilometers can she run in 8 hours?",
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
]


class ZooModel:
    """Stands in for the model inside an upstream loop and answers through the Zoo.

    The active block is read from the tokens: blocks fill left to right, so the
    first masked position lies in the block being decoded.
    """

    def __init__(self, handle: zoo.CacheHandle, prompt_length: int, block_length: int):
        self.handle = handle
        self.prompt_length = prompt_length
        self.block_length = block_length
        self.state = handle.request(prompt_length=prompt_length)
        self.device = next(handle.model.parameters()).device
        self.config = handle.model.config
        self.calls = 0

    def __call__(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> SimpleNamespace:
        masked = (input_ids == self.state.mask_token_id).any(dim=0).nonzero()
        first = int(masked[0]) if masked.numel() else input_ids.size(1) - 1
        block = (first - self.prompt_length) // self.block_length
        start = self.prompt_length + block * self.block_length
        logits = self.state.forward(input_ids, active=(start, start + self.block_length))
        self.calls += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


@dataclass
class Result:
    case: str
    prompt: int
    settings: dict[str, Any]
    upstream_steps: int | None
    zoo_steps: int
    response_tokens: int
    matching_tokens: int
    identical: bool
    recomputed_share: float


def encode(tokenizer, text: str, device) -> torch.Tensor:
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return torch.tensor(tokenizer(prompt)["input_ids"], device=device).unsqueeze(0)


def load_stock(model_id: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
    return tokenizer, model.to(device).eval()


# --------------------------------------------------------------------- cases


def case_adapter(args, tokenizer, model) -> list[Result]:
    """The adapter's full pass against the model's own forward."""
    results = []
    handle = zoo.attach(model, "none")
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        mask_id = handle.adapter.mask_token_id
        x = torch.full((1, prompt.size(1) + args.gen_length), mask_id, device=model.device)
        x[:, : prompt.size(1)] = prompt
        with torch.no_grad():
            ref = model(x).logits
            with handle.request(prompt_length=prompt.size(1)) as state:
                out = state.forward(x)
        diff = (out.float() - ref.float()).abs().max().item()
        same = int((out.argmax(-1) == ref.argmax(-1)).sum())
        print(
            f"[adapter] prompt {i}: max |logit diff| = {diff:.3e}, argmax equal {same}/{x.numel()}"
        )
        results.append(
            Result(
                "adapter",
                i,
                {"max_abs_logit_diff": diff},
                None,
                1,
                x.numel(),
                same,
                same == x.numel(),
                1.0,
            )
        )
    return results


def _fast_dllm(args, tokenizer, model, method: str) -> list[Result]:
    sys.path.insert(0, str(Path(args.root) / "v1" / "llada"))
    import generate as up  # type: ignore[import-not-found]
    from model.modeling_llada import LLaDAModelLM  # type: ignore[import-not-found]

    up_model = LLaDAModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    up_model = up_model.to(model.device).eval()
    up_fn = {
        "none": up.generate,
        "fast-dllm-prefix": up.generate_with_prefix_cache,
        "fast-dllm-dual": up.generate_with_dual_cache,
    }[method]
    kw = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        remasking="low_confidence",
        threshold=args.threshold,
    )
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        x_up, nfe_up = up_fn(up_model, prompt, **kw)
        shim = ZooModel(zoo.attach(model, method), prompt.size(1), args.block_length)
        x_zoo, _ = up.generate(shim, prompt, **kw)
        results.append(_result(method, i, kw, x_up, x_zoo, prompt.size(1), nfe_up, shim))
    return results


def case_fast_dllm_none(args, tokenizer, model):
    return _fast_dllm(args, tokenizer, model, "none")


def case_fast_dllm_prefix(args, tokenizer, model):
    return _fast_dllm(args, tokenizer, model, "fast-dllm-prefix")


def case_fast_dllm_dual(args, tokenizer, model):
    return _fast_dllm(args, tokenizer, model, "fast-dllm-dual")


def case_dkv_cache_decode(args, tokenizer, model) -> list[Result]:
    sys.path.insert(0, str(Path(args.root)))
    from generation_utils import llada_dkv_cache_decode as up  # type: ignore[import-not-found]

    up_model = up.LLaDAModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    up_model = up_model.to(model.device).eval()
    kw = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        x_up = up.generate(
            up_model,
            tokenizer,
            prompt,
            enable_cache=True,
            cache_reloading_step=args.refresh_interval,
            **kw,
        )
        handle = zoo.attach(model, "dkv-cache-decode", refresh_interval=args.refresh_interval)
        shim = ZooModel(handle, prompt.size(1), args.block_length)
        x_zoo = up.generate(shim, tokenizer, prompt, enable_cache=False, **kw)
        settings = {**kw, "refresh_interval": args.refresh_interval}
        results.append(
            _result("dkv-cache-decode", i, settings, x_up, x_zoo, prompt.size(1), None, shim)
        )
    return results


def case_dllm_cache(args, tokenizer, model) -> list[Result]:
    sys.path.insert(0, str(Path(args.root)))
    from dllm_cache import logout_cache_LLaDA, register_cache_LLaDA  # type: ignore[import-not-found]
    from dllm_cache.cache import dLLMCache, dLLMCacheConfig  # type: ignore[import-not-found]
    from utils import generate as up_generate  # type: ignore[import-not-found]

    settings = dict(
        prompt_interval_steps=args.prompt_interval,
        gen_interval_steps=args.response_interval,
        transfer_ratio=args.transfer_ratio,
    )
    kw = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        mask = torch.ones_like(prompt)
        handle = zoo.attach(
            model,
            "dllm-cache",
            prompt_interval=args.prompt_interval,
            response_interval=args.response_interval,
            transfer_ratio=args.transfer_ratio,
        )
        shim = ZooModel(handle, prompt.size(1), args.block_length)
        y_zoo = up_generate(prompt, mask, shim, **kw)

        dLLMCache.new_instance(**asdict(dLLMCacheConfig(**settings)))
        register_cache_LLaDA(model, "model.transformer.blocks")
        try:
            y_up = up_generate(prompt, mask, model, **kw)
        finally:
            logout_cache_LLaDA(model, "model.transformer.blocks")
        results.append(_result("dllm-cache", i, {**kw, **settings}, y_up, y_zoo, 0, None, shim))
    return results


def case_elastic_cache(args, tokenizer, model) -> list[Result]:
    sys.path.insert(0, str(Path(args.root) / "llada"))
    import generate as up  # type: ignore[import-not-found]
    from model.modeling_llada import LLaDAModelLM  # type: ignore[import-not-found]

    from dllm_cache_zoo.loops import sliding_window_generate

    up_model = LLaDAModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    up_model = up_model.to(model.device).eval()
    eos_id = 126081
    kw = dict(
        gen_length=args.gen_length,
        window_length=args.window_length,
        threshold=args.threshold if args.threshold is not None else 0.9,
    )
    settings = dict(gamma=args.gamma, track_num=args.track_num, block_caching=True)
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        x_up, nfe_up, _ = up.generate_with_elastic_cache(
            up_model, prompt, eos_id=eos_id, **kw, **settings
        )
        gen = sliding_window_generate(
            zoo.attach(model, "elastic-cache", **settings), prompt, eos_token_id=eos_id, **kw
        )
        results.append(
            _tokens_result(
                "elastic-cache",
                i,
                {**kw, **settings},
                x_up,
                gen.tokens,
                prompt.size(1),
                nfe_up,
                gen.steps,
                gen.stats.recomputed_share,
            )
        )
    return results


def _dpad(args, tokenizer, model, cache: str) -> list[Result]:
    sys.path.insert(0, str(Path(args.root) / "llada"))
    import generate_dropout as up  # type: ignore[import-not-found]
    from model.modeling_llada import LLaDAModelLM  # type: ignore[import-not-found]

    from dllm_cache_zoo.loops import llada_generate

    up_model = LLaDAModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    up_model = up_model.to(model.device).eval()
    up_fn = {
        "none": up.generate,
        "prefix": up.generate_with_prefix_cache,
        "dual": up.generate_with_dual_cache,
    }[cache]
    kw = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        remasking="low_confidence",
        threshold=args.threshold,
    )
    dpad = dict(window=args.window, k=args.k, scale=args.scale)
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        torch.manual_seed(args.seed)
        x_up, nfe_up = up_fn(
            up_model,
            prompt,
            dropout="gaussian",
            sigma=args.k,
            scale=args.scale,
            window=args.window,
            early_termination=False,
            **kw,
        )
        torch.manual_seed(args.seed)
        gen = llada_generate(zoo.attach(model, "dpad", cache=cache, **dpad), prompt, **kw)
        results.append(
            _tokens_result(
                f"dpad-{cache}",
                i,
                {**kw, **dpad, "seed": args.seed},
                x_up,
                gen.tokens,
                prompt.size(1),
                nfe_up,
                gen.steps,
                gen.stats.recomputed_share,
            )
        )
    return results


def case_dpad_none(args, tokenizer, model):
    return _dpad(args, tokenizer, model, "none")


def case_dpad_prefix(args, tokenizer, model):
    return _dpad(args, tokenizer, model, "prefix")


def case_dpad_dual(args, tokenizer, model):
    return _dpad(args, tokenizer, model, "dual")


class SparseShim(ZooModel):
    """Answers the Sparse-dLLM loop, which passes only the block after its second step."""

    def __call__(self, model_input, position_offset, cache_state, customcache):  # type: ignore[override]
        if cache_state != 2:
            self.x = model_input.clone()
        else:
            self.x[:, position_offset : position_offset + self.block_length] = model_input
        active = (position_offset, position_offset + self.block_length)
        logits = self.state.forward(self.x, active=active)
        self.calls += 1
        if cache_state == 2:
            logits = logits[:, active[0] : active[1]]
        return SimpleNamespace(logits=logits)


def case_sparse_dllm(args, tokenizer, model) -> list[Result]:
    sys.path.insert(0, str(Path(args.root) / "opencompass" / "models"))
    from sparse_dllm import llada_generate as up  # type: ignore[import-not-found]
    from sparse_dllm.configuration_llada import LLaDAConfig  # type: ignore[import-not-found]
    from sparse_dllm.modeling_llada import LLaDAModelLM  # type: ignore[import-not-found]

    config = LLaDAConfig.from_pretrained(args.model)
    config.block_len = args.block_length
    config.kernel_size = args.kernel_size
    config.keep_ratio = args.keep_ratio
    up_model = LLaDAModelLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16)
    up_model = up_model.to(model.device).eval()
    kw = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    settings = dict(keep_ratio=args.keep_ratio, kernel_size=args.kernel_size)
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        x_up = up.generate(up_model, prompt, **kw)
        shim = SparseShim(
            zoo.attach(model, "sparse-dllm", **settings), prompt.size(1), args.block_length
        )
        shim.config = config
        x_zoo = up.generate(shim, prompt, **kw)
        results.append(
            _result("sparse-dllm", i, {**kw, **settings}, x_up, x_zoo, prompt.size(1), None, shim)
        )
    return results


class GreedyShim(ZooModel):
    """Answers the dKV-Cache-Greedy loop and announces the positions it will reveal.

    The loop draws its unmasking order with ``torch.rand`` at the start of each
    block. ``draws`` receives those tensors, and the order is rebuilt from them.
    """

    def __init__(self, handle, prompt_length, block_length, steps_per_block, draws):
        super().__init__(handle, prompt_length, block_length)
        self.steps_per_block = steps_per_block
        self.draws = draws
        self.block = -1

    def __call__(self, input_ids, *args, **kwargs):  # type: ignore[override]
        mask_id = self.state.mask_token_id
        first = int((input_ids == mask_id).any(dim=0).nonzero()[0])
        block = (first - self.prompt_length) // self.block_length
        start = self.prompt_length + block * self.block_length
        end = start + self.block_length
        if block != self.block:
            self.block, self.i = block, 0
            scores = self.draws[-1].clone()
            scores[:, end:] = -float("inf")
            scores[:, :start] = -float("inf")
            self.order = torch.sort(scores, descending=True).indices
            total = (input_ids[:, start:end] == mask_id).sum(dim=1)
            base, rem = total // self.steps_per_block, total % self.steps_per_block
            quota = (
                base[:, None]
                + (
                    torch.arange(self.steps_per_block, device=base.device)[None] < rem[:, None]
                ).long()
            )
            self.bounds = torch.cat([torch.zeros_like(quota[:, :1]), quota.cumsum(1)], dim=1)
        reveal = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(input_ids.size(0)):
            lo, hi = int(self.bounds[b, self.i]), int(self.bounds[b, self.i + 1])
            reveal[b, self.order[b, lo:hi]] = True
        logits = self.state.forward(input_ids, active=(start, end), reveal=reveal)
        self.i += 1
        self.calls += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


def case_dkv_cache_greedy(args, tokenizer, model) -> list[Result]:
    sys.path.insert(0, str(Path(args.root)))
    from generation_utils import llada_dkv_cache_greedy as up  # type: ignore[import-not-found]

    up_model = up.LLaDAModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    up_model = up_model.to(model.device).eval()
    kw = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="random",
        window_size=args.greedy_window,
    )
    settings = dict(refresh_interval=args.refresh_interval, window=args.greedy_window)
    steps_per_block = args.steps // (args.gen_length // args.block_length)
    results = []
    for i, text in enumerate(PROMPTS):
        prompt = encode(tokenizer, text, model.device)
        torch.manual_seed(args.seed)
        x_up = up.generate(
            up_model,
            tokenizer,
            prompt,
            enable_cache=True,
            cache_reloading_step=args.refresh_interval,
            **kw,
        )

        draws: list[torch.Tensor] = []
        real_rand = torch.rand

        def recording_rand(*a, _rand=real_rand, _draws=draws, **k):
            out = _rand(*a, **k)
            _draws.append(out)
            return out

        shim = GreedyShim(
            zoo.attach(model, "dkv-cache-greedy", **settings),
            prompt.size(1),
            args.block_length,
            steps_per_block,
            draws,
        )
        torch.manual_seed(args.seed)
        torch.rand = recording_rand  # type: ignore[assignment]
        try:
            x_zoo = up.generate(shim, tokenizer, prompt, enable_cache=False, **kw)
        finally:
            torch.rand = real_rand  # type: ignore[assignment]
        results.append(
            _result(
                "dkv-cache-greedy",
                i,
                {**kw, **settings, "seed": args.seed},
                x_up,
                x_zoo,
                prompt.size(1),
                None,
                shim,
            )
        )
    return results


def _tokens_result(case, i, settings, x_up, x_zoo, prompt_len, nfe_up, nfe_zoo, share) -> Result:
    up_resp, zoo_resp = x_up[:, prompt_len:], x_zoo[:, prompt_len:]
    same = int((up_resp == zoo_resp).sum())
    total = up_resp.numel()
    result = Result(
        case,
        i,
        {k: v for k, v in settings.items() if v is not None},
        nfe_up,
        nfe_zoo,
        total,
        same,
        same == total,
        share,
    )
    print(
        f"[{case}] prompt {i}: {same}/{total} response tokens equal, "
        f"upstream steps {nfe_up}, zoo steps {nfe_zoo}, recomputed share {share:.3f}"
    )
    return result


def _result(case, i, settings, x_up, x_zoo, prompt_len, nfe_up, shim) -> Result:
    up_resp, zoo_resp = x_up[:, prompt_len:], x_zoo[:, prompt_len:]
    same = int((up_resp == zoo_resp).sum())
    total = up_resp.numel()
    result = Result(
        case,
        i,
        {k: v for k, v in settings.items() if v is not None},
        nfe_up,
        shim.calls,
        total,
        same,
        same == total,
        shim.state.stats.recomputed_share,
    )
    print(
        f"[{case}] prompt {i}: {same}/{total} response tokens equal, "
        f"upstream steps {nfe_up}, zoo steps {shim.calls}, "
        f"recomputed share {result.recomputed_share:.3f}"
    )
    return result


CASES = {
    "adapter": case_adapter,
    "fast-dllm-none": case_fast_dllm_none,
    "fast-dllm-prefix": case_fast_dllm_prefix,
    "fast-dllm-dual": case_fast_dllm_dual,
    "dkv-cache-decode": case_dkv_cache_decode,
    "dllm-cache": case_dllm_cache,
    "elastic-cache": case_elastic_cache,
    "dpad-none": case_dpad_none,
    "dpad-prefix": case_dpad_prefix,
    "dpad-dual": case_dpad_dual,
    "sparse-dllm": case_sparse_dllm,
    "dkv-cache-greedy": case_dkv_cache_greedy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--root", help="upstream checkout (not needed for --case adapter)")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--prompt-interval", type=int, default=100)
    parser.add_argument("--response-interval", type=int, default=7)
    parser.add_argument("--transfer-ratio", type=float, default=0.25)
    parser.add_argument("--window-length", type=int, default=16, help="elastic-cache window")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--track-num", type=int, default=1)
    parser.add_argument("--window", type=int, default=256, help="dpad suffix window")
    parser.add_argument("--k", type=float, default=4.0)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--greedy-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", help="write the results as JSON to this file")
    args = parser.parse_args()

    tokenizer, model = load_stock(args.model, args.device)
    results = CASES[args.case](args, tokenizer, model)
    ok = all(r.identical for r in results)
    print(f"[{args.case}] {'IDENTICAL' if ok else 'DIFFERENT'} on {len(results)} prompts")
    if args.out:
        payload = {
            "case": args.case,
            "model": args.model,
            "torch": torch.__version__,
            "zoo": zoo.__version__,
            "results": [asdict(r) for r in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
