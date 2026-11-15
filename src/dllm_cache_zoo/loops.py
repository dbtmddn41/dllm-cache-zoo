"""Reference generation loops for the examples and the checks.

The Zoo attaches to a loop the host already has, and these are not meant to
replace it. They are the published reference samplers, with the model call
swapped for ``state.forward``, so every method can be run and compared under one
decoding rule.

Sources, each at the revision the method pages pin:

- ``llada_generate``: the blockwise low-confidence sampler of LLaDA (``generate.py`` in
  ML-GSAI/LLaDA, MIT) with the optional confidence threshold of Fast-dLLM
  (``v1/llada/generate.py`` in NVlabs/Fast-dLLM, Apache-2.0).
- ``sliding_window_generate``: the decoding loop of Elastic-Cache without its cache
  (``llada/generate.py`` in VILA-Lab/Elastic-Cache, Apache-2.0).
- ``block_diffusion_generate``: the SDAR generation script (``generate.py`` in
  JetAstra/SDAR, MIT).
- ``sub_block_generate``: the generation function of the Fast-dLLM v2 checkpoint
  (``modeling.py`` in Efficient-Large-Model/Fast_dLLM_v2_7B, Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .request import CacheHandle, Stats


@dataclass
class Generation:
    """Tokens of the full sequence (prompt included) and what the run cost."""

    tokens: torch.Tensor
    #: Model forward passes.
    steps: int
    stats: Stats


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling in float64, as in the LLaDA reference. No-op at temperature 0."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask: torch.Tensor, steps: int) -> torch.Tensor:
    """How many tokens to reveal at each of ``steps`` steps, per row: ``(B, steps)``."""
    total = block_mask.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    cols = torch.arange(steps, device=block_mask.device).unsqueeze(0)
    return base.unsqueeze(1).expand(-1, steps) + (cols < rem.unsqueeze(1)).long()


def select_transfer(
    logits: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
    *,
    num_transfer: torch.Tensor | None,
    threshold: float | None = None,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick tokens to reveal among the masked positions.

    Returns the proposed tokens, the positions to reveal and the confidence of
    every position (``-inf`` where it is not masked). With ``threshold`` every
    masked position at or above it is revealed, and at least the most confident
    one. Otherwise the ``num_transfer`` most confident positions of each row are.
    """
    x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask, x0_p, neg_inf)

    if threshold is not None:
        transfer = mask & (confidence >= threshold)
        best = torch.argmax(confidence, dim=1, keepdim=True)
        transfer = (transfer | torch.zeros_like(transfer).scatter_(1, best, True)) & mask
        return x0, transfer, confidence

    if num_transfer is None:
        raise ValueError("num_transfer is needed when no threshold is set.")
    k = num_transfer.to(torch.long).clamp(min=0).view(-1, 1)
    _, order = torch.sort(confidence, dim=1, descending=True)
    cols = torch.arange(confidence.size(1), device=confidence.device).unsqueeze(0)
    picked = torch.zeros_like(mask, dtype=torch.int8).scatter(1, order, (cols < k).to(torch.int8))
    return x0, picked.bool() & mask, confidence


def _check_progress(steps_taken: int, block_length: int) -> None:
    """Every step reveals at least one position, so a block needs at most its length in steps."""
    if steps_taken > block_length:
        raise RuntimeError(
            f"A block of {block_length} positions is still masked after {steps_taken} steps. "
            "The model keeps predicting its mask token."
        )


@torch.no_grad()
def llada_generate(
    cache: CacheHandle,
    prompt: torch.Tensor,
    *,
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    threshold: float | None = None,
    mask_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Generation:
    """Blockwise masked-diffusion sampling for bidirectional models such as LLaDA.

    ``prompt`` is ``(B, P)``. The response of ``gen_length`` positions is filled
    block by block, left to right. Without ``threshold`` each block takes
    ``steps / num_blocks`` steps and reveals an equal share of it at each one.
    With ``threshold`` (confidence-aware parallel decoding) a step reveals every
    masked position of the block whose confidence reaches it.

    ``remasking="random_order"`` draws the unmasking order of each block before
    its first step, from ``generator``. The positions of each step are then known
    ahead of the forward pass and are announced to the cache as ``reveal``, which
    dKV-Cache-Greedy needs.
    """
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be a multiple of block_length.")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be a multiple of the number of blocks.")
    steps_per_block = steps // num_blocks

    device = prompt.device
    batch, prompt_len = prompt.shape
    with cache.request(prompt_length=prompt_len, mask_token_id=mask_token_id) as state:
        mask_id = state.mask_token_id
        x = torch.full((batch, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = prompt
        nfe = 0
        for b in range(num_blocks):
            start = prompt_len + b * block_length
            end = start + block_length
            num_transfer = get_num_transfer_tokens(x[:, start:end] == mask_id, steps_per_block)
            fixed_order = remasking == "random_order"
            if fixed_order:
                noise = torch.rand((batch, block_length), generator=generator, device=device)
                rank = noise.argsort(dim=-1, descending=True).argsort(dim=-1)
                done = torch.cumsum(num_transfer, dim=1)
            i = 0
            while True:
                block = x[:, start:end]
                mask = block == mask_id
                reveal = None
                if fixed_order:
                    j = min(i, steps_per_block - 1)
                    before = done[:, j - 1 : j] if j > 0 else torch.zeros_like(done[:, :1])
                    planned = (rank >= before) & (rank < done[:, j : j + 1]) & mask
                    reveal = torch.zeros_like(x, dtype=torch.bool)
                    reveal[:, start:end] = planned
                logits = state.forward(x, active=(start, end), output="active", reveal=reveal)
                nfe += 1
                if fixed_order:
                    x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
                    x[:, start:end] = torch.where(planned, x0, block)
                    i += 1
                    if not (x[:, start:end] == mask_id).any():
                        break
                    _check_progress(i, block_length)
                    continue
                quota = (
                    None if threshold is not None else num_transfer[:, min(i, steps_per_block - 1)]
                )
                x0, transfer, confidence = select_transfer(
                    logits,
                    mask,
                    block,
                    num_transfer=quota,
                    threshold=threshold,
                    temperature=temperature,
                    remasking=remasking,
                )
                x[:, start:end] = torch.where(transfer, x0, block)
                full_conf = torch.full(x.shape, -torch.inf, dtype=confidence.dtype, device=device)
                full_conf[:, start:end] = confidence
                state.observe(x, confidence=full_conf)
                i += 1
                if not (x[:, start:end] == mask_id).any():
                    break
                _check_progress(i, block_length)
        return Generation(tokens=x, steps=nfe, stats=state.stats)


@torch.no_grad()
def sliding_window_generate(
    cache: CacheHandle,
    prompt: torch.Tensor,
    *,
    gen_length: int = 512,
    window_length: int = 16,
    threshold: float | None = 0.9,
    tokens_per_step: int = 1,
    eos_token_id: int | None = None,
    mask_token_id: int | None = None,
) -> Generation:
    """Sliding-window decoding for one sequence, as in the Elastic-Cache recipe.

    The window is the first ``window_length`` masked positions. Each step reveals
    the window positions whose confidence reaches ``threshold`` (at least the
    most confident one), or the ``tokens_per_step`` most confident when
    ``threshold`` is None. Once an end-of-sequence token is revealed, masked
    positions after it are no longer decoded, and they are filled with that token
    at the end.
    """
    if prompt.size(0) != 1:
        raise ValueError("sliding_window_generate decodes one sequence at a time.")
    device = prompt.device
    prompt_len = prompt.size(1)
    with cache.request(prompt_length=prompt_len, mask_token_id=mask_token_id) as state:
        mask_id = state.mask_token_id
        x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = prompt
        masked = torch.arange(prompt_len, prompt_len + gen_length, device=device)
        seen_eos = False
        nfe = 0
        while masked.numel() > 0:
            if nfe > gen_length:
                raise RuntimeError(
                    f"{gen_length} positions are still masked after {nfe} steps. "
                    "The model keeps predicting its mask token."
                )
            window = masked[:window_length]
            span = (int(window[0]), int(window[-1]) + 1)
            logits = state.forward(x, active=span, output="active")
            nfe += 1
            logits = logits[:, window - span[0]]
            p = F.softmax(logits.to(torch.float64), dim=-1)
            conf, x0 = torch.max(p, dim=-1)
            if threshold is not None:
                keep = conf[0] >= min(threshold, float(conf.max()))
            else:
                top = torch.topk(conf[0], k=min(tokens_per_step, conf.size(1))).indices
                keep = torch.zeros_like(conf[0], dtype=torch.bool)
                keep[top] = True
            revealed = window[keep]
            x[0, revealed] = x0[0, keep]
            masked = masked[~torch.isin(masked, revealed)]
            if eos_token_id is not None and not seen_eos:
                eos_at = revealed[x0[0, keep] == eos_token_id]
                if eos_at.numel() > 0:
                    seen_eos = True
                    masked = masked[masked <= int(eos_at.min())]
        if eos_token_id is not None:
            x[x == mask_id] = eos_token_id
        return Generation(tokens=x, steps=nfe, stats=state.stats)


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    return torch.where(logits < values[..., -1, None], torch.full_like(logits, -torch.inf), logits)


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    drop = cumulative > p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    mask = torch.scatter(torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, drop)
    return logits.masked_fill(mask, -torch.inf)


def sample_tokens(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per position and return it with its probability."""
    shape = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = _top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = _top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token.view(*shape), torch.gather(probs, -1, token).view(*shape)


@torch.no_grad()
def block_diffusion_generate(
    cache: CacheHandle,
    prompt: torch.Tensor,
    *,
    mask_token_id: int,
    gen_length: int = 128,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    stop_token_ids: tuple[int, ...] = (),
) -> Generation:
    """Blockwise generation for block diffusion models such as SDAR.

    Blocks are aligned to multiples of ``block_length`` from position 0, so the
    last prompt tokens share a block with the first generated ones. Each block
    takes up to ``denoising_steps`` steps. ``remasking`` is ``"sequential"``,
    ``"low_confidence_static"`` or ``"low_confidence_dynamic"`` (reveal every
    position above ``confidence_threshold``, or the scheduled number).
    """
    device = prompt.device
    batch, prompt_len = prompt.shape
    num_blocks = (prompt_len + gen_length + block_length - 1) // block_length
    total = num_blocks * block_length
    x = torch.full((batch, total), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt
    quota = torch.full((denoising_steps,), block_length // denoising_steps, dtype=torch.long)
    quota[: block_length % denoising_steps] += 1

    with cache.request(prompt_length=prompt_len, mask_token_id=mask_token_id) as state:
        nfe = 0
        for b in range(prompt_len // block_length, num_blocks):
            start, end = b * block_length, (b + 1) * block_length
            for step in range(denoising_steps + 1):
                cur = x[:, start:end]
                mask = cur == mask_token_id
                if not mask.any():
                    break
                logits = state.forward(x, active=(start, end), output="active")
                nfe += 1
                x0, x0_p = sample_tokens(logits, temperature, top_k, top_p)
                n = int(quota[step])
                transfer = torch.zeros_like(mask)
                if remasking == "sequential":
                    for j in range(batch):
                        first = int(mask[j].nonzero()[0])
                        transfer[j, first : first + n] = True
                elif remasking in ("low_confidence_static", "low_confidence_dynamic"):
                    confidence = torch.where(mask, x0_p, -torch.inf)
                    for j in range(batch):
                        high = confidence[j] > confidence_threshold
                        if remasking == "low_confidence_dynamic" and high.sum() >= n:
                            transfer[j] = high
                        else:
                            transfer[j, torch.topk(confidence[j], n).indices] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")
                x[:, start:end] = torch.where(transfer, x0, cur)
            if stop_token_ids and any(
                bool((x[:, prompt_len:] == stop).any()) for stop in stop_token_ids
            ):
                break
        return Generation(tokens=x, steps=nfe, stats=state.stats)


def _sample_top_p(
    logits: torch.Tensor, top_p: float, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens and the distribution they came from, as the Fast-dLLM v2 sampler computes them."""
    if temperature <= 0:
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    remove = torch.cumsum(sorted_probs, dim=-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    drop = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
    probs = probs.masked_fill(drop, 0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    tokens = torch.multinomial(probs.flatten(0, -2), num_samples=1).view(probs.shape[:-1])
    return tokens, probs


@torch.no_grad()
def sub_block_generate(
    cache: CacheHandle,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    block_length: int = 32,
    sub_block_length: int = 8,
    threshold: float = 1.0,
    top_p: float = 0.95,
    temperature: float = 0.0,
    stop_token_id: int | None = None,
    mask_token_id: int | None = None,
) -> Generation:
    """Block diffusion with sub-blocks for one sequence, as in Fast-dLLM v2.

    The response is generated block by block, and each block sub-block by
    sub-block. A step reveals the positions of the sub-block whose probability
    exceeds ``threshold``, and at least the most probable one. The first token of
    a block comes from the last position of the block before it. Generation stops
    after ``max_new_tokens // block_length`` blocks, or once ``stop_token_id`` is
    revealed with no masked position before it.

    The returned tokens run to the end of the last block and can still hold
    masks after a stop token.
    """
    if prompt.size(0) != 1:
        raise ValueError("sub_block_generate decodes one sequence at a time.")
    if block_length % sub_block_length != 0:
        raise ValueError("block_length must be a multiple of sub_block_length.")
    device = prompt.device
    prompt_len = prompt.size(1)
    first_block = prompt_len // block_length
    num_blocks = max_new_tokens // block_length
    total = (first_block + num_blocks) * block_length

    with cache.request(prompt_length=prompt_len, mask_token_id=mask_token_id) as state:
        mask_id = state.mask_token_id
        x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = prompt
        nfe = 0

        def stopped(region_start: int, region_end: int) -> bool:
            if stop_token_id is None:
                return False
            hits = (x[0, region_start:region_end] == stop_token_id).nonzero()
            if hits.numel() == 0:
                return False
            return not bool((x[0, region_start : region_start + int(hits[0])] == mask_id).any())

        for b in range(first_block, first_block + num_blocks):
            start, end = b * block_length, (b + 1) * block_length
            if x[0, start] == mask_id:
                # The first position of a block is predicted from the block before it.
                span = (start, start + sub_block_length)
                logits = state.forward(x, active=span, output="active")
                nfe += 1
                x[0, start] = logits[0, 0].argmax()
            if stop_token_id is not None and bool(
                (x[0, prompt_len : start + 1] == stop_token_id).any()
            ):
                break
            region = max(prompt_len, start + 1)
            for sub_start in range(start, end, sub_block_length):
                sub_end = sub_start + sub_block_length
                while True:
                    mask = x[:, sub_start:sub_end] == mask_id
                    if not mask.any() or stopped(region, end):
                        break
                    logits = state.forward(x, active=(sub_start, sub_end), output="active")
                    nfe += 1
                    x1, probs = _sample_top_p(logits, top_p, temperature)
                    x1_p = torch.gather(probs, -1, x1.unsqueeze(-1)).squeeze(-1)
                    x1_p = torch.where(mask, x1_p, -torch.inf)
                    unmask = x1_p > threshold
                    unmask[torch.arange(x1.size(0)), x1_p.argmax(dim=-1)] = True
                    unmask &= mask
                    x[:, sub_start:sub_end] = torch.where(unmask, x1, x[:, sub_start:sub_end])
            if stopped(region, end):
                break
        return Generation(tokens=x, steps=nfe, stats=state.stats)
