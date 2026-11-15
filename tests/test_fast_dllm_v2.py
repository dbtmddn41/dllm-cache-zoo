"""Block diffusion with a next-token head, and the sub-block cache on top of it."""

from __future__ import annotations

import pytest
import torch
from toy import block_attention_mask

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import block_diffusion_generate, sub_block_generate

BLOCK = 8


def _prompt(length: int) -> torch.Tensor:
    return torch.randint(0, 90, (1, length), generator=torch.Generator().manual_seed(length))


def _generate(model, method, prompt, sub_block=2, **kw):
    return sub_block_generate(
        zoo.attach(model, method),
        prompt,
        max_new_tokens=24,
        block_length=BLOCK,
        sub_block_length=sub_block,
        mask_token_id=model.mask_token_id,
        **kw,
    )


def test_adapter_reads_block_size_and_shift(v2):
    adapter = zoo.attach(v2, "none").adapter
    assert (adapter.name, adapter.family) == ("fast-dllm-v2", "block-causal")
    assert (adapter.logit_shift, adapter.block_length) == (1, BLOCK)


@pytest.mark.parametrize("method", ["none", "block-cache", "fast-dllm-v2-sub-block"])
def test_logits_are_aligned_with_the_position_they_predict(v2, method):
    prompt = _prompt(11)
    x = torch.full((1, 24), v2.mask_token_id)
    x[:, :11] = prompt
    ref = v2(x[:, :16], attention_mask=block_attention_mask(16, BLOCK)).logits
    with zoo.attach(v2, method).request(11, mask_token_id=v2.mask_token_id) as state:
        out = state.forward(x, active=(12, 14))
    # position p is predicted by the hidden state at p - 1
    torch.testing.assert_close(out[:, 12:14], ref[:, 11:13], rtol=0, atol=1e-9)


@pytest.mark.parametrize("method", ["block-cache", "fast-dllm-v2-sub-block"])
def test_first_token_of_a_block_comes_from_the_block_before(v2, method):
    prompt = _prompt(16)  # ends on a block boundary
    x = torch.full((1, 32), v2.mask_token_id)
    x[:, :16] = prompt
    ref = v2(prompt, attention_mask=block_attention_mask(16, BLOCK)).logits
    with zoo.attach(v2, method).request(16, mask_token_id=v2.mask_token_id) as state:
        out = state.forward(x, active=(16, 18))
        assert state.last_computed[0].nonzero().flatten().tolist() == [16]
        torch.testing.assert_close(out[:, 16], ref[:, 15], rtol=0, atol=1e-9)
        x[0, 16] = int(out[0, 16].argmax())
        state.forward(x, active=(16, 18))
        assert state.last_computed[0].nonzero().flatten().tolist() == list(range(17, 24))


def test_a_sequence_cannot_start_with_a_mask(v2):
    x = torch.full((1, 8), v2.mask_token_id)
    with zoo.attach(v2, "block-cache").request(0, mask_token_id=v2.mask_token_id) as state:
        with pytest.raises(ValueError, match="first position"):
            state.forward(x, active=(0, 2))


@pytest.mark.parametrize("prompt_len", [5, 8, 16, 19])
def test_block_cache_matches_full_recomputation(v2, prompt_len):
    prompt = _prompt(prompt_len)
    base = _generate(v2, "none", prompt)
    out = _generate(v2, "block-cache", prompt)
    assert torch.equal(out.tokens, base.tokens)
    assert not (out.tokens == v2.mask_token_id).any()
    assert out.stats.recomputed_share < base.stats.recomputed_share


@pytest.mark.parametrize("prompt_len", [5, 16, 19])
def test_one_sub_block_per_block_is_the_block_cache(v2, prompt_len):
    prompt = _prompt(prompt_len)
    base = _generate(v2, "block-cache", prompt, sub_block=BLOCK)
    out = _generate(v2, "fast-dllm-v2-sub-block", prompt, sub_block=BLOCK)
    assert torch.equal(out.tokens, base.tokens)


@pytest.mark.parametrize("threshold", [1.0, 0.3])
def test_sub_block_cache_runs_less_than_the_block_cache(v2, threshold):
    prompt = _prompt(19)
    block = _generate(v2, "block-cache", prompt, threshold=threshold)
    sub = _generate(v2, "fast-dllm-v2-sub-block", prompt, threshold=threshold)
    assert not (sub.tokens == v2.mask_token_id).any()
    assert sub.stats.recomputed_share < block.stats.recomputed_share


def test_sub_block_pass_follows_the_reference_rule(v2):
    prompt = _prompt(16)
    x = torch.full((1, 24), v2.mask_token_id)
    x[:, :16] = prompt
    x[0, 16] = 3  # first token of the block, already revealed
    with zoo.attach(v2, "fast-dllm-v2-sub-block").request(16, mask_token_id=v2.mask_token_id) as s:
        ran = lambda: s.last_computed[0].nonzero().flatten().tolist()  # noqa: E731
        s.forward(x, active=(16, 18))  # the block has not been run: whole block
        assert ran() == list(range(17, 24))
        x[0, 17] = 4
        s.forward(x, active=(16, 18))  # stored block, first position revealed: sub-block only
        assert ran() == [17]
        s.forward(x, active=(18, 20))  # first position of the sub-block is masked: whole block
        assert ran() == list(range(17, 24))
        x[0, 18] = 5
        s.forward(x, active=(18, 20))
        assert ran() == [19]


def test_generation_stops_at_the_stop_token(v2):
    prompt = _prompt(19)
    full = _generate(v2, "block-cache", prompt)
    stop = int(full.tokens[0, 22])
    out = _generate(v2, "block-cache", prompt, stop_token_id=stop)
    assert out.steps < full.steps
    first = int((out.tokens[0, 19:] == stop).nonzero()[0])
    assert not (out.tokens[0, 19 : 19 + first] == v2.mask_token_id).any()


def test_span_must_stay_inside_a_block(v2):
    x = torch.full((1, 24), v2.mask_token_id)
    x[:, :11] = _prompt(11)
    with zoo.attach(v2, "block-cache").request(11, mask_token_id=v2.mask_token_id) as state:
        with pytest.raises(ValueError, match="crosses a block boundary"):
            state.forward(x, active=(12, 20))


def test_sub_block_cache_runs_on_sdar(sdar):
    """Without a next-token head, the first step of each sub-block runs the whole block."""
    prompt = _prompt(10)
    x = torch.full((1, 24), sdar.mask_token_id)
    x[:, :10] = prompt
    handle = zoo.attach(sdar, "fast-dllm-v2-sub-block", block_length=BLOCK)
    with handle.request(10, mask_token_id=sdar.mask_token_id) as state:
        state.forward(x, active=(10, 12))
        assert state.last_computed[0].sum() == BLOCK
        state.forward(x, active=(10, 12))
        assert state.last_computed[0].nonzero().flatten().tolist() == [10, 11]
        state.forward(x, active=(12, 14))
        assert state.last_computed[0].sum() == BLOCK
    out = block_diffusion_generate(
        zoo.attach(sdar, "block-cache"), prompt, mask_token_id=sdar.mask_token_id,
        gen_length=14, block_length=BLOCK, denoising_steps=4, top_k=1,
    )  # fmt: skip
    assert not (out.tokens == sdar.mask_token_id).any()
