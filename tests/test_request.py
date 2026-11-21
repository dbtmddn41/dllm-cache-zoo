"""The request object: output shapes, block tracking and helpers."""

from __future__ import annotations

import pytest
import torch

import dllm_cache_zoo as zoo
from dllm_cache_zoo.methods.d2cache import certainty_density, nucleus_select
from dllm_cache_zoo.methods.dkv_cache import nearby_masked
from dllm_cache_zoo.positions import mask_to_index, put_, take


def _x(llada, prompt, gen=16):
    return torch.cat([prompt, torch.full((prompt.size(0), gen), llada.config.mask_token_id)], dim=1)


def test_outputs_agree(llada, prompt):
    x = _x(llada, prompt)
    for method in ("none", "fast-dllm-prefix", "fast-dllm-dual", "dkv-cache-decode", "d2cache"):
        full, part = [], []
        for output, sink in (("full", full), ("active", part)):
            with zoo.attach(llada, method).request(prompt_length=11) as state:
                for _ in range(3):
                    sink.append(state.forward(x, active=(11, 19), output=output))
        for a, b in zip(full, part, strict=True):
            assert a.shape == (2, 27, 97) and b.shape == (2, 8, 97)
            assert torch.equal(a[:, 11:19], b)


def test_a_new_span_starts_a_new_block(llada, prompt):
    x = _x(llada, prompt)
    with zoo.attach(llada, "fast-dllm-dual").request(prompt_length=11) as state:
        state.forward(x, active=(11, 19))
        state.forward(x, active=slice(11, 19))
        assert (state.block_index, state.block_step, state.step) == (0, 2, 2)
        state.forward(x, active=range(19, 27))
        assert (state.block_index, state.block_step, state.step) == (1, 1, 3)
        assert state.last_computed.all()  # the first pass of a block is full


def test_default_span_is_the_response(llada, prompt):
    x = _x(llada, prompt)
    with zoo.attach(llada, "none").request(prompt_length=11) as state:
        state.forward(x)
        assert state.active == (11, 27)


def test_bad_input_is_reported(llada, prompt):
    x = _x(llada, prompt)
    state = zoo.attach(llada, "none").request(prompt_length=11)
    with pytest.raises(ValueError, match="does not fit"):
        state.forward(x, active=(20, 40))
    with pytest.raises(ValueError, match="Unknown output"):
        state.forward(x, output="block")
    with pytest.raises(ValueError, match=r"\(B, L\)"):
        state.forward(x[0])
    state.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.forward(x)
    with pytest.raises(ValueError, match="Unknown cache method"):
        zoo.attach(llada, "no-such-method")


def test_requests_do_not_share_state(llada, prompt):
    x = _x(llada, prompt)
    handle = zoo.attach(llada, "dllm-cache", prompt_interval=50, response_interval=50)
    one, two = handle.request(prompt_length=11), handle.request(prompt_length=11)
    first = one.forward(x)
    y = x.clone()
    y[:, 11:15] = 7
    two.forward(y)
    two.forward(y)
    again = handle.request(prompt_length=11).forward(x)
    assert torch.equal(first, again)
    assert one.kv.keys[0] is not two.kv.keys[0]


def test_take_and_put_roundtrip():
    t = torch.arange(2 * 6 * 3, dtype=torch.float).view(2, 6, 3)
    idx = torch.tensor([[0, 2, 5], [1, 2, 3]])
    assert torch.equal(take(t, idx)[1, 0], t[1, 1])
    assert torch.equal(take(t, idx[:1])[1, 2], t[1, 5])
    assert torch.equal(take(t, slice(2, 4)), t[:, 2:4])
    out = torch.zeros_like(t)
    put_(out, idx, take(t, idx))
    assert torch.equal(out[0, 5], t[0, 5]) and (out[0, 1] == 0).all()
    kv = torch.zeros(2, 4, 6, 3)
    put_(kv, idx, torch.ones(2, 4, 3, 3), dim=2)
    assert kv[1, :, 1].eq(1).all() and kv[1, :, 0].eq(0).all()


def test_mask_to_index_tops_up_short_rows():
    mask = torch.tensor([[True, False, True, True], [False, False, True, False]])
    index, topped = mask_to_index(mask)
    assert index.tolist() == [[0, 2, 3], [0, 1, 2]]
    assert topped.sum(1).tolist() == [3, 3]
    scores = torch.tensor([[0.0, 0, 0, 0], [0.1, 0.5, 0.0, 0.9]])
    index, _ = mask_to_index(mask, scores)
    assert index[1].tolist() == [1, 2, 3]


def test_certainty_density_is_higher_near_decoded_tokens():
    decoded = torch.zeros(1, 40, dtype=torch.bool)
    decoded[0, :5] = True
    density = certainty_density(decoded, sigma=3.0)[0]
    assert density[5] > density[20] > density[39] >= -1e-6
    full = certainty_density(torch.ones(1, 8, dtype=torch.bool), 3.0)
    assert full.allclose(torch.ones(1, 8), atol=0.01)


def test_nucleus_select_respects_the_mask():
    scores = torch.tensor([[5.0, 3.0, 1.0, 1.0]])
    allowed = torch.tensor([[False, True, True, True]])
    picked = nucleus_select(scores, 0.7, mask=allowed)
    assert picked.tolist() == [[False, True, False, False]]
    assert nucleus_select(scores, 0.0).tolist() == [[True, False, False, False]]


def test_nearby_masked_alternates_left_and_right():
    selected = torch.zeros(10, dtype=torch.bool)
    selected[5] = True
    masked = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    out = nearby_masked(selected, torch.tensor([5]), masked, window=3)
    assert out.nonzero().flatten().tolist() == [3, 4, 5, 7]
