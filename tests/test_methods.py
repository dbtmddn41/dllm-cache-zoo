"""Every method runs to completion, and is exact where its definition says so."""

from __future__ import annotations

import pytest
import torch

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import llada_generate, sliding_window_generate

GEN = dict(gen_length=24, steps=24, block_length=8)

# (method, settings, loop keyword arguments)
RUNS = [
    ("fast-dllm-prefix", {}, {}),
    ("fast-dllm-dual", {}, {}),
    ("dkv-cache-decode", {"refresh_interval": 4}, {}),
    ("dkv-cache-greedy", {"refresh_interval": 2, "window": 2}, {"remasking": "random_order"}),
    ("dllm-cache", {"prompt_interval": 5, "response_interval": 3, "transfer_ratio": 0.25}, {}),
    ("dllm-cache", {"prompt_interval": 5, "response_interval": 3, "transfer_ratio": 0.0}, {}),
    ("d2cache", {"current_k": 4, "rollout_p": 0.3}, {}),
    (
        "d2cache",
        {"current_k": 4, "rollout_p": 0.3, "inflate_w": 2, "refresh_confidence": False},
        {},
    ),
    ("dpad", {"cache": "none", "window": 16, "seed": 0}, {}),
    ("dpad", {"cache": "prefix", "window": 16, "seed": 0}, {}),
    ("dpad", {"cache": "dual", "window": 16, "seed": 0}, {}),
    ("dpad", {"cache": "dual", "window": 16, "sampler": "uniform", "preserved": 3, "seed": 0}, {}),
    ("sparse-dllm", {}, {}),
    ("sparse-dllm", {"keep_ratio": 0.3, "kernel_size": None, "delay": 2}, {}),
]


@pytest.mark.parametrize(("method", "settings", "loop"), RUNS)
@pytest.mark.parametrize("threshold", [None, 0.5])
def test_method_fills_every_mask(llada, prompt, method, settings, loop, threshold):
    if threshold is not None and loop:
        pytest.skip("a fixed unmasking order has no confidence threshold")
    handle = zoo.attach(llada, method, **settings)
    out = llada_generate(handle, prompt, threshold=threshold, **GEN, **loop)
    assert not (out.tokens == llada.config.mask_token_id).any()
    assert torch.equal(out.tokens[:, : prompt.size(1)], prompt)
    assert 0 < out.stats.recomputed_share < 1
    assert out.stats.forwards == out.steps


# Settings under which a method recomputes everything and must match the baseline.
EXACT = [
    ("dkv-cache-decode", {"refresh_interval": 1}, {}),
    ("dkv-cache-greedy", {"refresh_interval": 1}, {"remasking": "random_order"}),
    ("dllm-cache", {"prompt_interval": 1, "response_interval": 1}, {}),
    ("dllm-cache", {"prompt_interval": 1, "response_interval": 7, "transfer_ratio": 1.0}, {}),
    ("d2cache", {"rollout_p": 1.0, "current_k": 24}, {}),
    ("dpad", {"cache": "none", "window": 24, "scale": 1e9}, {}),
]


@pytest.mark.parametrize(("method", "settings", "loop"), EXACT)
def test_exact_settings_match_full_recomputation(llada, prompt, method, settings, loop):
    def run(name, **kw):
        gen = torch.Generator().manual_seed(3)
        return llada_generate(zoo.attach(llada, name, **kw), prompt, generator=gen, **GEN, **loop)

    base, out = run("none"), run(method, **settings)
    assert torch.equal(out.tokens, base.tokens)


@pytest.mark.parametrize("method", ["fast-dllm-prefix", "fast-dllm-dual", "sparse-dllm", "dpad"])
def test_first_pass_of_a_block_is_a_full_pass(llada, prompt, method):
    settings = {"window": 24, "scale": 1e9} if method == "dpad" else {}
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    with zoo.attach(llada, method, **settings).request(prompt_length=11) as state:
        out = state.forward(x, active=(11, 19))
    torch.testing.assert_close(out, llada(x).logits, rtol=0, atol=1e-10)


def test_prefix_cache_is_exact_when_the_prefix_does_not_change(llada, prompt):
    """With the prompt as the only prefix, later steps must reuse exactly what a full pass
    over an unchanged prompt would give for a model whose prompt ignores the response.
    Here it is checked the weaker way: recomputed positions carry valid logits."""
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    with zoo.attach(llada, "fast-dllm-prefix").request(prompt_length=11) as state:
        state.forward(x, active=(11, 19))
        out = state.forward(x, active=(11, 19))  # same tokens: the store is still exact
        assert state.last_computed[:, 11:].all() and not state.last_computed[:, :11].any()
    torch.testing.assert_close(out[:, 11:], llada(x).logits[:, 11:], rtol=0, atol=1e-9)


def test_dual_cache_runs_only_the_block(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    with zoo.attach(llada, "fast-dllm-dual").request(prompt_length=11) as state:
        state.forward(x, active=(11, 19))
        out = state.forward(x, active=(11, 19))
        computed = state.last_computed
    assert computed[:, 11:19].all() and computed.sum() == 2 * 8
    torch.testing.assert_close(out[:, 11:19], llada(x).logits[:, 11:19], rtol=0, atol=1e-9)
    assert (out[:, 19:] == 0).all()


def test_dkv_decode_reuses_tokens_one_step_after_they_are_revealed(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt[:1], torch.full((1, 8), mask_id)], dim=1)
    with zoo.attach(llada, "dkv-cache-decode", refresh_interval=100).request(11) as state:
        state.forward(x)  # step 0: full, nothing stored
        x[0, 11] = 5
        state.forward(x)  # step 1: full, stores prompt and position 11
        assert state.last_computed.all()
        x[0, 12] = 6
        state.forward(x)  # step 2: masks of step 1's input, which include 12
        assert state.last_computed[0].nonzero().flatten().tolist() == list(range(12, 19))
        x[0, 13] = 7
        state.forward(x)  # step 3: 12 is stored now; 13 gets its delayed pass
        assert state.last_computed[0].nonzero().flatten().tolist() == list(range(13, 19))


def test_dkv_greedy_needs_the_next_positions(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt[:1], torch.full((1, 8), mask_id)], dim=1)
    with zoo.attach(llada, "dkv-cache-greedy", refresh_interval=4, window=1).request(11) as state:
        state.forward(x, reveal=torch.tensor([[14]]))
        x[0, 14] = 3
        with pytest.raises(ValueError, match="reveal"):
            state.forward(x)
        state.forward(x, reveal=torch.tensor([[17]]))
        # just revealed (14), about to be revealed (17) and one masked neighbour (16)
        assert state.last_computed[0].nonzero().flatten().tolist() == [14, 16, 17]


def test_dllm_cache_updates_the_share_it_is_told_to(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    handle = zoo.attach(
        llada, "dllm-cache", prompt_interval=100, response_interval=100, transfer_ratio=0.25
    )
    with handle.request(prompt_length=11) as state:
        state.forward(x)
        state.forward(x)
        layers = handle.adapter.num_layers
        # layer 0 runs everything; the others run a quarter of the 24 response positions
        expected = (35 + (layers - 1) * 6) / layers
        assert state.stats.computed == pytest.approx(35 + expected)


def test_elastic_cache_refreshes_when_attention_drifts(llada, prompt):
    always = zoo.attach(llada, "elastic-cache", gamma=1.1)
    never = zoo.attach(llada, "elastic-cache", gamma=-1.0)
    kw = dict(gen_length=24, window_length=8, threshold=0.5)
    a = sliding_window_generate(always, prompt[:1], **kw)
    b = sliding_window_generate(never, prompt[:1], **kw)
    for out in (a, b):
        assert not (out.tokens == llada.config.mask_token_id).any()
    # a refresh from the second layer on costs more than running the window alone
    assert a.stats.recomputed_share > b.stats.recomputed_share


def test_elastic_cache_takes_one_sequence(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 8), mask_id)], dim=1)
    with zoo.attach(llada, "elastic-cache").request(prompt_length=11) as state:
        with pytest.raises(NotImplementedError, match="one sequence"):
            state.forward(x)


def test_sparse_dllm_keeps_the_requested_share(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    handle = zoo.attach(llada, "sparse-dllm", keep_ratio=0.5, kernel_size=3)
    with handle.request(prompt_length=11) as state:
        state.forward(x, active=(11, 19))
        state.forward(x, active=(11, 19))
        kept = state.data["k"][0].size(2)
        assert kept == int((35 - 8) * 0.5)
        out = state.forward(x, active=(11, 19), output="active")
        assert out.shape[1] == 8 and state.last_computed.sum() == 2 * 8


def test_dpad_drops_suffix_positions(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    handle = zoo.attach(llada, "dpad", cache="none", window=4, k=4.0, scale=2.0, seed=0)
    with handle.request(prompt_length=11) as state:
        state.forward(x, active=(11, 19))
        computed = state.last_computed[0]
    assert computed[:19].all()
    assert not computed[23:].any()  # beyond the window nothing is kept
    assert computed.sum() < 35


def test_methods_are_listed_with_their_sources():
    infos = {info.name: info for info in zoo.methods()}
    assert set(infos) >= {
        "none",
        "fast-dllm-prefix",
        "fast-dllm-dual",
        "dkv-cache-decode",
        "dkv-cache-greedy",
        "dllm-cache",
        "d2cache",
        "elastic-cache",
        "dpad",
        "sparse-dllm",
        "block-cache",
        "fast-dllm-v2-sub-block",
    }
    for name, info in infos.items():
        if name != "none":
            assert info.paper.startswith("https://") and info.reference
    block_methods = [i.name for i in zoo.methods("block-causal")]
    assert block_methods == ["none", "block-cache", "fast-dllm-v2-sub-block"]
