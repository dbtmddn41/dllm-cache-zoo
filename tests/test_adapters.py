"""The adapters reproduce the model's own forward."""

from __future__ import annotations

import pytest
import torch
from toy import block_attention_mask

import dllm_cache_zoo as zoo


def test_llada_full_pass_matches_model(llada, prompt):
    mask_id = llada.config.mask_token_id
    x = torch.cat([prompt, torch.full((2, 24), mask_id)], dim=1)
    handle = zoo.attach(llada, "none")
    assert handle.adapter.name == "llada" and handle.adapter.family == "bidirectional"
    with handle.request(prompt_length=prompt.size(1)) as state:
        out = state.forward(x)
    assert torch.equal(out, llada(x).logits)


def test_llada_mask_token_comes_from_config(llada):
    assert zoo.attach(llada, "none").adapter.mask_token_id == llada.config.mask_token_id


def test_sdar_block_pass_matches_model(sdar, prompt):
    block = 4
    x = torch.cat([prompt, torch.full((2, 13), sdar.mask_token_id)], dim=1)  # 24 positions
    handle = zoo.attach(sdar, "none", block_length=block)
    assert handle.adapter.name == "sdar" and handle.adapter.family == "block-causal"
    with handle.request(prompt_length=prompt.size(1), mask_token_id=sdar.mask_token_id) as state:
        for start in (8, 12, 20):
            end = start + block
            ref = sdar(x[:, :end], attention_mask=block_attention_mask(end, block)).logits
            out = state.forward(x, active=(start, end), output="active")
            torch.testing.assert_close(out, ref[:, start:end], rtol=0, atol=1e-10)


def test_unknown_model_is_rejected():
    with pytest.raises(TypeError, match="No adapter recognises"):
        zoo.attach(torch.nn.Linear(2, 2), "none")


def test_method_of_other_family_is_rejected(llada, sdar):
    with pytest.raises(ValueError, match="block-causal"):
        zoo.attach(llada, "block-cache")
    with pytest.raises(ValueError, match="bidirectional"):
        zoo.attach(sdar, "fast-dllm-dual")


def test_request_needs_a_mask_token(sdar):
    with pytest.raises(ValueError, match="mask token"):
        zoo.attach(sdar, "block-cache").request(prompt_length=3)
