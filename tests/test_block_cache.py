"""The completed-block cache is exact on a block-causal model."""

from __future__ import annotations

import pytest
import torch
from toy import block_attention_mask

import dllm_cache_zoo as zoo
from dllm_cache_zoo.loops import block_diffusion_generate
from dllm_cache_zoo.methods.block_cache import block_causal_mask


@pytest.mark.parametrize("prompt_len", [8, 11])  # aligned to the block, and with a remainder
def test_block_cache_matches_the_masked_full_pass(sdar, prompt_len):
    block = 4
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 90, (2, prompt_len), generator=gen)
    total = 24
    x = torch.full((2, total), sdar.mask_token_id)
    x[:, :prompt_len] = prompt
    handle = zoo.attach(sdar, "block-cache")
    with handle.request(prompt_length=prompt_len, mask_token_id=sdar.mask_token_id) as state:
        for start in range((prompt_len // block) * block, total, block):
            end = start + block
            for _ in range(2):  # two denoising steps per block
                ref = sdar(x[:, :end], attention_mask=block_attention_mask(end, block)).logits
                out = state.forward(x, active=(start, end), output="active")
                torch.testing.assert_close(out, ref[:, start:end], rtol=0, atol=1e-9)
                masked = (x[:, start:end] == sdar.mask_token_id).nonzero()
                if masked.numel():
                    row, col = masked[0].tolist()
                    x[row, start + col] = int(out[row, col].argmax())
            x[:, start:end] = torch.where(
                x[:, start:end] == sdar.mask_token_id, out.argmax(-1), x[:, start:end]
            )
        assert state.data["committed"] == total - block


def test_block_cache_and_baseline_generate_the_same_tokens(sdar):
    prompt = torch.randint(0, 90, (1, 10), generator=torch.Generator().manual_seed(2))
    kw = dict(
        mask_token_id=sdar.mask_token_id,
        gen_length=20,
        block_length=4,
        denoising_steps=4,
        top_k=1,
        remasking="low_confidence_static",
    )
    base = block_diffusion_generate(zoo.attach(sdar, "none"), prompt, **kw)
    out = block_diffusion_generate(zoo.attach(sdar, "block-cache"), prompt, **kw)
    assert torch.equal(out.tokens, base.tokens)
    assert not (out.tokens == sdar.mask_token_id).any()
    assert out.stats.recomputed_share < base.stats.recomputed_share


def test_blocks_are_decoded_left_to_right(sdar):
    x = torch.full((1, 16), sdar.mask_token_id)
    x[:, :4] = 1
    with zoo.attach(sdar, "block-cache").request(4, mask_token_id=sdar.mask_token_id) as state:
        state.forward(x, active=(8, 12))
        with pytest.raises(ValueError, match="left to right"):
            state.forward(x, active=(4, 8))


def test_mask_is_none_inside_one_block():
    assert block_causal_mask(8, 12, 4, torch.device("cpu")) is None
    mask = block_causal_mask(0, 8, 4, torch.device("cpu"))
    assert mask is not None and mask.shape == (1, 1, 8, 8)
    assert not mask[0, 0, 0, 4] and mask[0, 0, 4, 0] and mask[0, 0, 1, 3]
