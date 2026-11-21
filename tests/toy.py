"""Small random models with the module layout of released checkpoints.

They let the cache methods run on CPU without downloading weights. Each toy has
its own plain ``forward`` written the way the released model code computes it,
which is the reference the adapter is checked against.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def _never_predict(head: nn.Linear, token_id: int) -> None:
    """A trained model does not predict its mask token. Random weights can, which
    would leave a position masked forever, so the toy heads rule it out."""
    with torch.no_grad():
        head.bias.zero_()
        head.bias[token_id] = -1e4


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class _Rotary(nn.Module):
    """Mirrors ``RotaryEmbedding`` in the LLaDA model code."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._tables: tuple[torch.Tensor, torch.Tensor] | None = None

    def get_rotary_embedding(self, seq_len: int, device: torch.device):
        if self._tables is not None and self._tables[0].shape[-2] >= seq_len:
            sin, cos = self._tables
            return sin[:, :, :seq_len, :].to(device), cos[:, :, :seq_len, :].to(device)
        dim = self.head_dim
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        seq = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        positions = torch.cat((freqs, freqs), dim=-1)
        sin, cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self._tables = (sin, cos)
        return sin.to(device), cos.to(device)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        b, nh, t, hs = x.size()
        x = x.view(b, nh, t, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q_, k_ = q.float(), k.float()
        sin, cos = self.get_rotary_embedding(k_.shape[-2], q_.device)
        sin, cos = sin.type_as(q_), cos.type_as(q_)
        q_ = q_ * cos + self.rotate_half(q_) * sin
        k_ = k_ * cos + self.rotate_half(k_) * sin
        return q_.type_as(q), k_.type_as(k)


class _LLaDABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int):
        super().__init__()
        self.n_heads = n_heads
        self.attn_norm = _RMSNorm(d_model)
        self.ff_norm = _RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ff_proj = nn.Linear(d_model, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_hidden, bias=False)
        self.ff_out = nn.Linear(mlp_hidden, d_model, bias=False)
        self.act = nn.SiLU()
        self.rotary_emb = _Rotary(d_model // n_heads)
        self.q_norm = None
        self.k_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        xn = self.attn_norm(x)
        q = self.q_proj(xn).view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        k = self.k_proj(xn).view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        v = self.v_proj(xn).view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        q, k = self.rotary_emb(q, k)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.attn_out(att)
        h = self.ff_norm(x)
        h, up = self.ff_proj(h), self.up_proj(h)
        return x + self.ff_out(self.act(h) * up)


class _Inner(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab, d_model),
                emb_drop=nn.Dropout(0.0),
                ln_f=_RMSNorm(d_model),
                blocks=nn.ModuleList(
                    [_LLaDABlock(d_model, n_heads, 2 * d_model) for _ in range(n_layers)]
                ),
                ff_out=nn.Linear(d_model, vocab, bias=True),
            )
        )


class ToyLLaDA(nn.Module):
    """A bidirectional masked-diffusion model laid out like ``LLaDAModelLM``."""

    def __init__(
        self,
        vocab: int = 97,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(
            model_type="llada",
            n_heads=n_heads,
            n_kv_heads=n_heads,
            effective_n_kv_heads=n_heads,
            d_model=d_model,
            n_layers=n_layers,
            rope=True,
            rope_full_precision=True,
            input_emb_norm=False,
            weight_tying=False,
            scale_logits=False,
            mask_token_id=vocab - 1,
            vocab_size=vocab,
        )
        self.model = _Inner(vocab, d_model, n_heads, n_layers)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=1.5 / math.sqrt(p.size(-1)))
        _never_predict(self.model.transformer.ff_out, vocab - 1)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        tr = self.model.transformer
        x = tr.emb_drop(tr.wte(input_ids))
        for block in tr.blocks:
            x = block(x)
        return SimpleNamespace(logits=tr.ff_out(tr.ln_f(x)))


# --------------------------------------------------------------------------- SDAR


class _HeadNorm(_RMSNorm):
    """Norm over the head dimension, as SDAR applies to queries and keys."""


class _HFRotary(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class _SDARAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.head_dim = hidden // n_heads
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)
        self.q_norm = _HeadNorm(self.head_dim)
        self.k_norm = _HeadNorm(self.head_dim)

    def forward(self, x, cos, sin, mask):
        b, n, _ = x.shape
        shape = (b, n, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        def rot(t):
            half = t.shape[-1] // 2
            return t * cos + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * sin

        q, k = rot(q), rot(k)
        rep = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class _SDARMLP(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _SDARLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.self_attn = _SDARAttention(hidden, n_heads, n_kv_heads)
        self.mlp = _SDARMLP(hidden, 2 * hidden)
        self.input_layernorm = _RMSNorm(hidden, eps=1e-6)
        self.post_attention_layernorm = _RMSNorm(hidden, eps=1e-6)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class _SDARBase(nn.Module):
    def __init__(self, vocab, hidden, n_heads, n_kv_heads, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_SDARLayer(hidden, n_heads, n_kv_heads) for _ in range(n_layers)]
        )
        self.norm = _RMSNorm(hidden, eps=1e-6)
        self.rotary_emb = _HFRotary(hidden // n_heads)


class ToySDAR(nn.Module):
    """A block-causal model laid out like ``SDARForCausalLM``.

    ``forward`` takes the same block attention mask as the released model:
    ``(1, N, N)``, nonzero where a position may attend.
    """

    def __init__(self, vocab=101, hidden=48, n_heads=4, n_kv_heads=2, n_layers=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(
            model_type="sdar",
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            vocab_size=vocab,
        )
        self.mask_token_id = vocab - 1
        self.model = _SDARBase(vocab, hidden, n_heads, n_kv_heads, n_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=True)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=1.5 / math.sqrt(p.size(-1)))
        _never_predict(self.lm_head, self.mask_token_id)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, position_ids=None) -> SimpleNamespace:
        n = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(n, device=input_ids.device).unsqueeze(0)
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(x, position_ids)
        mask = None if attention_mask is None else attention_mask.bool().unsqueeze(1)
        for layer in self.model.layers:
            x = layer(x, cos, sin, mask)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))


class ToyFastDLLMv2(ToySDAR):
    """A block-causal model with a next-token head, laid out like ``Fast_dLLM_QwenForCausalLM``.

    The hidden state at position ``i`` predicts the token at ``i + 1``. The layout
    differs from SDAR only in having no norm on the per-head queries and keys.
    """

    def __init__(self, vocab=101, hidden=48, n_heads=4, n_kv_heads=2, n_layers=3, seed=1, bd=8):
        super().__init__(vocab, hidden, n_heads, n_kv_heads, n_layers, seed)
        self.config.model_type = "Fast_dLLM_Qwen"
        self.config.bd_size = bd
        for layer in self.model.layers:
            layer.self_attn.q_norm = None
            layer.self_attn.k_norm = None
            layer.self_attn.forward = _plain_attention.__get__(layer.self_attn)


def _plain_attention(self, x, cos, sin, mask):
    b, n, _ = x.shape
    shape = (b, n, -1, self.head_dim)
    q = self.q_proj(x).view(shape).transpose(1, 2)
    k = self.k_proj(x).view(shape).transpose(1, 2)
    v = self.v_proj(x).view(shape).transpose(1, 2)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rot(t):
        half = t.shape[-1] // 2
        return t * cos + torch.cat((-t[..., half:], t[..., :half]), dim=-1) * sin

    q, k = rot(q), rot(k)
    rep = self.n_heads // self.n_kv_heads
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
    return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


def block_attention_mask(length: int, block_length: int, device=None) -> torch.Tensor:
    """``(1, length, length)`` mask of a block diffusion model, as in the SDAR script."""
    blocks = torch.arange(length, device=device) // block_length
    return (blocks[None, :] <= blocks[:, None]).unsqueeze(0)
