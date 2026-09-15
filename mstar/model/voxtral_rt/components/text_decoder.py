"""Voxtral-Realtime text decoder: Mistral, plus time conditioning.

One structural difference from stock Mistral, and it is easy to miss: each
layer scales its post-attention hidden state by ``1 + ada_rms_norm(t_cond)``,
where ``t_cond`` is a sinusoidal embedding of ``num_delay_tokens``. The model
is told, per forward, how far behind the audio it is allowed to run.

``num_delay_tokens`` is fixed for a deployment, so ``t_cond`` -- and therefore
every layer's scale vector -- is CONSTANT. It is computed once at load and
cached, which removes two small matmuls per layer per token from the decode
path without changing a single output value.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.norm import RMSNorm
from mstar.model.voxtral_rt.components.audio_tower import apply_rope
from mstar.model.voxtral_rt.config import VoxtralTextConfig


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the delay, in tokens."""

    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        half = dim // 2
        inv_freq = torch.exp(-math.log(theta) * torch.arange(half).float() / half)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t * self.inv_freq.to(device=t.device, dtype=t.dtype)
        return torch.cat((emb.cos(), emb.sin()))


class AdaRMSNorm(nn.Module):
    """t_cond -> a per-channel scale. Bottlenecked through 32 dims."""

    def __init__(self, cfg: VoxtralTextConfig, bottleneck: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(cfg.hidden_size, bottleneck, bias=False)
        self.linear2 = nn.Linear(bottleneck, cfg.hidden_size, bias=False)

    def forward(self, t_cond: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(t_cond)))


class TextAttention(nn.Module):
    """GQA, no biases anywhere -- the opposite of the audio tower's convention."""

    def __init__(self, cfg: VoxtralTextConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache: dict | None, mask):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None and t > 1
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class TextMLP(nn.Module):
    def __init__(self, cfg: VoxtralTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextLayer(nn.Module):
    def __init__(self, cfg: VoxtralTextConfig):
        super().__init__()
        self.self_attn = TextAttention(cfg)
        self.mlp = TextMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ada_rms_norm = AdaRMSNorm(cfg)

    def forward(self, x, cos, sin, cache, mask, t_scale: torch.Tensor):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, mask)
        h = self.post_attention_layernorm(x) * (1 + t_scale)
        return x + self.mlp(h)


class TextDecoder(nn.Module):
    def __init__(self, cfg: VoxtralTextConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(TextLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # Filled by VoxtralRealtime once t_cond is known; one [1, hidden]
        # tensor per layer.
        self.t_scales: list[torch.Tensor] | None = None

    def forward(self, inputs_embeds, position_offset: int, caches, mask=None):
        b, t, _ = inputs_embeds.shape
        dev, dtype = inputs_embeds.device, inputs_embeds.dtype
        cfg = self.config
        inv = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, cfg.head_dim, 2, device=dev).float() / cfg.head_dim)
        )
        pos = torch.arange(position_offset, position_offset + t, device=dev).float()
        freqs = torch.outer(pos, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(
                h, cos, sin,
                caches[i] if caches is not None else None,
                mask, self.t_scales[i],
            )
        return self.norm(h)
