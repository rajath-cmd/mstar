"""Voxtral-Realtime audio tower: log-mel -> one embedding per 80 ms.

Runs ONCE per request, over the whole utterance, at prefill. That is not a
simplification of the streaming design -- it is equivalent to it. Every layer
here is causal with a fixed sliding window, so a dense pass over the full
sequence produces exactly the values an incremental pass with a KV cache would,
and it produces them in one kernel launch per layer instead of one per 80 ms.

The streaming path (feeding chunks with a conv padding cache and an encoder KV
cache) matters when audio arrives live; for offline transcription the whole
waveform is already in hand.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.norm import RMSNorm
from mstar.model.voxtral_rt.config import VoxtralAudioConfig


class CausalConv1d(nn.Conv1d):
    """Conv1d padded only on the left, so no output sees a future sample.

    Ordinary symmetric padding would let the first output frames peek forward,
    which is inaudible in a transcript but shifts every frame's alignment
    against the audio by half a kernel.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, bias=True)
        self.left_pad = (kernel_size - 1) * self.dilation[0] + 1 - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class Embedder(nn.Module):
    """Two GELU convs; the second has stride 2, halving the 100 Hz mel rate."""

    def __init__(self, cfg: VoxtralAudioConfig):
        super().__init__()
        self.conv1 = CausalConv1d(cfg.num_mel_bins, cfg.hidden_size, 3)
        self.conv2 = CausalConv1d(cfg.hidden_size, cfg.hidden_size, 3, stride=2)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``[B, n_mels, T]`` -> ``[B, T/2, hidden]``."""
        h = F.gelu(self.conv1(mel))
        h = F.gelu(self.conv2(h))
        # .contiguous() is load-bearing, not tidiness: the permute leaves a
        # transposed view whose row stride is the sequence length, and M*'s
        # RMSNorm dispatches to a FlashInfer kernel that requires a row stride
        # divisible by 8. Without it the tower dies on the first layer norm
        # with a stride complaint that says nothing about a permute.
        return h.permute(0, 2, 1).contiguous()


def _rope_tables(
    n: int, head_dim: int, theta: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(n, device=device).float()
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [B,1,T,D]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class AudioAttention(nn.Module):
    """Causal self-attention with a sliding window.

    Biases follow the checkpoint exactly: q, v and o carry one, k does not.
    That asymmetry is inherited from Whisper, and loading a k bias that does
    not exist -- or dropping the three that do -- fails silently as a slightly
    wrong transcript rather than as a shape error.
    """

    def __init__(self, cfg: VoxtralAudioConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.window = cfg.sliding_window
        d = cfg.num_attention_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, d, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, d, bias=True)
        self.o_proj = nn.Linear(d, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, cos, sin, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class AudioMLP(nn.Module):
    """SwiGLU; ``down_proj`` carries a bias, the gate and up projections do not."""

    def __init__(self, cfg: VoxtralAudioConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AudioLayer(nn.Module):
    def __init__(self, cfg: VoxtralAudioConfig):
        super().__init__()
        self.self_attn = AudioAttention(cfg)
        self.self_attn_layer_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.final_layer_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = AudioMLP(cfg)

    def forward(self, x: torch.Tensor, cos, sin, mask) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), cos, sin, mask)
        return x + self.mlp(self.final_layer_norm(x))


def sliding_causal_mask(n: int, window: int, device, dtype) -> torch.Tensor:
    """Additive mask: attend to ``[i-window+1, i]`` and nothing later."""
    idx = torch.arange(n, device=device)
    allowed = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - window)
    mask = torch.zeros(n, n, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


class AudioTower(nn.Module):
    """Embedder + 32 causal layers + final norm."""

    def __init__(self, cfg: VoxtralAudioConfig):
        super().__init__()
        self.config = cfg
        self.embedder = Embedder(cfg)
        self.layers = nn.ModuleList(AudioLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``[B, n_mels, T_mel]`` -> ``[B, T_mel/2, hidden]``."""
        h = self.embedder(mel)
        n = h.shape[1]
        cos, sin = _rope_tables(
            n, self.config.head_dim, self.config.rope_theta, h.device, h.dtype
        )
        cos, sin = cos[None], sin[None]
        mask = sliding_causal_mask(n, self.config.sliding_window, h.device, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        return self.norm(h)
