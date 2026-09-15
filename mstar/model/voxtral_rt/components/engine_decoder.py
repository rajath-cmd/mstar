"""Voxtral text decoder on M*'s engine resources.

The same 26-layer Mistral stack as ``text_decoder.py``, rebuilt on the shared
``Attention`` component so the engine owns the KV cache. That is the whole
point of the port: paged attention, continuous batching across requests, and
CUDA-graph capture of the decode step -- none of which a module holding its own
Python dict of K/V tensors can participate in.

``text_decoder.py`` is deliberately kept alongside this. It is the verified
oracle -- token-identical to the HF reference on the whole corpus -- and
``test_engine_parity.py`` asserts this path agrees with it. A second
implementation of a model is a liability unless something keeps the two honest;
that test is what does.

Two structural notes carried over from the reference:

* Each layer scales its post-attention state by ``1 + ada_rms_norm(t_cond)``,
  where ``t_cond`` encodes ``num_delay_tokens``. It is constant for a
  deployment, so the per-layer scale is precomputed once at load.
* The decoder is GQA (32 query heads over 8 KV heads) and carries no biases
  anywhere -- the opposite of the audio tower's Whisper-inherited convention.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.attention import Attention
from mstar.model.components.mlp import GatedMLP
from mstar.model.components.norm import RMSNorm
from mstar.model.voxtral_rt.components.text_decoder import AdaRMSNorm
from mstar.model.voxtral_rt.config import (
    TEXT_ATTN,
    TEXT_KV,
    TEXT_POS,
    VoxtralTextConfig,
)


class EngineTextLayer(nn.Module):
    """One Mistral block whose attention is planned by the engine."""

    def __init__(self, cfg: VoxtralTextConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            qkv_bias=False,
            o_bias=False,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            attn_key=TEXT_ATTN,
            kv_key=TEXT_KV,
            pos_key=TEXT_POS,
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = GatedMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            activation="silu",
            bias=False,
        )
        self.ada_rms_norm = AdaRMSNorm(cfg)

    def forward(self, hidden_states: torch.Tensor, t_scale: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states)
        )
        gated = self.post_attention_layernorm(hidden_states) * (1 + t_scale)
        return hidden_states + self.mlp(gated)


class EngineTextDecoder(nn.Module):
    """Decoder stack. Parameter paths mirror the checkpoint's layout."""

    def __init__(self, cfg: VoxtralTextConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            EngineTextLayer(cfg) for _ in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # One [1, hidden] tensor per layer, filled by prepare_time_conditioning.
        self.t_scales: list[torch.Tensor] | None = None

    def forward(self, inputs_embeds: torch.Tensor, *, label: str = "main") -> torch.Tensor:
        """``[n_tokens, hidden]`` packed across the batch -> same shape.

        No position argument and no mask: the engine's attention resource
        planned both before this ran, from the segments the submodule declared.

        The label and layer index are CURSORS on the shared resources, not
        arguments to ``Attention.forward`` -- the caller running the stack has
        to advance them. Forgetting ``set_layer_idx`` makes all 26 layers read
        and write layer 0's KV pages, and the failure is deceptive: a prefill
        still comes out bit-exact, because with an empty cache the attention is
        computed entirely from the q/k/v of that same call. Only the first
        DECODE step, which is the first read of cached keys, goes wrong -- so
        the model loads, prefills perfectly, and then transcribes nothing but
        padding.
        """
        if self.t_scales is None:
            raise RuntimeError(
                "time conditioning was never prepared; call "
                "prepare_time_conditioning() after loading weights. Without it "
                "every layer's AdaRMSNorm scale is missing and the decoder "
                "transcribes at the wrong times rather than failing."
            )
        hidden = inputs_embeds
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, (layer, t_scale) in enumerate(
            zip(self.layers, self.t_scales, strict=True)
        ):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden = layer(hidden, t_scale)
        return self.norm(hidden)
