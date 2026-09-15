"""Alignment pointer head: scores each (codec frame, word) pair.

Queries = per-frame hidden state at the configured layer; keys =
per-word mean-pooled text-position hidden. Two variants per the trained
sidecar's ``head_type``: ``"linear"`` (bilinear) or ``"mlp"`` (two
``Sequential(Linear, GELU, Linear)`` projections).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
)


def pool_word_keys(text_hidden: torch.Tensor, spans: list[WordSpan]) -> torch.Tensor:
    """Mean-pool ``text_hidden`` rows per ``WordSpan`` into ``(Nw, H)``."""
    rows = [text_hidden[s.token_start : s.token_end].mean(dim=0) for s in spans]
    return torch.stack(rows, dim=0) if rows else text_hidden.new_zeros((0, text_hidden.shape[-1]))


class AlignmentPointerHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        proj_size: int = 256,
        head_type: str = "linear",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.proj_size = int(proj_size)
        self.head_type = head_type
        if head_type == "mlp":

            def _proj() -> nn.Module:
                return nn.Sequential(
                    nn.Linear(self.hidden_size, self.proj_size),
                    nn.GELU(),
                    nn.Linear(self.proj_size, self.proj_size),
                )

            self.W_q = _proj()
            self.W_k = _proj()
        else:
            self.W_q = nn.Linear(self.hidden_size, self.proj_size, bias=False)
            self.W_k = nn.Linear(self.hidden_size, self.proj_size, bias=False)
        self._scale = 1.0 / math.sqrt(self.proj_size)

    def forward(self, frame_hidden: torch.Tensor, word_key_hidden: torch.Tensor) -> torch.Tensor:
        q = self.W_q(frame_hidden)
        k = self.W_k(word_key_hidden)
        return (q @ k.transpose(0, 1)) * self._scale
