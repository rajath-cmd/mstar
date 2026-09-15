"""Audio-tower output -> text embedding space.

Two linears with an activation between, not one. The single-linear shape is a
tempting simplification because ``linear_1`` alone already lands in the text
hidden size -- and it loads without error, because ``linear_2`` is square. It
just produces a wrong transcript.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.voxtral_rt.config import VoxtralRealtimeConfig

_ACT = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}


class Projector(nn.Module):
    def __init__(self, cfg: VoxtralRealtimeConfig):
        super().__init__()
        act = cfg.projector_hidden_act
        if act not in _ACT:
            raise ValueError(f"unsupported projector_hidden_act {act!r}")
        self.act = _ACT[act]
        self.linear_1 = nn.Linear(
            cfg.audio.hidden_size * cfg.downsample_factor, cfg.text.hidden_size, bias=False
        )
        self.linear_2 = nn.Linear(cfg.text.hidden_size, cfg.text.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))
