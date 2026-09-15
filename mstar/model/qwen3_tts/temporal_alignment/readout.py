"""Argmax-based pointer-scores → word timestamps (alternative to Viterbi).

The inference server uses :func:`viterbi.word_timestamps_viterbi_full_coverage`;
this variant is kept for parity with the training-side held-out eval and
local A/B against the DP decoder.
"""

from __future__ import annotations

import torch

from mstar.model.qwen3_tts.temporal_alignment.frame_rate import (
    frame_to_sec,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
)


def word_timestamps(
    scores: torch.Tensor,
    spans: list[WordSpan],
    rate: float = 12.5,
    monotonic: bool = False,
) -> list[dict]:
    """Argmax + optional running-max clamp; returns ``[{word, start, end}]``."""
    if scores.numel() == 0:
        return []
    active = scores.argmax(dim=-1)
    if monotonic:
        run = 0
        am = active.tolist()
        for i, v in enumerate(am):
            run = max(run, v)
            am[i] = run
        active = torch.tensor(am)
    out: list[dict] = []
    nf = active.shape[0]
    for w_idx, span in enumerate(spans):
        frames = [f for f in range(nf) if int(active[f]) == w_idx]
        if not frames:
            continue
        out.append(
            {
                "word": span.word,
                "start": round(frame_to_sec(frames[0], rate), 3),
                "end": round(frame_to_sec(frames[-1] + 1, rate), 3),
            }
        )
    out.sort(key=lambda d: d["start"])
    return out
