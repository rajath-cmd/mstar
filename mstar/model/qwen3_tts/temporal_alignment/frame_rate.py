"""Codec frame-rate conversion for Qwen3-TTS-12Hz.

12.5 Hz = 24000 (input sr) / 1920 (encode downsample). NOT 12 Hz — the
legacy ``data/tts_synth/event_frame_spans.py`` approximation must not
be reused here; a 5 % error over 10 s of audio is ~500 ms drift.
"""

from __future__ import annotations

CODEC_FRAME_RATE_HZ: float = 12.5


def sec_to_frame(t: float, rate: float = CODEC_FRAME_RATE_HZ) -> int:
    return int(round(float(t) * rate))


def frame_to_sec(f: int, rate: float = CODEC_FRAME_RATE_HZ) -> float:
    return float(f) / rate
