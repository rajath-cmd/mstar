"""Alignment metrics — re-exported from the server's Prometheus registry.

This module exists because the ported alignment package imports its collectors
from one place. It started as no-ops; now it forwards to the real registry, so
the same series names vllm-omni exposes appear on M*'s /metrics.
"""

from __future__ import annotations

from mstar.metrics.prometheus import (  # noqa: F401
    TTS_ALIGNMENT_CANCEL_TOTAL,
    TTS_ALIGNMENT_COMMIT_REVISION_TOTAL,
    TTS_ALIGNMENT_EVICT_TOTAL,
    TTS_ALIGNMENT_FINALIZE_TOTAL,
    TTS_ALIGNMENT_FINALIZED,
    TTS_ALIGNMENT_HEAD_LOADED,
    TTS_ALIGNMENT_INFLIGHT,
    TTS_ALIGNMENT_PARTIAL_TOTAL,
    TTS_ALIGNMENT_PARTIAL_WORDS,
    TTS_ALIGNMENT_POP_TOTAL,
    TTS_ALIGNMENT_POP_WAIT,
    TTS_ALIGNMENT_REGISTER_TOTAL,
    TTS_ALIGNMENT_SCALE_RATIO,
)
