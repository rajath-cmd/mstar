"""No-op Prometheus shims for the ported alignment package.

The upstream module imports a set of counters/gauges from vllm-omni's metrics
registry. M* has no Prometheus surface yet (Phase 4), and the alignment code
already guards every metric call behind a soft import, so these stand in with
the same names and a no-op API. Replacing this module with real collectors is
all Phase 4 needs to do on this side.
"""

from __future__ import annotations


class _NoOp:
    def labels(self, **_kwargs) -> "_NoOp":
        return self

    def inc(self, *_a, **_k) -> None:
        pass

    def observe(self, *_a, **_k) -> None:
        pass

    def set(self, *_a, **_k) -> None:
        pass


TTS_ALIGNMENT_CANCEL_TOTAL = _NoOp()
TTS_ALIGNMENT_COMMIT_REVISION_TOTAL = _NoOp()
TTS_ALIGNMENT_EVICT_TOTAL = _NoOp()
TTS_ALIGNMENT_FINALIZE_TOTAL = _NoOp()
TTS_ALIGNMENT_FINALIZED = _NoOp()
TTS_ALIGNMENT_HEAD_LOADED = _NoOp()
TTS_ALIGNMENT_INFLIGHT = _NoOp()
TTS_ALIGNMENT_PARTIAL_TOTAL = _NoOp()
TTS_ALIGNMENT_PARTIAL_WORDS = _NoOp()
TTS_ALIGNMENT_POP_TOTAL = _NoOp()
TTS_ALIGNMENT_POP_WAIT = _NoOp()
TTS_ALIGNMENT_REGISTER_TOTAL = _NoOp()
TTS_ALIGNMENT_SCALE_RATIO = _NoOp()
