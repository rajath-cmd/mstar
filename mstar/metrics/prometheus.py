"""Prometheus metrics for Qwen3-TTS serving.

Series names are frozen to vllm-omni's so an existing Grafana dashboard, alert
rule or recording rule keeps working when the backend is swapped. That is the
whole point of the naming here: a drop-in replacement that renames its metrics
is not drop-in for whoever is on call.

Collectors degrade to no-ops when ``prometheus_client`` is absent, so a slim
install or a worker image without it never breaks serving.
"""

from __future__ import annotations

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client import generate_latest as _generate_latest

    _AVAILABLE = True
    REGISTRY = CollectorRegistry()
except Exception:  # pragma: no cover — exercised on slim installs
    _AVAILABLE = False
    REGISTRY = None


class _NoOp:
    def labels(self, **_kwargs) -> "_NoOp":
        return self

    def inc(self, *_a, **_k) -> None:
        pass

    def observe(self, *_a, **_k) -> None:
        pass

    def set(self, *_a, **_k) -> None:
        pass


def _counter(name: str, doc: str, labels: list[str] | None = None):
    if not _AVAILABLE:
        return _NoOp()
    return Counter(name, doc, labels or [], registry=REGISTRY)


def _gauge(name: str, doc: str, labels: list[str] | None = None):
    if not _AVAILABLE:
        return _NoOp()
    return Gauge(name, doc, labels or [], registry=REGISTRY)


def _histogram(name: str, doc: str, buckets: list[float], labels: list[str] | None = None):
    if not _AVAILABLE:
        return _NoOp()
    return Histogram(name, doc, labels or [], buckets=buckets, registry=REGISTRY)


# --- request lifecycle -------------------------------------------------------

TTS_REQUESTS_TOTAL = _counter(
    "tts_requests_total", "TTS requests by endpoint, voice and status",
    ["endpoint", "voice", "status"],
)
TTS_ACTIVE_REQUESTS = _gauge("tts_active_requests", "In-flight TTS requests")
TTS_INPUT_TEXT_CHARS = _histogram(
    "tts_input_text_chars", "Input text length in characters",
    [16, 32, 64, 128, 256, 512, 1024, 2048], ["task_type"],
)
TTS_AUDIO_DURATION_SECONDS = _histogram(
    "tts_audio_duration_seconds", "Generated audio duration",
    [0.5, 1, 2, 5, 10, 20, 40, 80, 160],
)

# TTFA is the metric a voice agent feels; the buckets are placed around the
# measured operating range (40-500 ms) rather than spread evenly, so the
# p50/p90 a dashboard reads are not interpolated across a decade-wide bucket.
TTS_TTFA_SECONDS = _histogram(
    "tts_ttfa_seconds", "Time from input.done to the first PCM frame (WebSocket)",
    [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
)
TTS_GENERATION_SECONDS = _histogram(
    "tts_generation_seconds", "Wall time to synthesize one chunk",
    [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
)
TTS_RTF = _histogram(
    "tts_rtf", "Real-time factor (generation time / audio duration); >1 is slower than realtime",
    [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4.0],
)

# --- streaming sessions ------------------------------------------------------

TTS_STREAMING_SESSIONS = _gauge("tts_streaming_sessions", "Open WebSocket sessions")
TTS_STREAMING_SENTENCES = _histogram(
    "tts_streaming_sentences", "Chunks synthesized per turn",
    [1, 2, 3, 5, 8, 13, 21],
)
WS_CLOSE_REASONS_TOTAL = _counter(
    "ws_close_reasons_total", "WebSocket session closes by reason", ["reason"],
)
TTS_CANCEL_TOTAL = _counter("tts_cancel_total", "Client barge-ins (cancel frames)")

# --- alignment (word timestamps) --------------------------------------------
# Same names vllm-omni exposes; see its metrics module for the semantics.

TTS_ALIGNMENT_HEAD_LOADED = _gauge(
    "tts_alignment_head_loaded", "1 if the AlignmentPointerHead sidecar loaded, 0 otherwise",
)
TTS_ALIGNMENT_REGISTER_TOTAL = _counter(
    "tts_alignment_register_total", "Per-request registrations into the alignment registry",
)
TTS_ALIGNMENT_FINALIZE_TOTAL = _counter(
    "tts_alignment_finalize_total", "finalize() calls by outcome", ["outcome"],
)
TTS_ALIGNMENT_POP_TOTAL = _counter(
    "tts_alignment_pop_total", "pop_word_alignment() calls by outcome", ["outcome"],
)
TTS_ALIGNMENT_POP_WAIT = _histogram(
    "tts_alignment_pop_wait_seconds", "Wall time spent polling in pop_word_alignment()",
    [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2],
)
TTS_ALIGNMENT_CANCEL_TOTAL = _counter(
    "tts_alignment_cancel_total", "Alignment registry cancels (barge-in / cleanup)",
)
TTS_ALIGNMENT_EVICT_TOTAL = _counter(
    "tts_alignment_evict_total", "Alignment entries evicted by TTL",
)
TTS_ALIGNMENT_INFLIGHT = _gauge("tts_alignment_inflight", "In-flight alignment registrations")
TTS_ALIGNMENT_FINALIZED = _gauge("tts_alignment_finalized", "Finalized, unread alignments")
TTS_ALIGNMENT_PARTIAL_TOTAL = _counter(
    "tts_alignment_partial_total", "Streaming commit-horizon snapshots by outcome", ["outcome"],
)
TTS_ALIGNMENT_PARTIAL_WORDS = _histogram(
    "tts_alignment_partial_words", "Words committed per published snapshot",
    [1, 2, 3, 5, 8, 13, 21, 34, 55],
)
TTS_ALIGNMENT_COMMIT_REVISION_TOTAL = _counter(
    "tts_alignment_commit_revision_total",
    "finalize() found the committed prefix contradicted by the batch decode. "
    "Non-zero means words were published mid-synthesis that the final decode "
    "disagrees with — raise the commit margin.",
)
TTS_ALIGNMENT_SCALE_RATIO = _histogram(
    "tts_alignment_scale_ratio", "last_word_end / audio_duration on emitted chunks",
    [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 2.0],
)


def render() -> tuple[bytes, str]:
    """Exposition payload for ``GET /metrics``."""
    if not _AVAILABLE:
        return (b"# prometheus_client not installed\n", "text/plain; version=0.0.4")
    return _generate_latest(REGISTRY), "text/plain; version=0.0.4; charset=utf-8"
