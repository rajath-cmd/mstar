"""REST /v1/audio/speech contract, asserted identically against either server.

L1 (protocol) and L2 (behavioural) only — see the parity spec. Nothing here
compares audio bytes or word TIMES between servers: the path exposes no seed and
its sampling is not greedy, so two runs of the SAME server on the SAME text
already differ (measured 2026-09-04). Cross-server byte identity is not a
property this system has, and asserting it would be asserting something false.

vllm-omni IS the contract. If an assertion fails there, the assertion is wrong.
"""

from __future__ import annotations

import httpx

# Frozen to vllm-omni's OpenAICreateSpeechRequest.
SPEECH_FIELDS = [
    "input", "model", "voice", "instructions", "response_format", "speed",
    "stream_format", "task_type", "language", "ref_audio", "ref_text",
    "x_vector_only_mode", "max_new_tokens", "stream", "temperature",
    "top_k", "top_p", "repetition_penalty", "timestamp_type",
]


def post_speech(base_url: str, body: dict, timeout: float = 240.0) -> httpx.Response:
    return httpx.post(f"{base_url}/v1/audio/speech", json=body, timeout=timeout)


def assert_accepts_every_field(base_url: str, voice: str) -> None:
    """Every documented field is accepted (no 4xx) — one at a time, so a
    rejection names the offending field instead of the whole body."""
    probe = {
        "instructions": "Speak calmly.", "language": "Auto", "speed": 1.0,
        "task_type": "CustomVoice", "max_new_tokens": 512, "temperature": 0.7,
        "top_k": 30, "top_p": 0.95, "repetition_penalty": 1.05,
        "response_format": "wav", "stream_format": "audio", "stream": False,
    }
    for field, value in probe.items():
        r = post_speech(base_url, {"input": "Hello there.", "voice": voice, field: value})
        assert r.status_code == 200, f"{field}={value!r} rejected: {r.status_code} {r.text[:200]}"


def assert_raw_audio_without_timestamps(base_url: str, voice: str) -> None:
    """No timestamp_type => raw container bytes, not JSON."""
    r = post_speech(base_url, {"input": "Hello there.", "voice": voice})
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"].startswith("audio/"), r.headers["content-type"]
    assert r.content[:4] == b"RIFF", f"expected a WAV container, got {r.content[:8]!r}"
    assert len(r.content) > 1000, "suspiciously small audio"


def assert_envelope_with_timestamps(base_url: str, voice: str) -> None:
    """timestamp_type='word' => JSON envelope with the documented shape."""
    r = post_speech(base_url, {"input": "Hello world.", "voice": voice, "timestamp_type": "word"})
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"].startswith("application/json"), r.headers["content-type"]
    body = r.json()
    for key in ("audio", "format", "sample_rate", "duration_seconds", "timestamp_info"):
        assert key in body, f"envelope missing {key!r}; got {sorted(body)}"
    assert body["audio"], "empty audio in envelope"
    info = body["timestamp_info"]
    if info is not None:  # null is legal: checkpoint with no alignment head
        wa = info["word_alignment"]
        assert set(wa) == {"words", "word_start_time_seconds", "word_end_time_seconds"}
        n = len(wa["words"])
        assert len(wa["word_start_time_seconds"]) == n
        assert len(wa["word_end_time_seconds"]) == n
        starts = wa["word_start_time_seconds"]
        assert starts == sorted(starts), "word starts are not monotone"


def assert_batch_returns_one_result_per_input(base_url: str, voice: str) -> None:
    """A list input returns an index-aligned results array."""
    r = post_speech(base_url, {"input": ["One.", "Two.", "Three."], "voice": voice})
    assert r.status_code == 200, r.text[:200]
    results = r.json()["results"]
    assert [x["index"] for x in results] == [0, 1, 2]
    for item in results:
        assert item["audio"], f"empty audio in batch item {item['index']}"


def assert_rejects_unknown_timestamp_type(base_url: str, voice: str) -> None:
    r = post_speech(base_url, {"input": "Hi.", "voice": voice, "timestamp_type": "phoneme"})
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text[:200]}"


def assert_lists_voices(base_url: str, voice: str) -> None:
    r = httpx.get(f"{base_url}/v1/audio/voices", timeout=60.0)
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert "voices" in body, f"missing 'voices'; got {sorted(body)}"
    assert isinstance(body["voices"], list) and body["voices"], "empty voice list"
    names = [v.lower() for v in body["voices"] if isinstance(v, str)]
    assert voice.lower() in names, f"{voice!r} absent from {names}"
