"""Word-timestamp (TFA) parity — currently RED for M*, by design.

Kept out of test_rest_parity.py on purpose. The REST contract permits
``timestamp_info: null``, because that is genuinely what vllm-omni returns for a
checkpoint shipping no ``pointer_head.pt``. That clause is correct, but once the
served checkpoint DOES have a head it would let M* return null forever and still
report green — the contract suite would mask the exact gap this program exists
to close.

So the capability gets its own file. Against a TFA-head checkpoint these assert
that real words come back; M* fails them until Phase 3 ports the alignment head,
and that failure is the tracking signal. When Phase 3 lands these turn green
without being edited.

Run against a checkpoint that HAS a head, e.g.
sft-cv-13l-14pv-ticr-1e-13l-fulldata_17946/checkpoint-final:

    MSTAR_TTS_URL=... OMNI_TTS_URL=... pytest test/parity/test_alignment_parity.py -v
"""

import pytest

from test.parity import contract_rest


def _alignment(base_url: str, voice: str, text: str = "Hello world. This is a test.") -> dict | None:
    r = contract_rest.post_speech(base_url, {"input": text, "voice": voice, "timestamp_type": "word"})
    assert r.status_code == 200, r.text[:200]
    return r.json().get("timestamp_info")


def test_returns_word_timestamps(server, voice):
    """A head-bearing checkpoint must produce words, not null."""
    name, url = server
    info = _alignment(url, voice)
    if info is None:
        pytest.fail(
            f"{name} returned timestamp_info=null against a TFA-head checkpoint. "
            "For M* this is the expected Phase 3 gap (no alignment head ported yet); "
            "for vllm-omni it means the head did not load — check the startup log for "
            "'loaded AlignmentPointerHead'."
        )
    assert info["word_alignment"]["words"], f"{name} returned an empty word list"


def test_word_times_are_monotone_and_bounded(server, voice):
    name, url = server
    info = _alignment(url, voice)
    if info is None:
        pytest.fail(f"{name}: no timestamps (see test_returns_word_timestamps)")
    wa = info["word_alignment"]
    words = wa["words"]
    starts = wa["word_start_time_seconds"]
    ends = wa["word_end_time_seconds"]
    assert len(starts) == len(ends) == len(words)
    assert starts == sorted(starts), f"{name}: word starts are not monotone"
    for w, s, e in zip(words, starts, ends, strict=True):
        assert e >= s, f"{name}: {w!r} ends ({e}) before it starts ({s})"
        assert s >= 0, f"{name}: {w!r} has a negative start"


def test_word_sequence_covers_the_script(server, voice):
    """Every spoken token of the script appears, in order.

    The word list comes from prompt tokenisation rather than sampling, so it is
    stable across runs and directly comparable between servers — unlike the
    times, which are not (measured 2026-09-04: two runs of the SAME server on
    the SAME text give different word times).
    """
    name, url = server
    info = _alignment(url, voice, "Alpha bravo charlie delta echo.")
    if info is None:
        pytest.fail(f"{name}: no timestamps (see test_returns_word_timestamps)")
    got = [w.lower().strip(".,") for w in info["word_alignment"]["words"]]
    for expected in ("alpha", "bravo", "charlie", "delta", "echo"):
        assert expected in got, f"{name}: {expected!r} missing from {got}"
