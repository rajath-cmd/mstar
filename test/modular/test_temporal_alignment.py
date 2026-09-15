"""Unit tests for the temporal_alignment subpackage + WordAlignmentRegistry.

Pure-Python (no GPU, no vLLM engine). Live-server validation lives in
tests/qwen3-tts/scribe_v2_compare.py.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from mstar.api_server.openai.protocol import WordAlignment
from mstar.model.qwen3_tts.temporal_alignment import registry as registry_module
from mstar.model.qwen3_tts.temporal_alignment.frame_rate import (
    CODEC_FRAME_RATE_HZ,
    frame_to_sec,
    sec_to_frame,
)
from mstar.model.qwen3_tts.temporal_alignment.pointer_head import (
    AlignmentPointerHead,
    pool_word_keys,
)
from mstar.model.qwen3_tts.temporal_alignment.readout import (
    word_timestamps,
)
from mstar.model.qwen3_tts.temporal_alignment.registry import (
    WordAlignmentRegistry,
    get_registry,
)
from mstar.model.qwen3_tts.temporal_alignment.validation import (
    ABS_DRIFT_ENV_VAR,
    DEFAULT_MAX_WORD_DUR_S,
    DEFAULT_SCALE_HI,
    DEFAULT_SCALE_LO,
    MAX_WORD_DUR_ENV_VAR,
    MIN_WPS_ENV_VAR,
    AlignmentCheck,
    abs_drift_max_s_from_env,
    max_word_dur_s_from_env,
    min_wps_from_env,
    validate_word_alignment,
)
from mstar.model.qwen3_tts.temporal_alignment.viterbi import (
    _to_word_prob_matrix,
    monotonic_viterbi,
    monotonic_viterbi_full_coverage,
    word_timestamps_viterbi,
    word_timestamps_viterbi_full_coverage,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
    segment_words,
    spoken_words,
)


class TestFrameRate:
    def test_constant_is_12_5(self):
        # The training side derives this from 24000/1920 = 12.5 — must
        # match or every word timestamp drifts.
        assert CODEC_FRAME_RATE_HZ == 12.5

    def test_round_trip(self):
        # One frame at 12.5 Hz = 0.08 s; round-trip drifts by ≤ half a frame.
        for t in (0.0, 0.08, 0.16, 1.0, 5.5):
            f = sec_to_frame(t)
            assert abs(frame_to_sec(f) - t) <= 0.04 + 1e-9

    def test_frame_zero_is_audio_zero(self):
        assert frame_to_sec(0) == 0.0
        assert sec_to_frame(0.0) == 0


class TestAlignmentPointerHead:
    def test_linear_shape(self):
        head = AlignmentPointerHead(hidden_size=64, proj_size=16, head_type="linear")
        fq = torch.randn(7, 64)
        keys = torch.randn(4, 64)
        scores = head(fq, keys)
        assert scores.shape == (7, 4)

    def test_mlp_shape(self):
        head = AlignmentPointerHead(hidden_size=64, proj_size=16, head_type="mlp")
        fq = torch.randn(3, 64)
        keys = torch.randn(5, 64)
        scores = head(fq, keys)
        assert scores.shape == (3, 5)

    def test_mlp_state_dict_keys(self):
        # The trained head ships as the MLP variant; the load_head loader
        # has to populate W_q.0/2.{weight,bias} and W_k.0/2.{weight,bias}.
        head = AlignmentPointerHead(hidden_size=8, proj_size=4, head_type="mlp")
        sd = head.state_dict()
        assert set(sd.keys()) == {
            "W_q.0.weight",
            "W_q.0.bias",
            "W_q.2.weight",
            "W_q.2.bias",
            "W_k.0.weight",
            "W_k.0.bias",
            "W_k.2.weight",
            "W_k.2.bias",
        }

    def test_scale_is_reciprocal_sqrt_proj(self):
        head = AlignmentPointerHead(hidden_size=16, proj_size=256, head_type="linear")
        # 1/sqrt(256) = 0.0625
        assert abs(head._scale - (1.0 / 16.0)) < 1e-6


class TestPoolWordKeys:
    def test_one_token_per_word(self):
        # 3 tokens, 3 words: pooling each word returns that token's row.
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        spans = [
            WordSpan("a", 0, 1, False),
            WordSpan("b", 1, 2, False),
            WordSpan("c", 2, 3, False),
        ]
        keys = pool_word_keys(hidden, spans)
        assert keys.shape == (3, 2)
        assert torch.allclose(keys[1], torch.tensor([3.0, 4.0]))

    def test_multi_token_word_means(self):
        hidden = torch.tensor([[1.0], [3.0], [5.0]])
        spans = [WordSpan("abc", 0, 3, False)]  # one word, three tokens
        keys = pool_word_keys(hidden, spans)
        assert keys.shape == (1, 1)
        assert keys[0].item() == pytest.approx(3.0)  # mean of 1+3+5

    def test_empty_spans_returns_empty(self):
        hidden = torch.zeros((5, 4))
        keys = pool_word_keys(hidden, [])
        assert keys.shape == (0, 4)


class _FakeTokenizer:
    """Minimal tokenizer for tests — vocab is just (id == token string).
    No convert_tokens_to_string so segment_words takes the naive-decode
    fallback path (verifies the fallback works for tests)."""

    def __init__(self, vocab: dict[int, str]) -> None:
        self._vocab = vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._vocab[i] for i in ids]


class TestWordSegmentation:
    def test_three_words(self):
        # Tokens: "Ġhello", "Ġworld", "Ġfoo"
        vocab = {0: "Ġhello", 1: "Ġworld", 2: "Ġfoo"}
        spans = segment_words([0, 1, 2], _FakeTokenizer(vocab))
        assert [s.word for s in spans] == ["hello", "world", "foo"]

    def test_multi_piece_word(self):
        # "Ġfoobar" tokenized as Ġfoo + bar (one word, two BPE pieces).
        vocab = {0: "Ġfoo", 1: "bar"}
        spans = segment_words([0, 1], _FakeTokenizer(vocab))
        assert [s.word for s in spans] == ["foobar"]
        assert spans[0].token_start == 0
        assert spans[0].token_end == 2

    def test_first_token_no_space_starts_word(self):
        vocab = {0: "hello", 1: "Ġworld"}
        spans = segment_words([0, 1], _FakeTokenizer(vocab))
        assert [s.word for s in spans] == ["hello", "world"]

    def test_tag_span_is_atomic(self):
        # A bracket tag must emit as ONE span with is_tag=True regardless
        # of how the inner content tokenizes.
        vocab = {0: "Ġhello", 1: "[", 2: "Ġlaughs", 3: "]", 4: "Ġworld"}
        spans = segment_words([0, 1, 2, 3, 4], _FakeTokenizer(vocab))
        # Spans: hello | [laughs] | world
        assert len(spans) == 3
        assert spans[0].word == "hello" and not spans[0].is_tag
        assert spans[1].is_tag and "laughs" in spans[1].word
        assert spans[2].word == "world" and not spans[2].is_tag

    def test_spoken_words_filters_tags(self):
        spans = [
            WordSpan("hi", 0, 1, False),
            WordSpan("[laughs]", 1, 4, True),
            WordSpan("world", 4, 5, False),
        ]
        assert [s.word for s in spoken_words(spans)] == ["hi", "world"]

    def test_bare_marker_token_still_starts_a_new_word(self):
        # Empirically observed real-tokenizer pattern for digit runs: the
        # leading-space marker arrives as its OWN token ("Ġ") instead of
        # fused onto the following digit ("Ġ2"). "1" "," "Ġ" "2" "," "Ġ" "3"
        # must still split into three words, not collapse into one "1,2,3".
        vocab = {0: "1", 1: ",", 2: "Ġ", 3: "2", 4: ",", 5: "Ġ", 6: "3"}
        spans = segment_words([0, 1, 2, 3, 4, 5, 6], _FakeTokenizer(vocab))
        assert [s.word for s in spans] == ["1,", "2,", "3"]

    def test_bare_marker_token_at_end_flushes_final_word(self):
        # A bare marker as the very last token must still flush the word
        # that was accumulating before it, rather than silently dropping it.
        vocab = {0: "Ġhi", 1: "Ġ"}
        spans = segment_words([0, 1], _FakeTokenizer(vocab))
        assert [s.word for s in spans] == ["hi"]


class TestMonotonicViterbi:
    def test_diagonal_attn_recovers_diagonal_path(self):
        # 4 frames, 4 words, perfect diagonal attention → path = [0,1,2,3].
        attn = np.eye(4, dtype=np.float64) * 0.9 + 0.025
        path = monotonic_viterbi(attn)
        assert path.tolist() == [0, 1, 2, 3]

    def test_stay_repeats_when_attn_concentrated(self):
        # 4 frames, 2 words. Frames 0+1 strongly point to word 0,
        # frames 2+3 to word 1.
        attn = np.array(
            [
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
            ]
        )
        path = monotonic_viterbi(attn)
        assert path.tolist() == [0, 0, 1, 1]

    def test_empty_inputs_are_safe(self):
        path = monotonic_viterbi(np.zeros((0, 5)))
        assert path.shape == (0,)
        path = monotonic_viterbi(np.zeros((5, 0)))
        assert path.shape == (5,)

    def test_monotonic_never_decreases(self):
        # Random non-negative matrix — output path must be non-decreasing
        # and advance by at most 1 per step (the DP's structural guarantee).
        rng = np.random.default_rng(42)
        attn = rng.random((50, 8))
        path = monotonic_viterbi(attn)
        for i in range(1, len(path)):
            assert path[i] >= path[i - 1]
            assert path[i] - path[i - 1] <= 1

    def test_softmax_normalization_for_negative_scores(self):
        # Negative logits → _to_word_prob_matrix softmaxes over word axis.
        scores = np.array([[1.0, 0.0], [-5.0, 5.0]])
        probs = _to_word_prob_matrix(scores)
        assert probs.shape == scores.shape
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=-1), 1.0)


class TestMonotonicViterbiFullCoverage:
    def test_recovers_word_the_free_endpoint_decode_drops(self):
        # 3 words, 20 frames. word0 dominant frames 0-9; words 1 and 2 both
        # only ever get weak, never-quite-dominant mass for the rest — word0
        # stays cheaper than paying advance_penalty twice, so free-endpoint
        # Viterbi never leaves column 0 at all. This is the exact mechanism
        # behind a live-reproduced bug where the audio genuinely contained
        # every script word (verbatim-verified) but the free decode only
        # ever reported the first.
        scores = np.zeros((20, 3))
        scores[:10, 0] = 1.0
        scores[10:, 0] = 0.34
        scores[10:, 1] = 0.33
        scores[10:, 2] = 0.33
        free_path = monotonic_viterbi(scores)
        assert list(set(free_path.tolist())) == [0]  # free decode: stuck on word0

        forced_path = monotonic_viterbi_full_coverage(scores)
        assert forced_path is not None
        assert forced_path[-1] == 2  # forced to end on the last column
        assert list(forced_path) == sorted(forced_path)  # still monotonic

        spans = [
            WordSpan("hi", 0, 1, False),
            WordSpan("there", 1, 2, False),
            WordSpan("world", 2, 3, False),
        ]
        out = word_timestamps_viterbi_full_coverage(scores, spans)
        assert [d["word"] for d in out] == ["hi", "there", "world"]

    def test_infeasible_when_fewer_frames_than_words_falls_back(self):
        # 2 words, 1 frame: reaching column 1 needs 1 transition, but there's
        # only 1 frame total (0 transitions available) — infeasible by
        # construction, not a tuning knob. Must fall back to the
        # free-endpoint decode rather than fabricate a path.
        scores = np.array([[0.9, 0.1]])
        assert monotonic_viterbi_full_coverage(scores) is None
        spans = [WordSpan("hi", 0, 1, False), WordSpan("there", 1, 2, False)]
        out = word_timestamps_viterbi_full_coverage(scores, spans)
        assert [d["word"] for d in out] == ["hi"]  # same as the free decode

    def test_matches_free_decode_when_endpoints_agree(self):
        # Clean, unambiguous case: both decodes should produce the same path.
        scores = np.zeros((10, 2))
        scores[:5, 0] = 1.0
        scores[5:, 1] = 1.0
        assert list(monotonic_viterbi_full_coverage(scores)) == list(monotonic_viterbi(scores))

    def test_empty_inputs_are_safe(self):
        assert monotonic_viterbi_full_coverage(np.zeros((0, 0))).shape == (0,)
        assert word_timestamps_viterbi_full_coverage(np.zeros((0, 0)), []) == []


class TestWordTimestampsReadout:
    def test_argmax_assigns_words_to_frames(self):
        # 4 frames, 2 words. argmax: word 0 for frames 0+1, word 1 for 2+3.
        scores = torch.tensor(
            [
                [10.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [0.0, 10.0],
            ]
        )
        spans = [WordSpan("hi", 0, 1, False), WordSpan("you", 1, 2, False)]
        out = word_timestamps(scores, spans, rate=12.5)
        assert [d["word"] for d in out] == ["hi", "you"]
        # Frame 0 -> 0.0s; frame 2 -> 0.16s. End of word 0 = frame 2 ->
        # 0.16s. End of word 1 = frame 4 -> 0.32s.
        assert out[0]["start"] == pytest.approx(0.0)
        assert out[0]["end"] == pytest.approx(0.16, abs=0.005)
        assert out[1]["start"] == pytest.approx(0.16, abs=0.005)
        assert out[1]["end"] == pytest.approx(0.32, abs=0.005)

    def test_viterbi_mirrors_argmax_when_already_monotone(self):
        scores = torch.tensor(
            [
                [10.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [0.0, 10.0],
            ]
        )
        spans = [WordSpan("hi", 0, 1, False), WordSpan("you", 1, 2, False)]
        ts_argmax = word_timestamps(scores, spans)
        ts_viterbi = word_timestamps_viterbi(scores, spans)
        assert ts_argmax == ts_viterbi


def _make_registry_state(text: str = "hi you"):
    """Helper: build the args for register() that work with a fake tokenizer."""
    # Two words, one token each. token_start/end positions are just bookkeeping.
    fake_tok = _FakeTokenizer({0: "Ġhi", 1: "Ġyou"})
    head = AlignmentPointerHead(hidden_size=8, proj_size=4, head_type="linear")
    return {
        "text_token_ids": [0, 1],
        "text_token_start": 10,
        "text_token_end": 12,
        "layer": 3,
        "tokenizer": fake_tok,
        "head": head,
    }


class TestWordAlignmentRegistry:
    def test_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_register_then_pop_returns_none_before_finalize(self):
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        assert reg.is_registered("req-a")
        # Not finalized yet → pop returns None.
        assert reg.pop_word_alignment("req-a") is None

    def test_finalize_without_frames_returns_none(self):
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        # No accumulate_decode calls → no frames → finalize returns None,
        # registry doesn't stash anything.
        assert reg.finalize("req-a") is None
        assert reg.pop_word_alignment("req-a") is None

    def test_finalize_without_prefill_returns_none(self):
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        # Decode frames but no prefill captures → cannot build word keys.
        reg.accumulate_decode("req-a", torch.randn(8))
        assert reg.finalize("req-a") is None

    def test_round_trip_produces_word_alignment(self):
        reg = WordAlignmentRegistry()
        state = _make_registry_state()
        reg.register("req-a", **state)

        # Simulate a prefill chunk that covers the text-token range.
        # Chunk spans positions [8, 14) of the request's full prompt; the
        # text-token range is [10, 12). The registry should slice rows
        # 2 and 3 of the chunk hidden (= absolute positions 10 and 11).
        chunk_hidden = torch.arange(6 * 8).reshape(6, 8).float()
        reg.accumulate_prefill("req-a", chunk_hidden, 8, 14)

        # Three decode steps' frame queries.
        for _ in range(3):
            reg.accumulate_decode("req-a", torch.randn(8))

        alignment = reg.finalize("req-a")
        assert isinstance(alignment, WordAlignment)
        assert len(alignment.words) <= 2  # at most one per spoken-word span
        # pop drains the result.
        popped = reg.pop_word_alignment("req-a")
        assert popped == alignment
        assert reg.pop_word_alignment("req-a") is None  # already popped

    def test_request_isolation(self):
        reg = WordAlignmentRegistry()
        state = _make_registry_state()
        reg.register("req-a", **state)
        reg.register("req-b", **_make_registry_state())

        # Only req-a gets decode frames.
        chunk = torch.arange(6 * 8).reshape(6, 8).float()
        reg.accumulate_prefill("req-a", chunk, 8, 14)
        reg.accumulate_decode("req-a", torch.randn(8))
        reg.accumulate_decode("req-a", torch.randn(8))

        a_align = reg.finalize("req-a")
        b_align = reg.finalize("req-b")
        assert isinstance(a_align, WordAlignment)
        assert b_align is None  # b had no captures

    def test_cancel_clears_state(self):
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        reg.cancel("req-a")
        assert not reg.is_registered("req-a")
        assert reg.pop_word_alignment("req-a") is None

    def test_accumulate_after_cancel_is_noop(self):
        # Belt-and-suspenders — accumulate shouldn't blow up if the
        # request was cancelled between register and the post-forward
        # dispatch (this CAN happen with very fast barge-ins).
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        reg.cancel("req-a")
        reg.accumulate_decode("req-a", torch.randn(8))  # silently dropped
        reg.accumulate_prefill("req-a", torch.randn(4, 8), 0, 4)
        assert not reg.is_registered("req-a")

    def test_stats_reflect_state(self):
        reg = WordAlignmentRegistry()
        assert reg.stats() == {"inflight": 0, "finalized": 0}
        reg.register("a", **_make_registry_state())
        reg.register("b", **_make_registry_state())
        assert reg.stats()["inflight"] == 2

    def test_prefill_chunk_outside_text_range_is_skipped(self):
        # Chunk before text: positions 0..5, text at [10, 12) → no rows.
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        chunk = torch.randn(5, 8)
        reg.accumulate_prefill("req-a", chunk, 0, 5)
        # No text captures → finalize returns None even with decode frames.
        reg.accumulate_decode("req-a", torch.randn(8))
        assert reg.finalize("req-a") is None

    def test_unread_finalized_alignment_expires_via_ttl(self, monkeypatch, tmp_path):
        """An alignment nobody pops (client hung up before the timestamps
        frame) must not outlive the TTL — and its disk-IPC drop must not
        resurrect it afterwards."""
        monkeypatch.setattr(registry_module, "_DISK_IPC_DIR", tmp_path)
        reg = WordAlignmentRegistry()
        reg.register("req-a", **_make_registry_state())
        chunk = torch.arange(6 * 8).reshape(6, 8).float()
        reg.accumulate_prefill("req-a", chunk, 8, 14)
        reg.accumulate_decode("req-a", torch.randn(8))

        assert reg.finalize("req-a") is not None
        assert reg.stats()["finalized"] == 1
        assert (tmp_path / "req-a.json").exists()

        # Force-expire; eviction runs on the register()/finalize() heartbeat.
        monkeypatch.setattr(registry_module, "_REGISTRY_TTL_SECONDS", -1)
        reg.register("req-b", **_make_registry_state())

        assert reg.stats()["finalized"] == 0
        assert not (tmp_path / "req-a.json").exists()
        assert reg.pop_word_alignment("req-a", wait_timeout_s=0) is None


class TestEmitGuard:
    """Defense-in-depth guard applied at the `timestamps` emit point.

    The registry finalize path was observed to *transiently* return an
    alignment whose word list is correct but whose times are compressed
    toward zero (scale_ratio far below 1.0). The serving layer validates
    every alignment against the model-audio duration it emitted for the
    chunk and drops the timestamps frame when the alignment is physically
    impossible. These tests pin that guard.
    """

    def test_clean_full_coverage_passes_scale_one(self):
        # Full-coverage alignment ends exactly at the audio end → 1.0.
        chk = validate_word_alignment(["hello", "world"], [0.0, 0.5], [0.5, 1.0], audio_duration_s=1.0)
        assert chk.ok
        assert chk.scale_ratio == pytest.approx(1.0)

    def test_clean_within_band_passes(self):
        # A hair of trailing silence (last word ends at 0.92 of 1.0 s).
        chk = validate_word_alignment(["a", "b"], [0.0, 0.46], [0.44, 0.92], audio_duration_s=1.0)
        assert chk.ok
        assert DEFAULT_SCALE_LO <= chk.scale_ratio <= DEFAULT_SCALE_HI

    def test_compressed_times_suppressed(self):
        # THE observed corruption: words fine, times crushed toward 0.
        chk = validate_word_alignment(["a", "b"], [0.0, 0.02], [0.01, 0.04], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason.startswith("scale_ratio_out_of_range")
        assert chk.scale_ratio == pytest.approx(0.04)

    def test_scale_just_below_band_suppressed(self):
        chk = validate_word_alignment(["a"], [0.0], [0.84], audio_duration_s=1.0)
        assert not chk.ok

    def test_scale_just_inside_band_passes(self):
        chk = validate_word_alignment(["a"], [0.0], [0.86], audio_duration_s=1.0)
        assert chk.ok

    def test_non_monotonic_starts_suppressed(self):
        chk = validate_word_alignment(["a", "b", "c"], [0.0, 0.9, 0.3], [0.4, 1.0, 0.6], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason.startswith("non_monotonic_start")

    def test_end_before_start_suppressed(self):
        chk = validate_word_alignment(["a", "b"], [0.0, 0.5], [0.5, 0.3], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason.startswith("end_before_start")

    def test_time_past_audio_suppressed(self):
        chk = validate_word_alignment(["a", "b"], [0.0, 0.9], [0.5, 3.0], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason.startswith("past_audio")

    def test_nan_time_suppressed(self):
        chk = validate_word_alignment(["a"], [float("nan")], [0.5], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason == "non_finite_start"

    def test_negative_time_suppressed(self):
        chk = validate_word_alignment(["a"], [-0.5], [0.4], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason == "negative_start"

    def test_length_mismatch_suppressed(self):
        chk = validate_word_alignment(["a", "b"], [0.0], [0.5, 1.0], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason.startswith("length_mismatch")

    def test_empty_suppressed(self):
        chk = validate_word_alignment([], [], [], audio_duration_s=1.0)
        assert not chk.ok
        assert chk.reason == "empty"

    def test_unknown_audio_duration_skips_scale_but_keeps_structure(self):
        # No audio ref → structural checks still run, scale skipped.
        ok = validate_word_alignment(["a", "b"], [0.0, 0.5], [0.5, 1.0], audio_duration_s=None)
        assert ok.ok and ok.reason == "ok_no_audio_ref"
        bad = validate_word_alignment(["a", "b"], [0.0, 0.5], [0.5, 0.3], audio_duration_s=None)
        assert not bad.ok

    def test_small_epsilon_tolerated(self):
        # Sub-frame rounding must not trip end<start or monotonicity.
        chk = validate_word_alignment(["a", "b"], [0.0, 0.50], [0.50, 1.0], audio_duration_s=1.0)
        assert chk.ok
        # ends[i] a hair below starts[i] within eps → still ok (audio
        # duration chosen so the scale check stays in-band and isolates
        # the end>=start epsilon tolerance).
        chk2 = validate_word_alignment(["a"], [0.5], [0.48], audio_duration_s=0.5)
        assert chk2.ok

    def test_returns_alignmentcheck_type(self):
        assert isinstance(validate_word_alignment(["a"], [0.0], [1.0], 1.0), AlignmentCheck)


class TestAbsoluteDriftCap:
    """Change 2: the OPTIONAL absolute per-chunk drift cap.

    The existing scale guard is RELATIVE, so on a LONG chunk a mild transient
    can drift by whole seconds while ``scale_ratio`` stays in-band and passes.
    This cap is GATED OFF BY DEFAULT (``abs_drift_max_s=None`` ⇒ strict no-op)
    and only rejects extra chunks when explicitly enabled. It is UNVERIFIED —
    the -1.6s corruption does not reproduce on a fresh server.
    """

    # A LONG chunk whose scale_ratio (0.90) is comfortably inside the relative
    # band [0.85, 1.15], yet whose ABSOLUTE end-time drift is 0.10 * 30 = 3.0s.
    # These synthetic 2-word/30s chunks have multi-second per-word durations, so
    # the (default-on) runaway guard is disabled here to isolate the abs-drift
    # cap under test; real 30s chunks have ~60 sub-second words.
    _WORDS = ["a", "b"]
    _STARTS = [0.0, 13.0]
    _ENDS = [13.0, 27.0]  # last_word_end = 27.0, audio = 30.0 → scale 0.90
    _AUDIO = 30.0

    def test_default_off_is_a_noop_for_in_band_long_drift(self):
        # Default (no abs_drift_max_s arg) must accept exactly what the
        # relative guard accepts — the long in-band-but-drifted chunk passes.
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, self._AUDIO, max_word_dur_s=None)
        assert chk.ok
        assert chk.scale_ratio == pytest.approx(0.90)

    def test_explicit_none_is_a_noop(self):
        chk = validate_word_alignment(
            self._WORDS, self._STARTS, self._ENDS, self._AUDIO, abs_drift_max_s=None, max_word_dur_s=None
        )
        assert chk.ok

    def test_flag_on_rejects_in_band_long_drift(self):
        # Same chunk, cap enabled at 0.4s: 3.0s absolute drift > 0.4s → drop.
        chk = validate_word_alignment(
            self._WORDS, self._STARTS, self._ENDS, self._AUDIO, abs_drift_max_s=0.4, max_word_dur_s=None
        )
        assert not chk.ok
        assert chk.reason.startswith("abs_drift")
        # scale_ratio is still reported for observability.
        assert chk.scale_ratio == pytest.approx(0.90)

    def test_flag_on_passes_when_drift_under_threshold(self):
        # A long chunk that barely drifts: scale 0.99 on 30s → 0.3s < 0.4s.
        chk = validate_word_alignment(
            ["a", "b"], [0.0, 14.0], [14.0, 29.7], 30.0, abs_drift_max_s=0.4, max_word_dur_s=None
        )
        assert chk.ok
        assert chk.scale_ratio == pytest.approx(0.99)

    def test_flag_on_still_accepts_clean_short_chunk(self):
        # The common case (full-coverage, scale ~1.0) is untouched by the cap.
        chk = validate_word_alignment(["hello", "world"], [0.0, 0.5], [0.5, 1.0], 1.0, abs_drift_max_s=0.4)
        assert chk.ok

    def test_flag_on_relative_guard_still_fires_first(self):
        # An out-of-band ratio is rejected by the relative guard regardless of
        # the cap (the cap only ADDS rejections, never removes them).
        chk = validate_word_alignment(["a", "b"], [0.0, 0.02], [0.01, 0.04], 1.0, abs_drift_max_s=0.4)
        assert not chk.ok
        assert chk.reason.startswith("scale_ratio_out_of_range")


class TestAbsDriftEnvReader:
    """``abs_drift_max_s_from_env`` — default-disabled, fail-safe parsing."""

    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(ABS_DRIFT_ENV_VAR, raising=False)
        assert abs_drift_max_s_from_env() is None

    def test_positive_value_parsed(self, monkeypatch):
        monkeypatch.setenv(ABS_DRIFT_ENV_VAR, "0.4")
        assert abs_drift_max_s_from_env() == pytest.approx(0.4)

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv(ABS_DRIFT_ENV_VAR, "0")
        assert abs_drift_max_s_from_env() is None

    def test_negative_disables(self, monkeypatch):
        monkeypatch.setenv(ABS_DRIFT_ENV_VAR, "-1.0")
        assert abs_drift_max_s_from_env() is None

    def test_garbage_disables(self, monkeypatch):
        monkeypatch.setenv(ABS_DRIFT_ENV_VAR, "not-a-float")
        assert abs_drift_max_s_from_env() is None

    def test_empty_disables(self, monkeypatch):
        monkeypatch.setenv(ABS_DRIFT_ENV_VAR, "   ")
        assert abs_drift_max_s_from_env() is None


class TestRunawayGuard:
    """The runaway per-word-duration ceiling (default 8.0 s, ENABLED).

    A stochastic model repetition-loop collapses into a single word whose
    DURATION spans the garbage tail (calibration: clean max 4.08 s; runaways
    5.44 / 14.5 / 71.5 s), while the last word still lands at the audio end so
    ``scale_ratio ≈ 1.0`` and the relative guard passes it. This catches that.
    """

    # A real runaway shape (cf. captured dump p5_r3 chunk 3): the first word is
    # normal, the second absorbs a ~71 s garbage tail. scale_ratio = 79.5/80 =
    # 0.994 → INSIDE the relative band, so ONLY the runaway guard can catch it.
    _WORDS = ["under", "concurrent"]
    _STARTS = [0.0, 8.0]
    _ENDS = [0.5, 79.5]  # word 2 duration = 71.5 s
    _AUDIO = 80.0

    def test_default_on_rejects_runaway_word(self):
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, self._AUDIO)
        assert not chk.ok
        assert chk.reason.startswith("runaway_word_dur")

    def test_runaway_would_otherwise_pass_relative_guard(self):
        # Prove the relative scale guard is blind to it: with the runaway guard
        # disabled, scale_ratio ≈ 0.994 is in-band and the chunk passes.
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, self._AUDIO, max_word_dur_s=None)
        assert chk.ok
        assert chk.scale_ratio == pytest.approx(0.994, abs=1e-3)

    def test_runs_without_audio_reference(self):
        # The guard is intrinsic to the alignment — no audio duration needed.
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, audio_duration_s=None)
        assert not chk.ok
        assert chk.reason.startswith("runaway_word_dur")

    def test_clean_word_at_calibrated_max_passes(self):
        # The observed clean max per-word duration is 4.08 s — well under 8.0.
        chk = validate_word_alignment(["a", "b"], [0.0, 4.5], [4.08, 5.0], audio_duration_s=5.0)
        assert chk.ok

    def test_ceiling_is_eight_seconds(self):
        # Boundary: a 7.9 s word passes, an 8.1 s word is rejected.
        below = validate_word_alignment(["w"], [0.0], [7.9], audio_duration_s=8.0)
        assert below.ok
        above = validate_word_alignment(["w"], [0.0], [8.1], audio_duration_s=8.2)
        assert not above.ok
        assert above.reason.startswith("runaway_word_dur")

    def test_disabled_via_none_allows_runaway(self):
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, self._AUDIO, max_word_dur_s=None)
        assert chk.ok

    def test_custom_ceiling_via_param(self):
        # A tighter ceiling (5.0 s) catches the mild 5.44 s runaway flavor.
        mild = validate_word_alignment(["a", "b"], [0.0, 1.0], [0.5, 6.44], 7.0, max_word_dur_s=5.0)
        assert not mild.ok
        assert mild.reason.startswith("runaway_word_dur")

    def test_structural_checks_still_fire_first(self):
        # A non-monotonic alignment is rejected structurally before the runaway
        # check even when it also has a long word (ordering is stable).
        chk = validate_word_alignment(["a", "b"], [5.0, 0.0], [6.0, 71.0], audio_duration_s=72.0)
        assert not chk.ok
        assert chk.reason.startswith("non_monotonic_start")


class TestRunawayLowWpsFloor:
    """The OPTIONAL words-per-second floor (default None ⇒ OFF)."""

    # 6 words uniformly smeared across 30 s → 0.20 wps. No single word exceeds
    # the duration ceiling (each ~0.1 s with 5 s gaps), so only the wps floor
    # can catch this uniform-smear flavor.
    _WORDS = ["one", "two", "three", "four", "five", "six"]
    _STARTS = [0.0, 6.0, 12.0, 18.0, 24.0, 29.9]
    _ENDS = [0.1, 6.1, 12.1, 18.1, 24.1, 30.0]
    _AUDIO = 30.0

    def test_off_by_default_passes_uniform_smear(self):
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, self._AUDIO)
        assert chk.ok  # runaway word-dur guard alone does not catch uniform smear

    def test_floor_on_rejects_low_wps(self):
        chk = validate_word_alignment(self._WORDS, self._STARTS, self._ENDS, self._AUDIO, min_wps=0.8)
        assert not chk.ok
        assert chk.reason.startswith("runaway_low_wps")

    def test_floor_on_passes_normal_speech(self):
        # ~2.3 wps normal speech is well above a 0.8 floor.
        words = ["a", "b", "c", "d", "e"]
        starts = [0.0, 0.4, 0.8, 1.2, 1.6]
        ends = [0.4, 0.8, 1.2, 1.6, 2.0]
        chk = validate_word_alignment(words, starts, ends, 2.0, min_wps=0.8)
        assert chk.ok

    def test_floor_ignored_below_min_word_count(self):
        # Fewer than the min word count → wps not meaningful → not applied.
        chk = validate_word_alignment(["a", "b"], [0.0, 5.0], [0.1, 5.1], 6.0, min_wps=0.8)
        assert chk.ok


class TestRunawayEnvReaders:
    """``max_word_dur_s_from_env`` (default 8.0) and ``min_wps_from_env`` (off)."""

    def test_max_word_dur_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv(MAX_WORD_DUR_ENV_VAR, raising=False)
        assert max_word_dur_s_from_env() == pytest.approx(DEFAULT_MAX_WORD_DUR_S)

    def test_max_word_dur_positive_value_parsed(self, monkeypatch):
        monkeypatch.setenv(MAX_WORD_DUR_ENV_VAR, "5.0")
        assert max_word_dur_s_from_env() == pytest.approx(5.0)

    def test_max_word_dur_zero_disables(self, monkeypatch):
        monkeypatch.setenv(MAX_WORD_DUR_ENV_VAR, "0")
        assert max_word_dur_s_from_env() is None

    def test_max_word_dur_garbage_disables(self, monkeypatch):
        monkeypatch.setenv(MAX_WORD_DUR_ENV_VAR, "not-a-float")
        assert max_word_dur_s_from_env() is None

    def test_min_wps_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(MIN_WPS_ENV_VAR, raising=False)
        assert min_wps_from_env() is None

    def test_min_wps_positive_value_parsed(self, monkeypatch):
        monkeypatch.setenv(MIN_WPS_ENV_VAR, "0.8")
        assert min_wps_from_env() == pytest.approx(0.8)

    def test_min_wps_zero_disables(self, monkeypatch):
        monkeypatch.setenv(MIN_WPS_ENV_VAR, "0")
        assert min_wps_from_env() is None


# TestEmitScaleRatioLogging lived here upstream. It exercises
# OmniStreamingSpeechHandler._log_emit_scale_ratio — a vllm-omni SERVING-layer
# diagnostic, not part of the alignment package this file covers. It returns
# alongside M*'s own timestamp-emit path.

