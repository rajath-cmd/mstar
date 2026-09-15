"""Physical sanity-guard for a word alignment vs. the audio emitted.

The pointer-head + Viterbi finalize path in :mod:`registry` returns, for
each request, per-word start/end times derived from the number of decode
frames captured for that request (``end = n_frames / rate``). On a freshly
provisioned server this is exact: the last word ends at the end of the
emitted audio, so ``scale_ratio = last_word_end / audio_duration == 1.0``
(observed n=660, scale_ratio exactly 1.000, MAE 70-108 ms).

A long-lived server instance was observed to *transiently* corrupt this:
the word LIST stays correct but the TIMES compress toward zero
(``scale_ratio`` far below 1.0, downstream alignment MAE in the seconds).
The corruption is per-instance and sticky, i.e. a race / state-leak in the
per-request alignment registry, NOT a deterministic math bug (a math bug
would break every run). The precise leak has not been pinned from static
analysis; see the module docstring in ``registry.py`` and the PR notes.

Until that is fixed upstream, the serving layer validates every alignment
against the audio it actually emitted for that chunk and DROPS the
``timestamps`` frame (still emitting the audio) when the alignment is
physically impossible. The inf2-evals client treats a missing
``word_alignment`` as "excluded", so this converts silent corruption into
a clean, honest no-op at the source.

This module is deliberately pure-Python (no torch, no vLLM engine) so the
guard is unit-testable without a GPU.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

# The last word must end within this fraction of the emitted audio
# duration. A clean full-coverage alignment lands at exactly 1.0 (the last
# frame == the last audio frame), so this band is generous on both sides:
# real speech may end a hair before the trailing silence (low side), and a
# little rounding slack is allowed above.
DEFAULT_SCALE_LO: float = 0.85
DEFAULT_SCALE_HI: float = 1.15

# Absolute per-time slack (seconds): keeps rounding / tiny clips from
# tripping the monotonicity, non-negativity and end>=start checks.
DEFAULT_EPS_S: float = 0.05

# ── ABSOLUTE per-chunk drift cap (UNVERIFIED, default DISABLED) ──────────────
#
# The existing scale guard is RELATIVE: both the ``scale_ratio`` band and the
# ``past_tol = 0.15 * audio_duration`` slack scale with the chunk length. On a
# LONG single chunk a mild transient can therefore emit *seconds* of absolute
# word drift while ``scale_ratio`` stays inside [0.85, 1.15] and the chunk
# passes. This adds an optional ABSOLUTE bound: reject when
# ``abs(1 - scale_ratio) * audio_duration_s`` (the absolute end-time drift, in
# seconds) exceeds a threshold.
#
# IMPORTANT — this guard is UNVERIFIED. The -1.6s corruption it targets is a
# rare, transient, per-server-instance state-leak that does NOT reproduce on a
# fresh server (~120 syntheses clean to ~80ms), so we cannot yet confirm this
# bound catches the real event without also dropping legitimate long chunks.
# It is therefore GATED OFF BY DEFAULT (``abs_drift_max_s=None`` ⇒ no-op) and
# only becomes enable-able once the success-path scale_ratio logging
# (``[ALIGN_SCALE]`` lines in ``serving_speech_stream``) yields live evidence
# of the abnormal scale_ratio distribution on long chunks. Enable via the env
# var below; leave unset in production until validated.
ABS_DRIFT_ENV_VAR: str = "QWEN3_TTS_ALIGN_ABS_DRIFT_MAX_S"
DEFAULT_ABS_DRIFT_MAX_S: float | None = None

# ── RUNAWAY plausibility guard (CALIBRATED, default ENABLED) ─────────────────
#
# A separate corruption mode from the transient state-leak above: a *stochastic
# model runaway*. Under temperature sampling with no repetition penalty the
# talker occasionally fails to terminate and loops, emitting tens of seconds of
# repeated garbage codec frames for a chunk. The pointer-head alignment is
# monotonic over ALL decode frames, so the runaway collapses into a single word
# whose DURATION spans the whole garbage tail — while the last word still lands
# at the audio end, so ``scale_ratio ≈ 1.0`` and the RELATIVE guard above passes
# it. Those stretched times then pollute the downstream timestamp metric.
#
# The intrinsic signature is a physically-impossible per-word duration: real
# speech words are sub-2 s; a runaway word is 5–70 s. This guard rejects a chunk
# whose maximum per-word duration exceeds ``max_word_dur_s``.
#
# CALIBRATION (245 captured chunks, 30 long utterances): clean max-per-word
# duration p99 = 2.88 s, MAX = 4.08 s; runaway = {5.44, 14.5, 71.5} s. The
# default 8.0 s sits ~2× above the observed clean max and catches the severe
# (metric-dominating) runaways with margin. Unlike the abs-drift cap this is
# calibrated against real clean+runaway data, so it is ENABLED by default; tune
# or disable via the env var. A value of 0 / non-positive / NaN disables it.
MAX_WORD_DUR_ENV_VAR: str = "QWEN3_TTS_ALIGN_MAX_WORD_DUR_S"
DEFAULT_MAX_WORD_DUR_S: float | None = 8.0

# Optional secondary runaway signal: an implausibly low words-per-second over
# the chunk (uniform-smear runaways, where the words spread evenly rather than
# collapsing into one). Clean speech is ≥ ~1.4 wps; runaways reach 0.2–1.1 wps.
# The clean/runaway separation on wps is THIN (clean min 1.37 vs runaway 1.09),
# so this is OFF by default (opt-in) to avoid dropping legitimate slow chunks;
# it only applies to chunks with at least ``_MIN_WORDS_FOR_WPS`` words.
MIN_WPS_ENV_VAR: str = "QWEN3_TTS_ALIGN_MIN_WPS"
DEFAULT_MIN_WPS: float | None = None
_MIN_WORDS_FOR_WPS: int = 4


def _positive_float_from_env(env_var: str, default: float | None) -> float | None:
    """Shared reader: returns a positive float from ``env_var`` or ``default``.

    An unset/blank var returns ``default``. A set-but-non-positive / NaN /
    unparsable value returns ``None`` (guard DISABLED) so a misconfigured env
    var can only ever *weaken* a guard, never silently corrupt behavior.
    """
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not (val == val) or val <= 0:  # NaN or non-positive ⇒ disabled  # noqa: PLR0124 (NaN test: NaN != itself)
        return None
    return val


def max_word_dur_s_from_env() -> float | None:
    """Read the runaway per-word-duration ceiling (seconds) from the env.

    Defaults to ``DEFAULT_MAX_WORD_DUR_S`` (guard ENABLED). Set the env var to
    a positive float to tune it, or to ``0`` to disable.
    """
    return _positive_float_from_env(MAX_WORD_DUR_ENV_VAR, DEFAULT_MAX_WORD_DUR_S)


def min_wps_from_env() -> float | None:
    """Read the optional runaway words-per-second floor from the env.

    Defaults to ``DEFAULT_MIN_WPS`` (``None`` ⇒ OFF). Set the env var to a
    positive float to enable it.
    """
    return _positive_float_from_env(MIN_WPS_ENV_VAR, DEFAULT_MIN_WPS)


def abs_drift_max_s_from_env() -> float | None:
    """Read the absolute-drift threshold (seconds) from the environment.

    Returns ``None`` (guard DISABLED — current default behavior) unless
    ``QWEN3_TTS_ALIGN_ABS_DRIFT_MAX_S`` is set to a positive float. A value of
    ``0`` or a non-positive / unparsable value also disables the guard, so a
    misconfigured env var can never silently start dropping timestamps.
    """
    raw = os.environ.get(ABS_DRIFT_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_ABS_DRIFT_MAX_S
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_ABS_DRIFT_MAX_S
    if not (val == val) or val <= 0:  # NaN or non-positive ⇒ disabled  # noqa: PLR0124 (NaN test: NaN != itself)
        return DEFAULT_ABS_DRIFT_MAX_S
    return val


@dataclass(frozen=True)
class AlignmentCheck:
    """Outcome of :func:`validate_word_alignment`.

    ``ok`` — safe to emit. ``reason`` — machine-readable tag for logging.
    ``scale_ratio`` — last_word_end / audio_duration when computable.
    """

    ok: bool
    reason: str
    scale_ratio: float | None = None


def _is_finite(v: float) -> bool:
    # NaN != NaN; inf/-inf compare out of any finite band.
    return v == v and v not in (float("inf"), float("-inf"))  # noqa: PLR0124 (NaN test: NaN != itself)


def validate_word_alignment(
    words: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    audio_duration_s: float | None,
    *,
    scale_lo: float = DEFAULT_SCALE_LO,
    scale_hi: float = DEFAULT_SCALE_HI,
    eps_s: float = DEFAULT_EPS_S,
    abs_drift_max_s: float | None = DEFAULT_ABS_DRIFT_MAX_S,
    max_word_dur_s: float | None = DEFAULT_MAX_WORD_DUR_S,
    min_wps: float | None = DEFAULT_MIN_WPS,
) -> AlignmentCheck:
    """Return whether this chunk-local alignment is physically consistent.

    ``words`` / ``starts`` / ``ends`` are the per-word arrays for ONE chunk,
    in chunk-local seconds (t=0 == the chunk's first model PCM sample), i.e.
    BEFORE any turn-timeline offset is added. ``audio_duration_s`` is the
    model-audio duration emitted for that same chunk (excludes leading
    inter-chunk silence). Pass ``None`` when the duration is unknown — the
    scale/bounds checks are then skipped but the structural checks still run.

    A ``False`` result means: do NOT emit a ``timestamps`` frame for this
    chunk (emit the audio normally).

    ``abs_drift_max_s`` is the OPTIONAL absolute per-chunk drift cap. When
    ``None`` (the default) it is a strict no-op — every alignment the relative
    guard accepts is still accepted. When set to a positive float, a chunk is
    additionally rejected if ``abs(1 - scale_ratio) * audio_duration_s`` (the
    absolute end-time drift in seconds) exceeds it, catching a long chunk whose
    ``scale_ratio`` stays in-band yet drifts by whole seconds. See
    ``ABS_DRIFT_ENV_VAR`` — this guard is UNVERIFIED and off by default.

    ``max_word_dur_s`` is the RUNAWAY per-word-duration ceiling (default 8.0 s,
    ENABLED): reject when any word's ``end - start`` exceeds it. This catches a
    model repetition-loop that ``scale_ratio ≈ 1.0`` is blind to (the runaway
    collapses into one multi-second word). ``min_wps`` is an OPTIONAL secondary
    words-per-second floor (default ``None`` ⇒ off). Both are intrinsic to the
    alignment and run even when ``audio_duration_s`` is ``None``. See
    ``MAX_WORD_DUR_ENV_VAR`` / ``MIN_WPS_ENV_VAR``.
    """
    n = len(words)
    if n == 0:
        return AlignmentCheck(False, "empty")
    if len(starts) != n or len(ends) != n:
        return AlignmentCheck(
            False,
            f"length_mismatch(words={n},starts={len(starts)},ends={len(ends)})",
        )

    # Finite + non-negative times.
    for name, arr in (("start", starts), ("end", ends)):
        for v in arr:
            if not _is_finite(float(v)):
                return AlignmentCheck(False, f"non_finite_{name}")
            if float(v) < -eps_s:
                return AlignmentCheck(False, f"negative_{name}")

    # Per-word end >= start.
    for i in range(n):
        if float(ends[i]) < float(starts[i]) - eps_s:
            return AlignmentCheck(False, f"end_before_start@{i}")

    # Monotonic non-decreasing starts.
    for i in range(1, n):
        if float(starts[i]) < float(starts[i - 1]) - eps_s:
            return AlignmentCheck(False, f"non_monotonic_start@{i}")

    # RUNAWAY guard — intrinsic to the alignment (no audio ref needed). A model
    # repetition-loop collapses into one word with a multi-second duration; real
    # speech words are sub-2 s. Reject when the max per-word duration exceeds the
    # ceiling. See MAX_WORD_DUR_ENV_VAR / module notes.
    if max_word_dur_s is not None:
        for i in range(n):
            word_dur = float(ends[i]) - float(starts[i])
            if word_dur > max_word_dur_s:
                return AlignmentCheck(
                    False,
                    f"runaway_word_dur(dur_s={word_dur:.3f},max={max_word_dur_s:.3f},@{i})",
                )

    # Optional secondary runaway signal: implausibly low words-per-second over
    # the chunk span (uniform-smear runaways). OFF by default — thin clean
    # separation; opt-in via MIN_WPS_ENV_VAR. Only applied when there are enough
    # words for the rate to be meaningful and the span is positive.
    if min_wps is not None and n >= _MIN_WORDS_FOR_WPS:
        span_s = float(ends[-1]) - float(starts[0])
        if span_s > 0:
            wps = n / span_s
            if wps < min_wps:
                return AlignmentCheck(
                    False,
                    f"runaway_low_wps(wps={wps:.3f},min={min_wps:.3f},n={n},span_s={span_s:.3f})",
                )

    # Bounds + scale vs. emitted audio.
    if audio_duration_s is not None and audio_duration_s > 0:
        max_end = max(float(e) for e in ends)
        # No word may end meaningfully past the emitted audio. Allow the
        # larger of the absolute epsilon and a small relative margin so a
        # legitimate final-frame rounding doesn't trip.
        past_tol = max(eps_s, 0.15 * audio_duration_s)
        if max_end > audio_duration_s + past_tol:
            return AlignmentCheck(
                False,
                f"past_audio(max_end={max_end:.3f},audio={audio_duration_s:.3f})",
                max_end / audio_duration_s,
            )
        scale_ratio = float(ends[-1]) / audio_duration_s
        if not (scale_lo <= scale_ratio <= scale_hi):
            return AlignmentCheck(
                False,
                f"scale_ratio_out_of_range({scale_ratio:.3f})",
                scale_ratio,
            )
        # Optional ABSOLUTE drift cap (default OFF ⇒ no-op). Catches a long
        # chunk that passed the relative band above but drifted by whole
        # seconds. UNVERIFIED — see ABS_DRIFT_ENV_VAR / module notes.
        if abs_drift_max_s is not None:
            abs_drift_s = abs(1.0 - scale_ratio) * audio_duration_s
            if abs_drift_s > abs_drift_max_s:
                return AlignmentCheck(
                    False,
                    f"abs_drift(drift_s={abs_drift_s:.3f},max={abs_drift_max_s:.3f})",
                    scale_ratio,
                )
        return AlignmentCheck(True, "ok", scale_ratio)

    return AlignmentCheck(True, "ok_no_audio_ref")


def validate_partial_word_alignment(
    words: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    max_end_s: float | None,
    *,
    eps_s: float = DEFAULT_EPS_S,
    max_word_dur_s: float | None = DEFAULT_MAX_WORD_DUR_S,
) -> AlignmentCheck:
    """Guard for a MID-SYNTHESIS prefix of an alignment.

    The full-chunk guard cannot be reused. Its ``scale_ratio`` band
    ([0.85, 1.15]) asserts the words span essentially all of the emitted audio,
    which is true at the end of a chunk and false by construction here.

    More importantly, a partial must NOT be checked against the PCM emitted so
    far. The talker decodes codec frames ahead of what code2wav has converted
    and the serving layer has put on the wire, and the commit horizon rides the
    DECODED frames — so a committed word legitimately ends after the audio the
    client has received. That lead is the entire point of the feature: the word
    arrives before you hear it. An earlier version of this function bounded
    ``max_end`` by the emitted audio and rejected 926 of ~1000 valid partial
    frames on a concurrency sweep, silently collapsing the feature back toward
    per-chunk behaviour under load.

    ``max_end_s`` is therefore the TEXT-PROPORTIONAL ceiling
    (``serving_speech._runaway_audio_cap_seconds``) — the same calibration the
    audio runaway early-stop uses — not the emitted duration. It catches an
    alignment claiming times no amount of speech for this text could reach,
    while leaving the decode lead alone. Pass ``None`` to run structural checks
    only.

    A ``False`` result means: do not emit this partial frame. The end-of-chunk
    frame still carries every word, so a rejected partial costs cadence, never
    completeness.
    """
    check = validate_word_alignment(
        words,
        starts,
        ends,
        None,
        eps_s=eps_s,
        max_word_dur_s=max_word_dur_s,
        min_wps=None,
    )
    if not check.ok:
        return check
    if max_end_s is not None and max_end_s > 0 and ends:
        max_end = max(float(e) for e in ends)
        if max_end > max_end_s + eps_s:
            return AlignmentCheck(
                False,
                f"partial_beyond_text_cap(max_end={max_end:.3f},cap={max_end_s:.3f})",
                max_end / max_end_s,
            )
    return AlignmentCheck(True, "ok")
