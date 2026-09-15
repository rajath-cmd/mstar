"""Monotonic Viterbi DP for the pointer-head alignment matrix.

Stay/advance DP with ``advance_penalty=0.5`` under ``-log(attn)`` cost
— globally optimal monotonic frame→word path. The argmax variant in
:mod:`readout` is the alternative, but the trained head was evaluated
against this Viterbi.
"""

from __future__ import annotations

import numpy as np
import torch

from mstar.model.qwen3_tts.temporal_alignment.frame_rate import (
    frame_to_sec,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
)


def monotonic_viterbi(attn_weights: np.ndarray, advance_penalty: float = 0.5) -> np.ndarray:
    """Return ``(n_steps,)`` int path; non-decreasing, advances by ≤ 1/step."""
    n_steps, text_len = attn_weights.shape
    if n_steps == 0 or text_len == 0:
        return np.zeros(n_steps, dtype=np.int32)
    cost = -np.log(np.clip(attn_weights, 1e-10, 1.0))
    INF = 1e20
    dp = np.full((n_steps, text_len), INF, dtype=np.float64)
    backptr = np.zeros((n_steps, text_len), dtype=np.int32)
    dp[0, 0] = cost[0, 0]
    for t in range(1, n_steps):
        for j in range(text_len):
            stay = dp[t - 1, j]
            advance = dp[t - 1, j - 1] + advance_penalty if j > 0 else INF
            if stay <= advance:
                dp[t, j] = stay + cost[t, j]
                backptr[t, j] = j
            else:
                dp[t, j] = advance + cost[t, j]
                backptr[t, j] = j - 1
    path = np.zeros(n_steps, dtype=np.int32)
    path[n_steps - 1] = int(np.argmin(dp[n_steps - 1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def monotonic_viterbi_full_coverage(attn_weights: np.ndarray, advance_penalty: float = 0.5) -> np.ndarray | None:
    """Same DP as ``monotonic_viterbi``, but the path is constrained to END
    at the LAST column (``text_len - 1``) instead of picking whichever
    column is cheapest at the final frame.

    ``monotonic_viterbi``'s ``path[n_steps-1] = argmin(dp[n_steps-1])`` lets
    the optimal path stop early and never advance through the remaining
    words if per-frame scores never clearly justify paying
    ``advance_penalty`` to move on — confirmed via a live reproduction where
    the audio genuinely contained every script word but the free-endpoint
    decode only covered the first ~70% of them, despite having far more
    frames than words remaining. That's a decode-time optimization
    artifact, not a hard "ran out of audio" wall — forcing the endpoint
    removes the "free to stop early" option entirely; the DP still finds
    the globally cheapest path, just only among paths that visit every
    column.

    Returns ``None`` (not a path) when ``n_steps < text_len``: reaching the
    last column would need more transitions than there are frames, which is
    mathematically infeasible (each step advances at most one column), not
    a tuning knob. Callers must fall back to ``monotonic_viterbi`` in that
    case — a forced path there would fabricate timing for words that may
    never have actually been generated.
    """
    n_steps, text_len = attn_weights.shape
    if n_steps == 0 or text_len == 0:
        return np.zeros(n_steps, dtype=np.int32)
    if n_steps < text_len:
        return None
    cost = -np.log(np.clip(attn_weights, 1e-10, 1.0))
    INF = 1e20
    dp = np.full((n_steps, text_len), INF, dtype=np.float64)
    backptr = np.zeros((n_steps, text_len), dtype=np.int32)
    dp[0, 0] = cost[0, 0]
    for t in range(1, n_steps):
        for j in range(text_len):
            stay = dp[t - 1, j]
            advance = dp[t - 1, j - 1] + advance_penalty if j > 0 else INF
            if stay <= advance:
                dp[t, j] = stay + cost[t, j]
                backptr[t, j] = j
            else:
                dp[t, j] = advance + cost[t, j]
                backptr[t, j] = j - 1
    path = np.zeros(n_steps, dtype=np.int32)
    path[n_steps - 1] = text_len - 1  # the only real change vs monotonic_viterbi
    for t in range(n_steps - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def _to_word_prob_matrix(scores: np.ndarray) -> np.ndarray:
    """Softmax negative scores over the word axis so ``-log`` is well-defined."""
    if scores.size == 0:
        return scores
    if scores.min() < 0.0:
        m = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(m)
        return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-10, None)
    return scores


def word_timestamps_viterbi(
    scores,
    spans: list[WordSpan],
    rate: float = 12.5,
    advance_penalty: float = 0.5,
) -> list[dict]:
    """Viterbi-decode a ``(Nf, Nw)`` score matrix to per-word timing dicts.

    ``scores`` accepts torch tensors (any dtype) or numpy arrays. Returns
    ``[{word, start, end}, ...]`` sorted by ``start``; words with no
    assigned frame are omitted.
    """
    if hasattr(scores, "detach"):
        # numpy() rejects bf16/fp16 — cast to fp32 first.
        arr = scores.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        arr = np.asarray(scores)
    arr = arr.astype(np.float64)
    if arr.size == 0 or arr.shape[1] == 0 or not spans:
        return []
    probs = _to_word_prob_matrix(arr)
    path = monotonic_viterbi(probs, advance_penalty=advance_penalty)
    nf = path.shape[0]
    out: list[dict] = []
    for w_idx, span in enumerate(spans):
        frames = [f for f in range(nf) if int(path[f]) == w_idx]
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


def word_timestamps_viterbi_full_coverage(
    scores,
    spans: list[WordSpan],
    rate: float = 12.5,
    advance_penalty: float = 0.5,
) -> list[dict]:
    """Like ``word_timestamps_viterbi``, but forces the decode to cover
    every word span at least once when there are enough frames to do so.

    Falls back to the free-endpoint decode (``word_timestamps_viterbi``'s
    behavior) when ``n_frames < n_words``, since forcing an infeasible
    endpoint would fabricate timing for words that may never have been
    generated at all — see ``monotonic_viterbi_full_coverage``'s docstring.
    """
    if hasattr(scores, "detach"):
        # numpy() rejects bf16/fp16 — cast to fp32 first.
        arr = scores.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        arr = np.asarray(scores)
    arr = arr.astype(np.float64)
    if arr.size == 0 or arr.shape[1] == 0 or not spans:
        return []
    probs = _to_word_prob_matrix(arr)
    path = monotonic_viterbi_full_coverage(probs, advance_penalty=advance_penalty)
    if path is None:
        path = monotonic_viterbi(probs, advance_penalty=advance_penalty)
    nf = path.shape[0]
    out: list[dict] = []
    for w_idx, span in enumerate(spans):
        frames = [f for f in range(nf) if int(path[f]) == w_idx]
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
