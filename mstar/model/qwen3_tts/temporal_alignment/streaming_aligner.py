"""Incremental (streaming) monotonic Viterbi over the pointer-head trellis.

Why this exists
---------------
``registry.finalize`` runs the pointer head and :func:`viterbi.
word_timestamps_viterbi_full_coverage` ONCE, over the whole decode, after the
request completes. The serving layer therefore cannot emit a ``timestamps``
frame until the chunk's audio is fully synthesised — a 240-char chunk is ~15 s
of speech with no alignment available mid-way. A barge-in that lands inside
that window has no word boundary to splice on and the consumer falls back to a
words-per-second estimate.

This module produces the SAME alignment, incrementally, while decode is still
running. Two properties make that possible:

1. **The head is per-frame separable.** ``scores = W_q(frame_hidden) @
   W_k(word_keys).T``. ``W_k`` depends only on the prompt (fixed once prefill
   ends) and each frame's query depends only on that frame. So a new frame
   costs one small projection + one ``(1, d) @ (d, Nw)`` matmul — the total
   work over a request is identical to the single batch pass, just spread out.

2. **The DP is Markov in ``t``.** ``dp[t]`` depends only on ``dp[t-1]``, so the
   trellis extends row by row. Only the ``(T, Nw)`` int32 backpointer table is
   retained (48 KB at T=200, Nw=60).

Commit horizon (when is a partial word safe to publish?)
--------------------------------------------------------
A prefix of the path can be published only if a later frame will not change
it. That is decided by **backtrace merging**: at frame ``T``, walk the
backpointers backwards from every live state at once; once the walks collapse
onto a single state at frame ``t0``, every path still in play agrees about
frames ``0..t0``, so that prefix can be emitted.

Quantifying over ALL states makes this a theorem — the finally-decoded path,
free-endpoint or full-coverage, is one of the paths being quantified over — but
it is a useless one. A state many words AHEAD of where the audio actually is
is reachable only by racing forward from frame 0, so its backtrace disagrees
with the real one everywhere. Measured on a sharply peaked, near-noiseless
trellis: 150 frames in, the all-states merge had committed **1** frame, and
across 40 trials it committed **0 %** of words before the end of the request.

So the live set is pruned two ways (see :meth:`StreamingAligner._merge`):
feasibility (``j > t`` is unreachable — exact), and a **beam margin** in nats
(heuristic). The margin is the whole trade-off, measured over 40 trials per
cell, lag in frames behind the newest frame (80 ms each):

===========  ========  ========  =====================  ===============
trellis      margin    lag p50   lag p90                revised trials
===========  ========  ========  =====================  ===============
speech-like  inf       63 f      141 f                  0 / 40 (0 % emitted)
speech-like  10        2 f       5 f                    0 / 40
speech-like  5         1 f       3 f                    0 / 40
random       10        24 f      67 f                   11 / 40
random       20        45 f      95 f                   0 / 40
===========  ========  ========  =====================  ===============

At the default margin a speech-shaped trellis commits within ~160 ms of the
newest frame and 94 % of a request's words are published before the request
ends, with no committed word ever contradicting the batch decode. An
unstructured trellis (the "random" row — no alignment signal at all, which is
what a wrongly loaded head looks like) degrades into revisions instead of stalling,
which is why ``registry.finalize`` re-checks the committed prefix against the
final decode and logs + counts any disagreement rather than trusting it
silently. ``commit_margin=inf`` recovers the exact-but-inert merge.

Caveat, deliberately narrow: :func:`viterbi._to_word_prob_matrix` decides
whether to softmax from ``scores.min() < 0`` over the WHOLE matrix, which a
prefix cannot know. This module always softmaxes row-wise. For a trained
``AlignmentPointerHead`` the scores are scaled dot products and always contain
negatives, so the two agree; the equality tests assert exactly that
precondition rather than pretending it is unconditional.
"""

from __future__ import annotations

import numpy as np
import torch

from mstar.model.qwen3_tts.temporal_alignment.frame_rate import (
    CODEC_FRAME_RATE_HZ,
    frame_to_sec,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
)

_INF = 1e20

# Beam width (nats) for the commit horizon. See ``StreamingAligner._merge``.
# 10 nats is ~4 orders of magnitude of posterior; a trailing path that far
# behind winning back the lead has not been observed on real or synthetic
# trellises. Override per-request or via VLLM_TTS_WORD_TS_COMMIT_MARGIN.
DEFAULT_COMMIT_MARGIN: float = 10.0


def row_softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax over the word axis, numerically stabilised. Row-wise."""
    if scores.size == 0:
        return scores
    m = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(m)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-10, None)


def extend_trellis(
    dp_prev: np.ndarray,
    cost_row: np.ndarray,
    advance_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One vectorised step of the stay/advance DP in :mod:`viterbi`.

    Mirrors the reference inner loop exactly, ties included (``stay <=
    advance`` picks stay), so the streaming trellis is identical to the
    batch one cell for cell.
    """
    nw = dp_prev.shape[0]
    advance = np.empty_like(dp_prev)
    advance[0] = _INF
    if nw > 1:
        advance[1:] = dp_prev[:-1] + advance_penalty
    take_stay = dp_prev <= advance
    dp_new = np.where(take_stay, dp_prev, advance) + cost_row
    idx = np.arange(nw, dtype=np.int32)
    backptr = np.where(take_stay, idx, idx - 1).astype(np.int32)
    return dp_new, backptr


class StreamingAligner:
    """Frame-by-frame pointer-head readout with an exact commit horizon.

    Construct once per request after prefill (the word keys are fixed then),
    then ``push`` decode-frame hiddens as they are produced. ``committed_words``
    returns every word whose timing can no longer change.
    """

    def __init__(
        self,
        head: torch.nn.Module,
        word_key_hidden: torch.Tensor,
        spans: list[WordSpan],
        *,
        rate: float = CODEC_FRAME_RATE_HZ,
        advance_penalty: float = 0.5,
        commit_margin: float = DEFAULT_COMMIT_MARGIN,
    ) -> None:
        self.spans = spans
        self.rate = float(rate)
        self.advance_penalty = float(advance_penalty)
        self.commit_margin = float(commit_margin)
        self.n_words = len(spans)

        self._device = next(head.parameters()).device
        self._dtype = next(head.parameters()).dtype
        self._head = head
        # W_k over the word keys is prompt-fixed: project once, reuse for every
        # frame. This is what makes a pushed frame O(Nw) instead of O(Nw * H).
        with torch.no_grad():
            keys = word_key_hidden.to(device=self._device, dtype=self._dtype)
            self._k_proj = head.W_k(keys)  # (Nw, d)
        self._scale = float(getattr(head, "_scale", 1.0))

        self._n_frames = 0
        self._dp_prev: np.ndarray | None = None
        self._backptr: list[np.ndarray] = []
        # Committed path prefix — grows only, never rewritten.
        self._committed_path: list[int] = []

    # -- ingest ------------------------------------------------------------

    def push(self, frame_hidden: torch.Tensor) -> None:
        """Extend the trellis by one or more decode frames.

        ``frame_hidden`` is ``(H,)`` or ``(n_new, H)`` in the head's space.
        Raises nothing the caller must handle: an empty tensor is a no-op.
        """
        if self.n_words == 0:
            return
        x = frame_hidden
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] == 0:
            return
        with torch.no_grad():
            q = self._head.W_q(x.to(device=self._device, dtype=self._dtype))
            # NOTE: a windowed head (``window > 0``) contextualises each query
            # with its neighbours via a depthwise conv, which is NOT causal —
            # a frame's query would change once its right neighbours arrive.
            # ``StreamingAligner.supports`` rejects those heads up front, so
            # reaching here means the head is per-frame separable.
            scores = (q @ self._k_proj.transpose(0, 1)) * self._scale
        arr = scores.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float64)
        probs = row_softmax(arr)
        cost = -np.log(np.clip(probs, 1e-10, 1.0))

        for t in range(cost.shape[0]):
            if self._dp_prev is None:
                dp = np.full((self.n_words,), _INF, dtype=np.float64)
                dp[0] = cost[t, 0]
                self._dp_prev = dp
                # Frame 0 is trivially determined: dp[0, j>0] is infeasible.
                self._backptr.append(np.zeros((self.n_words,), dtype=np.int32))
                self._committed_path = [0]
            else:
                self._dp_prev, bp = extend_trellis(self._dp_prev, cost[t], self.advance_penalty)
                self._backptr.append(bp)
            self._n_frames += 1

    @staticmethod
    def supports(head: torch.nn.Module) -> bool:
        """True when the head can be read out frame-causally.

        ``window > 0`` heads mix neighbouring frames' queries through a
        non-causal depthwise conv, so an early frame's score is not final until
        its right context exists — incremental commit would be unsound.
        """
        return int(getattr(head, "window", 0) or 0) == 0

    # -- commit ------------------------------------------------------------

    def _merge(self) -> tuple[int, int]:
        """``(t0, state)`` — the last frame whose state is settled, and it.

        Walk the backpointers backwards from every *live* state at the newest
        frame at once; the frame where those walks collapse to a single state
        is the commit horizon.

        Two prunings decide which states count as live, and they are the
        difference between a horizon that moves and one that never leaves
        frame 0:

        * **Feasibility (exact).** A monotonic path starts on word 0 at frame
          0 and advances at most one word per frame, so state ``j > t`` is
          unreachable at frame ``t``. The vectorised DP step still writes
          backpointers for those columns and their walks are meaningless.

        * **Beam margin (heuristic, ``commit_margin``).** States whose partial
          cost is far above the best are dropped. Without this the horizon is
          pinned: a state many words AHEAD of where the audio actually is can
          only be reached by racing forward from frame 0, so its backtrace
          disagrees with the real one everywhere, and an all-states merge
          therefore almost never fires even on a sharply peaked, essentially
          noiseless trellis (measured: 1 frame committed out of 150).

        The margin is what makes commits *practically* rather than provably
        irrevocable: a committed word is revised only if a path currently more
        than ``commit_margin`` nats behind goes on to win. ``revision_rate``
        in the streaming-alignment tests measures how often that happens
        against the batch decode; at the default it is zero on both random and
        speech-shaped trellises. Set ``commit_margin`` to ``inf`` to recover
        the exact-but-inert all-states merge.
        """
        t_last = self._n_frames - 1
        if t_last <= 0:
            return 0, 0  # frame 0 is always word 0
        n_feasible = min(self.n_words, t_last + 1)
        live = np.arange(n_feasible, dtype=np.int32)
        if self._dp_prev is not None and np.isfinite(self.commit_margin):
            dp = self._dp_prev[:n_feasible]
            live = live[dp <= dp.min() + self.commit_margin]
            if live.size == 0:
                return len(self._committed_path) - 1, 0
        cur = live
        for t in range(t_last, 0, -1):
            cur = self._backptr[t][cur]
            if cur.max() == cur.min():
                return t - 1, int(cur[0])
        return 0, 0

    def _extend_committed_path(self) -> None:
        t0, state = self._merge()
        last_committed = len(self._committed_path) - 1
        if t0 <= last_committed:
            return
        tail: list[int] = []
        t, cur = t0, state
        while t > last_committed:
            tail.append(cur)
            cur = int(self._backptr[t][cur])
            t -= 1
        self._committed_path.extend(reversed(tail))

    @property
    def n_committed_frames(self) -> int:
        return len(self._committed_path)

    def committed_words(self) -> list[dict]:
        """Words whose start AND end are final, in voicing order.

        A word is final once the committed path has moved PAST it — while the
        path is still sitting on the last committed word that word's end time
        can still grow. Output shape matches
        :func:`viterbi.word_timestamps_viterbi_full_coverage`: words the path
        never visits are omitted, times are rounded to milliseconds.
        """
        if self.n_words == 0 or self._n_frames == 0:
            return []
        self._extend_committed_path()
        path = self._committed_path
        if len(path) < 2:
            return []
        # The path is non-decreasing, so its last entry IS the frontier word —
        # the one still being voiced. Everything strictly before it is final.
        frontier = path[-1]
        if frontier <= 0:
            return []
        out: list[dict] = []
        run_word = path[0]
        run_start = 0
        for f in range(1, len(path)):
            if path[f] == run_word:
                continue
            if run_word < frontier:
                out.append(self._word_dict(run_word, run_start, f - 1))
            run_word = path[f]
            run_start = f
        return out

    def _word_dict(self, w_idx: int, first_frame: int, last_frame: int) -> dict:
        return {
            "word": self.spans[w_idx].word,
            "start": round(frame_to_sec(first_frame, self.rate), 3),
            "end": round(frame_to_sec(last_frame + 1, self.rate), 3),
        }
