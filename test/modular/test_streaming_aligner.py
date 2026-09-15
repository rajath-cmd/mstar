"""Unit tests for the incremental commit horizon (``streaming_aligner``).

The property that matters downstream is not "the streaming decode is close to
the batch decode" but "a word that has been PUT ON THE WIRE is never
contradicted later" — the WebSocket protocol is append-only, so a revision is
unrecoverable. These tests pin that as an equality against
``word_timestamps_viterbi_full_coverage``, which is what ``registry.finalize``
actually runs, on both speech-shaped and adversarial trellises.
"""

import numpy as np
import pytest
import torch

from mstar.model.qwen3_tts.temporal_alignment.pointer_head import (
    AlignmentPointerHead,
)
from mstar.model.qwen3_tts.temporal_alignment.streaming_aligner import (
    DEFAULT_COMMIT_MARGIN,
    StreamingAligner,
    extend_trellis,
    row_softmax,
)
from mstar.model.qwen3_tts.temporal_alignment.viterbi import (
    monotonic_viterbi,
    word_timestamps_viterbi_full_coverage,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
)


def _spans(n: int) -> list[WordSpan]:
    return [WordSpan(word=f"w{i}", token_start=i, token_end=i + 1, is_tag=False) for i in range(n)]


class _ScoreHead(torch.nn.Module):
    """Replays a fixed ``(T, Nw)`` score matrix through the head interface.

    ``StreamingAligner`` only ever calls ``W_q`` (per pushed frame) and ``W_k``
    (once), so a head whose ``W_k`` is the identity and whose ``W_q`` returns
    the next rows of a canned matrix lets a test control the trellis exactly
    while still exercising the real projection-caching code path.
    """

    def __init__(self, scores: np.ndarray) -> None:
        super().__init__()
        self.scores = np.asarray(scores, dtype=np.float64)
        self._scale = 1.0
        self.window = 0
        self.cursor = 0
        self.register_parameter("_anchor", torch.nn.Parameter(torch.zeros(1)))

    def W_k(self, keys: torch.Tensor) -> torch.Tensor:  # noqa: N802 — mirrors the real head
        return torch.eye(self.scores.shape[1], dtype=torch.float32)

    def W_q(self, x: torch.Tensor) -> torch.Tensor:  # noqa: N802
        n = x.shape[0]
        out = torch.tensor(self.scores[self.cursor : self.cursor + n], dtype=torch.float32)
        self.cursor += n
        return out

    def forward(self, frames: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.scores, dtype=torch.float32)


def _speech_like(n_frames: int, n_words: int, seed: int, sharp: float = 3.0, noise: float = 0.6) -> np.ndarray:
    """A trellis shaped like a real aligner's: peaked on a moving diagonal."""
    rng = np.random.default_rng(seed)
    per_word = n_frames / n_words
    s = np.array([[-sharp * abs(j - t / per_word) for j in range(n_words)] for t in range(n_frames)])
    return s + rng.normal(0, noise, s.shape)


def _drive(scores: np.ndarray, *, margin: float = DEFAULT_COMMIT_MARGIN, batch: int = 1):
    """Push a whole trellis through an aligner; return (aligner, snapshots)."""
    T, W = scores.shape
    spans = _spans(W)
    aligner = StreamingAligner(_ScoreHead(scores), torch.zeros(W, W), spans, commit_margin=margin)
    snapshots = []
    for t in range(0, T, batch):
        aligner.push(torch.zeros(min(batch, T - t), W))
        snapshots.append(aligner.committed_words())
    return aligner, snapshots


def _batch_words(scores: np.ndarray) -> list[dict]:
    W = scores.shape[1]
    return word_timestamps_viterbi_full_coverage(torch.tensor(scores, dtype=torch.float32), _spans(W))


class TestTrellisStepMatchesReference:
    """``extend_trellis`` is the vectorised twin of ``monotonic_viterbi``'s loop."""

    @pytest.mark.parametrize("seed", range(5))
    def test_cell_for_cell_against_the_reference_loop(self, seed: int):
        rng = np.random.default_rng(seed)
        T, W, penalty = 40, 9, 0.5
        probs = row_softmax(rng.normal(0, 1, (T, W)))
        cost = -np.log(np.clip(probs, 1e-10, 1.0))

        # Reference: the scalar loop from viterbi.monotonic_viterbi.
        INF = 1e20
        dp_ref = np.full((T, W), INF)
        bp_ref = np.zeros((T, W), dtype=np.int32)
        dp_ref[0, 0] = cost[0, 0]
        for t in range(1, T):
            for j in range(W):
                stay = dp_ref[t - 1, j]
                advance = dp_ref[t - 1, j - 1] + penalty if j > 0 else INF
                if stay <= advance:
                    dp_ref[t, j] = stay + cost[t, j]
                    bp_ref[t, j] = j
                else:
                    dp_ref[t, j] = advance + cost[t, j]
                    bp_ref[t, j] = j - 1

        dp = np.full((W,), INF)
        dp[0] = cost[0, 0]
        for t in range(1, T):
            dp, bp = extend_trellis(dp, cost[t], penalty)
            np.testing.assert_allclose(dp, dp_ref[t], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(bp, bp_ref[t])

    def test_ties_resolve_to_stay(self):
        """``stay <= advance`` — the reference keeps the current word on a tie."""
        dp_prev = np.array([1.0, 1.0, 1.0])
        # advance[j] = dp_prev[j-1] + 0.0 == dp_prev[j], an exact tie.
        _, bp = extend_trellis(dp_prev, np.zeros(3), advance_penalty=0.0)
        np.testing.assert_array_equal(bp, [0, 1, 2])


class TestCommittedWordsAreNeverRevised:
    @pytest.mark.parametrize("seed", range(12))
    def test_speech_like_committed_prefix_equals_batch_decode(self, seed: int):
        rng = np.random.default_rng(seed)
        T = int(rng.integers(60, 200))
        W = int(rng.integers(4, min(T // 4, 30)))
        scores = _speech_like(T, W, seed)
        aligner, _ = _drive(scores)
        committed = aligner.committed_words()
        ref = _batch_words(scores)
        assert committed, "a speech-shaped trellis must commit something before the end"
        assert ref[: len(committed)] == committed

    @pytest.mark.parametrize("seed", range(12))
    def test_snapshots_only_ever_grow(self, seed: int):
        """Each poll's list starts with the previous poll's, exactly.

        This is what lets the WebSocket layer send deltas and the client append
        without any reconciliation.
        """
        scores = _speech_like(150, 20, seed)
        _, snapshots = _drive(scores)
        for earlier, later in zip(snapshots, snapshots[1:]):
            assert later[: len(earlier)] == earlier

    @pytest.mark.parametrize("seed", range(6))
    def test_random_trellis_is_safe_at_a_wide_margin(self, seed: int):
        """No alignment signal at all — what a wrongly loaded head looks like.

        The default margin is tuned for a peaked trellis and does revise here;
        a wide margin trades horizon lag for safety and stops revising. The
        production guard for this case is the finalize-time revision counter,
        not a margin that is safe against every possible input.
        """
        rng = np.random.default_rng(1000 + seed)
        scores = rng.normal(0, 1, (120, 12))
        aligner, _ = _drive(scores, margin=40.0)
        committed = aligner.committed_words()
        ref = _batch_words(scores)
        assert ref[: len(committed)] == committed


class TestCommitHorizonBehaviour:
    def test_infinite_margin_is_exact_but_commits_almost_nothing(self):
        """Pins the finding that motivated the beam: the all-states merge stalls.

        Quantifying over every state includes ones reachable only by racing
        forward from frame 0, whose backtrace never rejoins the real path, so
        the horizon does not move. Exact and useless — hence ``commit_margin``.
        """
        scores = _speech_like(150, 20, seed=7)
        exact, _ = _drive(scores, margin=float("inf"))
        beamed, _ = _drive(scores, margin=DEFAULT_COMMIT_MARGIN)
        assert exact.committed_words() == []
        assert len(beamed.committed_words()) >= 0.8 * len(_batch_words(scores))

    def test_horizon_trails_the_newest_frame_by_only_a_few_frames(self):
        """Lag is the barge-in latency floor; keep it inside a few frames."""
        scores = _speech_like(150, 20, seed=3)
        T = scores.shape[0]
        spans = _spans(20)
        aligner = StreamingAligner(_ScoreHead(scores), torch.zeros(20, 20), spans)
        lags = []
        for t in range(T):
            aligner.push(torch.zeros(1, 20))
            aligner.committed_words()
            lags.append(t + 1 - aligner.n_committed_frames)
        # 80 ms per codec frame; p90 under 10 frames keeps the horizon under a
        # second behind the newest decoded frame.
        assert np.percentile(lags, 90) <= 10, f"p90 commit lag {np.percentile(lags, 90)} frames"

    def test_frontier_word_is_withheld(self):
        """The word currently being voiced can still grow — never publish it."""
        scores = _speech_like(120, 10, seed=11)
        aligner, _ = _drive(scores)
        committed = aligner.committed_words()
        ref = _batch_words(scores)
        assert len(committed) < len(ref)

    def test_batched_push_matches_frame_by_frame_push(self):
        """The runner batches frames between snapshots; that must not matter."""
        scores = _speech_like(96, 12, seed=5)
        one, _ = _drive(scores, batch=1)
        four, _ = _drive(scores, batch=4)
        assert one.committed_words() == four.committed_words()


class TestSupportsGate:
    def test_plain_head_is_supported(self):
        assert StreamingAligner.supports(AlignmentPointerHead(hidden_size=16, proj_size=8))

    def test_windowed_head_is_rejected(self):
        """A depthwise temporal conv is not causal — early queries are not final.

        The head shipped in this repo has no ``window`` yet (the training-side
        AlignmentPointerHead does, see RESULTS.md Phase 10). The gate is here so
        that if such a checkpoint is ever dropped in, incremental publication
        turns itself off instead of committing queries that later change.
        """
        head = AlignmentPointerHead(hidden_size=16, proj_size=8)
        assert StreamingAligner.supports(head)
        head.window = 2
        assert not StreamingAligner.supports(head)


class TestDegenerateInputs:
    def test_no_words_is_a_noop(self):
        aligner = StreamingAligner(_ScoreHead(np.zeros((4, 1))), torch.zeros(0, 1), [])
        aligner.push(torch.zeros(1, 1))
        assert aligner.committed_words() == []

    def test_no_frames_commits_nothing(self):
        aligner = StreamingAligner(_ScoreHead(np.zeros((0, 3))), torch.zeros(3, 3), _spans(3))
        assert aligner.committed_words() == []

    def test_single_frame_commits_nothing(self):
        """One frame pins word 0 as the frontier, and the frontier is withheld."""
        scores = _speech_like(1, 3, seed=0)
        aligner, _ = _drive(scores)
        assert aligner.committed_words() == []

    def test_single_word_never_commits(self):
        """With one word there is no frontier to move past."""
        scores = _speech_like(30, 1, seed=0)
        aligner, _ = _drive(scores)
        assert aligner.committed_words() == []


class TestRowSoftmaxAgreesWithTheBatchNormalisation:
    def test_matches_when_the_matrix_has_negatives(self):
        """The precondition the module docstring claims, asserted rather than assumed.

        ``viterbi._to_word_prob_matrix`` softmaxes only when the WHOLE matrix
        has a negative entry — a decision a prefix cannot make. Trained-head
        scores are scaled dot products and always do, so the two normalisations
        coincide in production; this is the check that they do.
        """
        rng = np.random.default_rng(0)
        scores = rng.normal(0, 1, (25, 6))
        assert scores.min() < 0
        streamed = row_softmax(scores)
        # The batch decode's path under its own normalisation must match ours.
        from mstar.model.qwen3_tts.temporal_alignment.viterbi import (
            _to_word_prob_matrix,
        )

        np.testing.assert_allclose(streamed, _to_word_prob_matrix(scores), rtol=1e-12)
        np.testing.assert_array_equal(
            monotonic_viterbi(streamed),
            monotonic_viterbi(_to_word_prob_matrix(scores)),
        )
