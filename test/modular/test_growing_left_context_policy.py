"""GrowingLeftContextChunkPolicy: vllm-omni's streaming vocoder semantics.

The property that matters is COVERAGE: across a whole stream, every codec frame
must be emitted exactly once. A policy that drops frames still produces fluent
audio -- it just starts late or skips a word -- which is the worst failure mode
available here, so it is asserted directly rather than inferred from shapes.
"""

import pytest

from mstar.streaming.chunk_policy import (
    GrowingLeftContextChunkPolicy,
    LeftContextChunkPolicy,
)


def simulate(policy, n_frames: int):
    """Drive the StreamBuffer contract and return (fed, emitted) per pop.

    Mirrors ``StreamBuffer.pop_chunk``: take ``window_size()`` items from the
    front of the buffer, then drop ``next_chunk_size()`` of them.
    """
    buffer = list(range(n_frames))
    produced = 0
    emitted_total = 0
    pops = []
    while True:
        window = policy.window_size()
        if produced < n_frames:
            produced = min(n_frames, max(produced, window))
        available = produced
        if available < window:
            break
        fed = buffer[:window]
        overlap = min(policy._left_context, emitted_total)
        emitted = fed[overlap:]
        pops.append((fed, emitted))
        emitted_total += len(emitted)
        stride = policy.next_chunk_size(available)
        buffer = buffer[stride:]
        produced -= stride
        policy.register_chunk(stride)
        if emitted_total >= n_frames:
            break
    return pops, emitted_total


def test_every_frame_is_emitted_exactly_once():
    policy = GrowingLeftContextChunkPolicy(chunk=2, left_context=1, first_chunk=1)
    pops, _ = simulate(policy, n_frames=41)
    emitted = [f for _fed, emit in pops for f in emit]
    assert emitted == sorted(emitted), "frames emitted out of order"
    assert len(emitted) == len(set(emitted)), "a frame was emitted twice"
    assert emitted[0] == 0, "the stream lost its opening frame"
    assert emitted == list(range(len(emitted)))


def test_first_chunk_shortens_time_to_first_audio():
    """One frame to first audio, then full-size chunks."""
    policy = GrowingLeftContextChunkPolicy(chunk=4, left_context=2, first_chunk=1)
    pops, _ = simulate(policy, n_frames=40)
    assert len(pops[0][1]) == 1, "first emission should be first_chunk frames"
    assert len(pops[0][0]) == 1, "first window needs only first_chunk frames"
    # Steady state returns to the configured chunk.
    assert [len(e) for _f, e in pops[2:5]] == [4, 4, 4]


def test_context_grows_to_the_target_then_saturates():
    policy = GrowingLeftContextChunkPolicy(chunk=1, left_context=4)
    contexts = []
    for pop in range(8):
        contexts.append(policy.window_size() - policy._emit_size(pop))
        policy.register_chunk(policy.next_chunk_size(1000))
    assert contexts == [0, 1, 2, 3, 4, 4, 4, 4]


def test_chunk_one_is_expressible_unlike_the_non_growing_policy():
    """The whole point of the policy.

    chunk=1 forces left_context=0 under the non-growing policy, and that
    configuration produces no audio at all on a live server.
    """
    GrowingLeftContextChunkPolicy(chunk=1, left_context=15)  # must not raise
    with pytest.raises(ValueError, match="chunk > left_context"):
        LeftContextChunkPolicy(chunk=1, left_context=15)


def test_matches_the_non_growing_policy_in_steady_state():
    """Same window and stride once the context has filled."""
    grow = GrowingLeftContextChunkPolicy(chunk=4, left_context=2)
    for _ in range(6):
        grow.register_chunk(grow.next_chunk_size(1000))
    fixed = LeftContextChunkPolicy(chunk=4, left_context=2)
    fixed.register_chunk(fixed.next_chunk_size(1000))
    assert grow.window_size() == fixed.window_size() == 6
    assert grow.next_chunk_size(1000) == fixed.next_chunk_size(1000) == 4


def test_rejects_a_first_chunk_larger_than_the_steady_chunk():
    with pytest.raises(ValueError, match="first_chunk must be in"):
        GrowingLeftContextChunkPolicy(chunk=2, left_context=1, first_chunk=4)
