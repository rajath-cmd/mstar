from abc import ABC, abstractmethod


class ChunkPolicy(ABC):
    """Determines when a StreamBuffer has enough items for the consumer node."""
    def __init__(self):
        self.first_chunk_read = False
        self.items_consumed = 0

    def register_chunk(self, chunk_size: int):
        self.first_chunk_read = True
        self.items_consumed += chunk_size

    @abstractmethod
    def is_ready(self, buffer_len: int) -> bool:
        """Return True if the buffer has enough items for a chunk."""
        ...

    @abstractmethod
    def next_chunk_size(self, buffer_len: int) -> int:
        """Return the number of items to consume for the next chunk.

        Only called when is_ready() returns True.
        For sliding-window policies this is the stride, not the window.
        """
        ...

    @abstractmethod
    def window_size(self) -> int:
        """Return the full window of items to include in the chunk.

        For non-overlapping policies, equals next_chunk_size.
        For sliding-window policies, this is larger than the stride —
        the buffer retains older items so the chunk contains the full window.
        """
        ...

    def continue_after_producer_done(self) -> bool:
        """Whether the buffer should keep producing (empty) chunks after the
        producer signals done and all buffered items have been consumed.

        Default ``False``: partition-done is propagated to the conductor after
        the last item is flushed.

        Set to ``True`` for connections where the consumer must keep running
        after the producer finishes (e.g., Thinker→Talker: the Talker
        continues generating codec tokens after the Thinker hits text EOS).
        In this case the buffer produces empty chunks (``_collate([])`` →
        ``{"data": None}``), and the consumer's partition-done is determined
        by its own model logic, not by the StreamBuffer.
        """
        return False


class SlidingWindowChunkPolicy(ChunkPolicy):
    """Fixed-size sliding window that advances by a stride.

    Each pop_chunk returns `window` items and advances the consumed
    pointer by `stride`. Old items before the window are discarded.

    Example (Orpheus SNAC): window=28 tokens (4 frames), stride=7 (1 frame).
    """

    def __init__(self, window: int, stride: int):
        super().__init__()
        self._window = window
        self._stride = stride

    def is_ready(self, buffer_len: int) -> bool:
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        return self._stride

    def window_size(self) -> int:
        return self._window


class GrowingLeftContextChunkPolicy(ChunkPolicy):
    """Streaming vocoder policy whose left context GROWS to its target.

    Matches vllm-omni's ``chunked_decode`` and VoxServe's detokenizer stepping:

        Iter 0: codes[0 : chunk]                     ctx = 0
        Iter 1: codes[0 : chunk + 1*chunk]           ctx = min(L, 1*chunk)
        Iter k: codes[k*chunk - ctx : k*chunk + chunk]
                                                     ctx = min(L, k*chunk)

    The stride is ALWAYS ``chunk``; the context simply grows from nothing up to
    ``left_context`` as history accumulates. ``LeftContextChunkPolicy`` instead
    takes the full context from the very first pop, which forces it to advance
    by ``chunk - left_context`` and therefore to require ``chunk > left_context``.

    That constraint is the reason M* could not run ``chunk_frames=1``: with
    one-frame chunks the only legal context is 0, and
    ``chunk=1, left_context=0`` produces no audio at all. This policy removes
    it, so ``chunk=1, left_context=15`` -- what vllm-omni actually ships -- is
    expressible: audio leaves after ONE 80 ms frame instead of two, with MORE
    vocoder context at the boundaries rather than less.

    Because the buffer drops ``stride`` items from its front on every pop, a
    growing window is expressed as a stride that starts at zero and rises to
    ``chunk`` as the context fills:

        stride_k = chunk - (ctx_{k+1} - ctx_k)

    which is 0 while the context is still growing and ``chunk`` once it
    saturates. Nothing is lost while the stride is 0 -- those frames are still
    needed as context for the next pop.

    ``first_chunk`` shortens the FIRST emission only. This is the single
    cheapest latency win available to a streaming TTS server, and it is what
    VoxServe calls ``first_chunk_frames`` and vllm-omni calls
    ``codec_chunk_frames_at_begin``: time-to-first-audio is set by how many
    frames must be decoded before ANY audio leaves, while throughput is set by
    the steady-state chunk. Decoupling the two costs nothing -- one smaller
    vocoder call, once per request -- and it is strictly better than lowering
    ``chunk`` globally, which pays the same overhead on every chunk forever.
    """

    def __init__(self, chunk: int, left_context: int, first_chunk: int | None = None):
        super().__init__()
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        if left_context < 0:
            raise ValueError(f"left_context must be >= 0, got {left_context}")
        if first_chunk is not None and not 1 <= first_chunk <= chunk:
            raise ValueError(
                f"first_chunk must be in [1, chunk]; got first_chunk={first_chunk}, "
                f"chunk={chunk}. A first chunk LARGER than the steady-state chunk "
                f"would raise time-to-first-audio, which is the opposite of why "
                f"this knob exists."
            )
        self._chunk = chunk
        self._left_context = left_context
        self._first_chunk = chunk if first_chunk is None else first_chunk
        self._pops = 0

    def _emit_size(self, pop_index: int) -> int:
        return self._first_chunk if pop_index == 0 else self._chunk

    def _emitted_before(self, pop_index: int) -> int:
        """Frames already turned into emitted audio before ``pop_index``."""
        if pop_index <= 0:
            return 0
        return self._first_chunk + (pop_index - 1) * self._chunk

    def _context_at(self, pop_index: int) -> int:
        """Context frames available before pop ``pop_index``."""
        return min(self._left_context, self._emitted_before(pop_index))

    def register_chunk(self, chunk_size: int):
        super().register_chunk(chunk_size)
        self._pops += 1

    def is_ready(self, buffer_len: int) -> bool:
        return buffer_len >= self.window_size()

    def next_chunk_size(self, buffer_len: int) -> int:
        del buffer_len
        grow = self._context_at(self._pops + 1) - self._context_at(self._pops)
        return self._emit_size(self._pops) - grow

    def window_size(self) -> int:
        return self._context_at(self._pops) + self._emit_size(self._pops)


class LeftContextChunkPolicy(ChunkPolicy):
    """Chunk policy for streaming vocoders with left-context overlap.

    Matches HuggingFace's ``Qwen3OmniMoeCode2Wav.chunked_decode`` pattern:

        Iter 0: codes[0 : chunk]                → emit all (no context)
        Iter 1: codes[chunk-ctx : 2*chunk]       → trim first ctx, emit rest
        Iter 2: codes[2*chunk-ctx : 3*chunk]     → trim first ctx, emit rest

    The first pop returns ``chunk`` items (no context).  Subsequent pops
    return ``chunk + left_context`` items, where the leading ``left_context``
    items OVERLAP with the tail of the previous chunk.  This overlap allows
    the causal ConvNet vocoder to "warm up" its internal state on frames
    it has already processed, ensuring a smooth transition at chunk
    boundaries.

    The key invariant: the first pop advances by ``chunk - left_context``
    (not ``chunk``), so the last ``left_context`` items of the first chunk
    remain in the buffer as overlap for the second pop.  All subsequent
    pops advance by ``chunk``.
    """

    def __init__(self, chunk: int, left_context: int):
        super().__init__()
        # The first-pop invariant above advances by ``chunk - left_context``,
        # which is only a forward advance when chunk > left_context. With
        # chunk <= left_context it goes backwards (chunk=1, ctx=25 advances by
        # -24) and the stream silently loses its opening: measured on Qwen3-TTS,
        # a 6.08 s utterance came back as 4.64 s with the first ~18 frames gone,
        # transcribing as "lazy dog while the morning light..." instead of "The
        # quick brown fox jumps over the lazy dog while...". Truncated audio that
        # still sounds fluent is the worst possible failure here, so reject the
        # configuration rather than produce it.
        if chunk <= left_context:
            raise ValueError(
                f"LeftContextChunkPolicy requires chunk > left_context, got "
                f"chunk={chunk}, left_context={left_context}. The first pop "
                f"advances by chunk - left_context ({chunk - left_context}), so "
                f"this config drops the start of every stream. For lower latency "
                f"reduce BOTH (e.g. chunk=4, left_context=2)."
            )
        self._chunk = chunk
        self._left_context = left_context
        self._window = chunk + left_context

    def is_ready(self, buffer_len: int) -> bool:
        if not self.first_chunk_read:
            return buffer_len >= self._chunk
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        # First pop: advance by (chunk - left_context) so the tail of the
        # first chunk stays in the buffer as overlap for the next pop.
        if not self.first_chunk_read:
            return self._chunk - self._left_context
        return self._chunk

    def window_size(self) -> int:
        if not self.first_chunk_read:
            return self._chunk
        return self._window


class FixedChunkPolicy(ChunkPolicy):
    """Release non-overlapping chunks of fixed size.

    Each pop_chunk returns exactly `chunk_size` items and advances by
    `chunk_size`. No overlap, no sliding window.

    Args:
        chunk_size: number of items per chunk.
        continue_after_done: if True, keep producing empty chunks after
            the producer finishes and all buffered items are consumed.
    """

    def __init__(self, chunk_size: int, continue_after_done: bool = False):
        super().__init__()
        self._chunk_size = chunk_size
        self._continue_after_done = continue_after_done

    def is_ready(self, buffer_len) -> bool:
        return buffer_len >= self._chunk_size

    def next_chunk_size(self, buffer_len: int) -> int:
        return self._chunk_size

    def window_size(self) -> int:
        return self._chunk_size

    def continue_after_producer_done(self) -> bool:
        return self._continue_after_done
