# Ported VERBATIM from vllm-omni (text_chunker.py) — do not reimplement.
#
# Chunk boundaries decide where prosody breaks and where the inter-chunk
# pause lands, so a reimplementation would diverge in exactly the way the
# parity contract forbids. Both modules are engine-agnostic (xml_segmenter
# imports only 're'; text_chunker imports only xml_segmenter), so they port
# unchanged apart from the import path below.
#
# Upstream: vllm-omni @ raj/ws-incremental-word-timestamps

"""Pluggable text chunking strategies for streaming TTS input.

Provides a Protocol-based interface for splitting incoming text into chunks
suitable for audio generation, with four built-in implementations:

- StreamingChunker: prosody-driven default. Same primary segmenter as
  SentenceChunker, plus (a) merging of short sentences into a chunk-size band
  matched to the training distribution, and (b) secondary clause-internal
  splitting when a single sentence exceeds ``max_chunk_chars``. Used when
  the upstream is streaming raw LLM tokens (no client-side segmentation).
- SentenceChunker: XML-tag- and abbreviation-aware sentence splitting only.
  Kept for backward compatibility — produces a chunk per sentence, which is
  too choppy when the upstream is already streaming whole sentences.
- TagAwareChunker: SentenceChunker plus prosody-driven merging. Same family
  as StreamingChunker but without the secondary-split safety net for an
  oversized single sentence. Kept for compatibility.
- NoSplitChunker: Passthrough — buffers everything until flush().

All four build on :func:`xml_segmenter.segment`, which keeps paralinguistic
tag spans atomic (a paired ``<whispers>...</whispers>`` span is never split,
nor is any tag's internals; the ``[[lang:xx]]`` code-switch marker stays
attached to its following text) and does not mistake an abbreviation period
("U.S.", "Dr.") for a sentence boundary.

The default sizing band (``min_chunk_chars=100``, ``max_chunk_chars=500``) was
chosen to land each emitted chunk inside the p25–p99 region of the Qwen3-TTS
training-text distribution (synth-voices-v2 corpus: p10=44, p50=99, p90=199,
p99=882, hard-validator cap 600). Chunks shorter than ~p25 produce out-of-
distribution choppy utterances; chunks longer than the hard cap push the
model into the rare passage tail (0.7% of training mass).

Factory function ``create_chunker(strategy, **kwargs)`` instantiates the
appropriate implementation.
"""

from typing import Protocol, runtime_checkable

from mstar.api_server.openai.xml_segmenter import (
    XML_TAG_RE,
    find_boundaries,
    find_secondary_boundaries,
    segment,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class TextChunker(Protocol):
    """Interface for incremental text chunkers used by the streaming TTS handler."""

    def add_text(self, text: str) -> list[str]:
        """Add text and return any complete chunks ready for generation."""
        ...

    def flush(self) -> str | None:
        """Return any remaining buffered text, or None if empty."""
        ...


# ---------------------------------------------------------------------------
# SentenceChunker — default strategy
# ---------------------------------------------------------------------------
class SentenceChunker:
    """XML-tag- and abbreviation-aware sentence splitter (default strategy).

    Splits at English and CJK sentence boundaries. Periods that close an
    abbreviation ("Dr.", "U.S.", "p.m.") do not trigger a split, and
    paralinguistic tags are never broken — a paired ``<whispers>...</whispers>``
    span (which may itself contain sentence-final punctuation) stays inside a
    single chunk.

    Short fragments below ``min_sentence_length`` are kept buffered and merged
    with the following sentence so a tiny isolated phrase is not synthesized
    on its own.
    """

    def __init__(self, min_sentence_length: int = 20) -> None:
        self._buffer: str = ""
        self._min_sentence_length = min_sentence_length

    def add_text(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        return self._extract_sentences()

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def _extract_sentences(self) -> list[str]:
        units, remainder = segment(self._buffer)
        if not units:
            return []

        sentences: list[str] = []
        carry = ""
        for unit in units:
            text = (carry + unit).strip()
            carry = ""
            if not text:
                continue
            if len(text) >= self._min_sentence_length:
                sentences.append(text)
            else:
                # Too short on its own (e.g. "Hi.") — merge with what follows.
                carry = text + " "

        # A short trailing fragment is pushed back onto the buffer so it can
        # merge with the next incoming text rather than being emitted alone.
        self._buffer = carry + remainder
        return sentences


# ---------------------------------------------------------------------------
# TagAwareChunker — prosody-driven merging on top of tag-aware splitting
# ---------------------------------------------------------------------------
class TagAwareChunker:
    """Sentence splitting with prosody-driven merging and a chunk-size band.

    Builds on the same tag-aware segmentation as :class:`SentenceChunker`
    (paired-tag spans atomic, abbreviation periods ignored) and then merges
    the resulting sentences into larger chunks:

    1. Chunks below ``min_chunk_chars`` are merged with the next sentence.
    2. A short trailing sentence (< ``short_sentence_chars``) is merged with
       the next sentence for smoother prosody.
    3. A self-closing tag near a sentence boundary keeps the two sentences
       together so the tag is not stranded.
    4. Merging stops before a chunk would exceed ``max_chunk_chars``.
    """

    def __init__(
        self,
        min_chunk_chars: int = 80,
        max_chunk_chars: int = 500,
        tag_lookahead_chars: int = 50,
        short_sentence_chars: int = 40,
    ) -> None:
        self._buffer: str = ""
        self._min_chunk_chars = min_chunk_chars
        self._max_chunk_chars = max_chunk_chars
        self._tag_lookahead_chars = tag_lookahead_chars
        self._short_sentence_chars = short_sentence_chars

    def add_text(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        return self._extract_chunks()

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def _extract_chunks(self) -> list[str]:
        """Split buffer into chunks respecting tag context and chunk size."""
        complete, remainder = segment(self._buffer)
        if not complete:
            # Nothing terminated yet — keep buffering.
            return []

        self._buffer = remainder
        return self._merge_sentences(complete)

    def _merge_sentences(self, sentences: list[str]) -> list[str]:
        """Merge a sentence list into chunks respecting tag/length constraints."""
        if not sentences:
            return []

        chunks: list[str] = []
        current = sentences[0]

        for next_sent in sentences[1:]:
            merged = current.rstrip() + " " + next_sent.lstrip()

            # Force split before exceeding max_chunk_chars, but only once the
            # current chunk has reached the minimum size.
            if len(merged) > self._max_chunk_chars and len(current.strip()) >= self._min_chunk_chars:
                stripped = current.strip()
                if stripped:
                    chunks.append(stripped)
                current = next_sent
                continue

            should_merge = (
                # Rule 1: current chunk still below minimum size.
                len(current.strip()) < self._min_chunk_chars
                # Rule 2: trailing sentence is short — group for prosody.
                or len(self._last_sentence_fragment(current)) < self._short_sentence_chars
                # Rule 3: a tag bridges the boundary — keep the sentences together.
                or self._has_tag_near_end(current, self._tag_lookahead_chars)
                or self._has_tag_near_start(next_sent, self._tag_lookahead_chars)
            )

            if should_merge:
                current = merged
            else:
                stripped = current.strip()
                if stripped:
                    chunks.append(stripped)
                current = next_sent

        # Push the trailing chunk back so it can still merge with incoming text.
        if current.strip():
            sep = " " if self._buffer and not self._buffer[:1].isspace() else ""
            self._buffer = current + sep + self._buffer
        return chunks

    @staticmethod
    def _last_sentence_fragment(text: str) -> str:
        """Return the last sentence-like fragment of a multi-sentence string."""
        cuts = find_boundaries(text)
        last = text[cuts[-1] :] if cuts else text
        return last.strip()

    @staticmethod
    def _has_tag_near_start(text: str, lookahead: int) -> bool:
        """Check if an XML tag appears within the first ``lookahead`` chars."""
        return bool(XML_TAG_RE.search(text[:lookahead]))

    @staticmethod
    def _has_tag_near_end(text: str, lookahead: int) -> bool:
        """Check if an XML tag appears within the last ``lookahead`` chars."""
        window = text[-lookahead:] if len(text) >= lookahead else text
        return bool(XML_TAG_RE.search(window))


# ---------------------------------------------------------------------------
# StreamingChunker — default for streaming-LLM input
# ---------------------------------------------------------------------------
class StreamingChunker:
    """Prosody-driven streaming chunker (default strategy).

    Same primary segmentation as :class:`SentenceChunker` (tag- and
    abbreviation-aware sentence boundaries). On top of that:

    1. **Merging band**: chunks below ``min_chunk_chars`` merge with the next
       sentence so a single short clause is not synthesized in isolation.
    2. **Tag bridging**: a paralinguistic tag adjacent to a sentence boundary
       keeps the two sentences together so the tag is not stranded.
    3. **Secondary splitting**: when a single sentence is longer than
       ``max_chunk_chars``, the chunker falls back to clause-internal break
       points (``;``, ``:``, ``—``, ``,``, coordinating conjunctions) and
       picks the candidate closest to a target near the middle of the band.
       This is the path the previous :class:`TagAwareChunker` lacked — it
       would emit a single oversized chunk whenever the LLM produced a long
       run-on sentence, which is well out of the model's training distribution.

    Boundaries inside a paired-tag span (``<whispers>...</whispers>``,
    ``<rapidly=N>...</rapidly>``, ``<slowly=N>...</slowly>``) and inside the
    ``[[lang:xx]]`` code-switch marker are never used, regardless of priority.
    """

    def __init__(
        self,
        min_chunk_chars: int = 100,
        max_chunk_chars: int = 500,
        tag_lookahead_chars: int = 50,
        short_sentence_chars: int = 40,
        secondary_split_enabled: bool = True,
    ) -> None:
        if min_chunk_chars < 1:
            raise ValueError(f"min_chunk_chars must be >= 1, got {min_chunk_chars}")
        if max_chunk_chars < min_chunk_chars:
            raise ValueError(f"max_chunk_chars ({max_chunk_chars}) must be >= min_chunk_chars ({min_chunk_chars})")
        self._buffer: str = ""
        self._min_chunk_chars = min_chunk_chars
        self._max_chunk_chars = max_chunk_chars
        self._tag_lookahead_chars = tag_lookahead_chars
        self._short_sentence_chars = short_sentence_chars
        self._secondary_split_enabled = secondary_split_enabled

    def add_text(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        return self._extract_chunks()

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        if not remaining:
            return None
        # If the flushed tail is itself oversized we still apply the secondary
        # split — but only return the first chunk and stash the rest back on
        # the buffer (which the caller will pull via a follow-up flush).
        # Simpler: in practice flush() is called at input.done, so there is no
        # "next add_text"; emit the tail as-is.  An oversized tail at flush
        # time is rare and the worker can still handle it.
        return remaining

    def flush_chunks(self) -> list[str]:
        """Drain the buffer into MULTIPLE sentence-sized chunks.

        Unlike :meth:`flush` (which returns the buffer as a single tail
        string), this splits the buffered text at sentence boundaries —
        the same logic ``add_text`` uses — and runs ``_merge_sentences``
        WITHOUT pushing the last chunk back to the buffer.

        Use this on ``input.done`` when the client needs per-sentence
        interleaved delivery (e.g. word-timestamp streaming). On a buffer
        of "A. B. C. D." this returns up to 4 chunks instead of "A. B. C.
        D." as one. Subject to the same ``min_chunk_chars`` merging band
        as the streaming path, so very short trailing sentences may still
        merge with their predecessor.

        Returns ``[]`` if the buffer is empty.
        """
        remaining = self._buffer
        if not remaining.strip():
            self._buffer = ""
            return []

        # The boundary regex needs a trailing whitespace to recognise the
        # final sentence terminator. On input.done there is no more text
        # coming, so we artificially append a newline to let segment()
        # see the last "."/"!"/"?" as a real boundary.
        sentinel = "\n"
        complete, remainder = segment(remaining + sentinel)
        # Strip the trailing sentinel back off the last unit (it was a
        # whitespace char, so .rstrip() on the unit removes it cleanly).
        if complete and complete[-1].endswith(sentinel):
            complete[-1] = complete[-1][: -len(sentinel)]

        chunks: list[str] = []
        if complete:
            # Re-use the merging band rules but DO NOT push the trailing
            # chunk back — flush_chunks must drain the buffer fully.
            self._buffer = ""  # clear the merge-back target
            merged = self._merge_sentences(complete)
            chunks.extend(merged)
            # _merge_sentences pushed the last current back into _buffer.
            # Recover it and run it through _emit_chunk — which loops the
            # secondary split until each piece fits under max_chunk_chars
            # (a single oversized run-on sentence may yield 3+ chunks).
            tail = self._buffer.strip()
            self._buffer = ""
            if tail:
                self._emit_chunk(chunks, tail)
        else:
            # No sentence boundaries even with sentinel — emit raw remainder
            # via _emit_chunk so the secondary-split loop applies.
            tail = remainder.strip() if remainder else remaining.strip()
            self._buffer = ""
            if tail:
                self._emit_chunk(chunks, tail)
        return chunks

    def _extract_chunks(self) -> list[str]:
        complete, remainder = segment(self._buffer)
        if not complete:
            # If the buffer itself is past max and contains no sentence
            # terminator, fall back to secondary split rather than letting
            # the buffer grow unboundedly — this is the run-on-sentence path.
            if self._secondary_split_enabled and len(self._buffer.strip()) > self._max_chunk_chars:
                pre, post = self._secondary_split(self._buffer)
                if pre:
                    self._buffer = post
                    return [pre.strip()]
            return []

        self._buffer = remainder
        return self._merge_sentences(complete)

    def _merge_sentences(self, sentences: list[str]) -> list[str]:
        if not sentences:
            return []

        chunks: list[str] = []
        current = sentences[0]

        for next_sent in sentences[1:]:
            merged = current.rstrip() + " " + next_sent.lstrip()

            if len(merged) > self._max_chunk_chars and len(current.strip()) >= self._min_chunk_chars:
                self._emit_chunk(chunks, current)
                current = next_sent
                continue

            should_merge = (
                len(current.strip()) < self._min_chunk_chars
                or len(self._last_sentence_fragment(current)) < self._short_sentence_chars
                or self._has_tag_near_end(current, self._tag_lookahead_chars)
                or self._has_tag_near_start(next_sent, self._tag_lookahead_chars)
            )

            if should_merge:
                current = merged
            else:
                self._emit_chunk(chunks, current)
                current = next_sent

        # Push the trailing chunk back so it can still merge with incoming text.
        if current.strip():
            sep = " " if self._buffer and not self._buffer[:1].isspace() else ""
            self._buffer = current + sep + self._buffer
        return chunks

    def _emit_chunk(self, chunks: list[str], chunk: str) -> None:
        """Append ``chunk`` to ``chunks``, secondary-splitting if oversized."""
        stripped = chunk.strip()
        if not stripped:
            return
        if not self._secondary_split_enabled or len(stripped) <= self._max_chunk_chars:
            chunks.append(stripped)
            return

        # Oversized single sentence: split repeatedly until each piece fits.
        remaining = stripped
        while len(remaining) > self._max_chunk_chars:
            pre, post = self._secondary_split(remaining)
            if not pre:
                # No usable secondary boundary — emit the whole oversized
                # chunk and let the model deal with it (better than splitting
                # mid-word or mid-tag).
                break
            chunks.append(pre.strip())
            remaining = post.lstrip()
        if remaining.strip():
            chunks.append(remaining.strip())

    def _secondary_split(self, text: str) -> tuple[str, str]:
        """Pick the best secondary boundary inside [min, max] of ``text``.

        Returns ``(pre, post)``.  When no usable candidate exists, returns
        ``("", text)`` and the caller should give up and emit the oversized
        chunk as-is.
        """
        min_off = self._min_chunk_chars
        max_off = min(self._max_chunk_chars, len(text))
        candidates = find_secondary_boundaries(text, min_off, max_off)
        if not candidates:
            return "", text

        # Target sits near the middle of the band — bias toward larger chunks
        # so we don't emit a stream of micro-utterances.
        target = (min_off + max_off) // 2
        # Pick by (distance-to-target, -weight) so on a tie the stronger
        # boundary wins.
        cut, _ = min(candidates, key=lambda cw: (abs(cw[0] - target), -cw[1]))
        return text[:cut], text[cut:]

    @staticmethod
    def _last_sentence_fragment(text: str) -> str:
        cuts = find_boundaries(text)
        last = text[cuts[-1] :] if cuts else text
        return last.strip()

    @staticmethod
    def _has_tag_near_start(text: str, lookahead: int) -> bool:
        return bool(XML_TAG_RE.search(text[:lookahead]))

    @staticmethod
    def _has_tag_near_end(text: str, lookahead: int) -> bool:
        window = text[-lookahead:] if len(text) >= lookahead else text
        return bool(XML_TAG_RE.search(window))


# ---------------------------------------------------------------------------
# NoSplitChunker — passthrough (client controls boundaries)
# ---------------------------------------------------------------------------
class NoSplitChunker:
    """Never splits — buffers all text until flush().

    Use when the client wants full control over chunk boundaries by sending
    each chunk as a separate ``input.text`` + ``input.done`` pair.
    """

    def __init__(self) -> None:
        self._buffer: str = ""

    def add_text(self, text: str) -> list[str]:
        if text:
            self._buffer += text
        return []

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_chunker(strategy: str = "streaming", **kwargs) -> TextChunker:
    """Create a TextChunker instance for the given strategy.

    Args:
        strategy: One of ``"streaming"`` (default), ``"sentence"``,
            ``"tag_aware"``, or ``"none"``.
        **kwargs: Strategy-specific parameters forwarded to the constructor.
            Unknown kwargs are dropped per-strategy so callers can pass a
            single config dict regardless of which strategy is selected.

    Returns:
        A TextChunker implementation.

    Raises:
        ValueError: If the strategy is not recognized or constructor args
            are themselves invalid (e.g. ``max < min``).
    """
    if strategy == "streaming":
        accepted = {
            "min_chunk_chars",
            "max_chunk_chars",
            "tag_lookahead_chars",
            "short_sentence_chars",
            "secondary_split_enabled",
        }
        return StreamingChunker(**{k: v for k, v in kwargs.items() if k in accepted})
    if strategy == "sentence":
        accepted = {"min_sentence_length"}
        return SentenceChunker(**{k: v for k, v in kwargs.items() if k in accepted})
    if strategy == "tag_aware":
        accepted = {
            "min_chunk_chars",
            "max_chunk_chars",
            "tag_lookahead_chars",
            "short_sentence_chars",
        }
        return TagAwareChunker(**{k: v for k, v in kwargs.items() if k in accepted})
    if strategy == "none":
        return NoSplitChunker()
    raise ValueError(
        f"Unknown chunking strategy: {strategy!r}. Must be one of: 'streaming', 'sentence', 'tag_aware', 'none'."
    )
