# Ported VERBATIM from vllm-omni (xml_segmenter.py) — do not reimplement.
#
# Chunk boundaries decide where prosody breaks and where the inter-chunk
# pause lands, so a reimplementation would diverge in exactly the way the
# parity contract forbids. Both modules are engine-agnostic (xml_segmenter
# imports only 're'; text_chunker imports only xml_segmenter), so they port
# unchanged apart from the import path below.
#
# Upstream: vllm-omni @ raj/ws-incremental-word-timestamps

"""XML-tag-aware sentence segmentation for streaming TTS input.

The Qwen3-TTS model accepts paralinguistic tags in canonical XML form. A
naive sentence splitter (``(?<=[.!?])\\s+``) breaks in two ways that this
module fixes:

1. **Abbreviations.** In ``"...the respected U.S. presidents..."`` the period
   after the "S" in "U.S." is not a sentence boundary, but a bare regex treats
   it as one and truncates the utterance after "U.S.".

2. **Tags.** A tag may sit at a clause/sentence boundary, and *paired* tags
   (``<whispers>...</whispers>``, ``<rapidly=n>...</rapidly>``,
   ``<slowly=n>...</slowly>``) wrap content that may itself contain
   sentence-final punctuation. Splitting inside a paired-tag span orphans the
   opening and closing halves into separate chunks, so each chunk is
   synthesized with a malformed tag.

This module exposes one segmentation primitive, :func:`segment`, that both
chunking strategies in :mod:`text_chunker` build on.

Tag inventory mirrors ``data/tts_synth/tts_data_gen/tags/registry.py``. The
numeric arguments are matched permissively: the segmenter only needs to
recognize a tag *span* so it is never split — value validation happens
elsewhere in the pipeline.
"""

import re

# ---------------------------------------------------------------------------
# Canonical XML paralinguistic tags
# ---------------------------------------------------------------------------
# Self-closing (ephemeral + pause) tags. The pause duration is matched
# permissively (``=...s`` optional) so a slightly off-grammar value still
# registers as an atomic span rather than being split mid-tag.
_SELF_CLOSING = r"(?:laughs|sighs|gasps|clears_throat)/|pause(?:=[0-9.]+s?)?/"

# Paired-tag open / close forms.
_PAIRED_OPEN_BODY = r"whispers|rapidly=[1-5]|slowly=[1-5]"
_PAIRED_CLOSE_BODY = r"/whispers|/rapidly|/slowly"

#: Code-switch language marker (``[[lang:en]]``, ``[[lang:ar]]``, ...). The
#: marker is preserved end-to-end through training and synthesis (see
#: ``data/tts_synth/tts_data_gen/quality/text_validators.py::strip_unknown_tags``).
#: It must travel attached to the following text — splitting between the
#: marker and its content gives the model a shape it never saw.
_CS_LANG_TAG = r"\[\[lang:[a-zA-Z]{2,3}\]\]"

#: Matches any recognized XML tag (self-closing or paired open/close) OR the
#: ``[[lang:xx]]`` code-switch marker. Used by ``_inside_tag`` to prevent
#: splits inside an atomic span and by the chunker's "tag near boundary"
#: heuristic to merge adjacent sentences so the tag is not stranded.
XML_TAG_RE = re.compile(
    r"<(?:" + _SELF_CLOSING + r"|" + _PAIRED_OPEN_BODY + r"|" + _PAIRED_CLOSE_BODY + r")>" + r"|" + _CS_LANG_TAG
)
_OPEN_TAG_RE = re.compile(r"<(?:" + _PAIRED_OPEN_BODY + r")>")
_CLOSE_TAG_RE = re.compile(r"<(?:" + _PAIRED_CLOSE_BODY + r")>")

# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------
# Lowercased tokens whose trailing period is NOT a sentence boundary. Dotted
# forms ("u.s", "e.g") are stored without the trailing dot. Single-word
# abbreviations that are also common emphatic sentences ("No.") are
# deliberately excluded — merging two sentences is harmless, but suppressing
# a real boundary after a one-word reply is not worth the rare "No. 5" case.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # Titles
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "gen",
        "sen",
        "rep",
        "gov",
        "col",
        "capt",
        "lt",
        "sgt",
        "cmdr",
        # Common Latin / measure abbreviations
        "vs",
        "etc",
        "approx",
        "dept",
        "est",
        "fig",
        "vol",
        "inc",
        "ltd",
        "corp",
        "co",
        # Dotted forms (stored without trailing dot)
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.k",
        "u.n",
        "e.u",
        "ph.d",
    }
)

# A sentence-final punctuation run, optionally followed by a closing quote or
# bracket, immediately before whitespace. The whitespace is a lookahead so the
# cut point falls just past the punctuation.
_ENGLISH_BOUNDARY_RE = re.compile(r"([.!?]+)([\"')\]]?)(?=\s)")

# CJK full-width terminators (parity with the previous splitter behavior).
_CJK_BOUNDARY_RE = re.compile(r"[。！？，；]")


def _tag_events(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Scan ``text`` for XML tags.

    Returns:
        ``(spans, events)`` where ``spans`` is the list of ``(start, end)``
        character ranges occupied by tags, and ``events`` is a sorted list of
        ``(start, delta)`` paired-tag depth changes (+1 open, -1 close).
    """
    spans: list[tuple[int, int]] = []
    events: list[tuple[int, int]] = []
    for m in XML_TAG_RE.finditer(text):
        spans.append((m.start(), m.end()))
        frag = m.group()
        if _OPEN_TAG_RE.fullmatch(frag):
            events.append((m.start(), 1))
        elif _CLOSE_TAG_RE.fullmatch(frag):
            events.append((m.start(), -1))
    events.sort()
    return spans, events


def _inside_tag(spans: list[tuple[int, int]], pos: int) -> bool:
    """Return True if character offset ``pos`` falls within a tag span."""
    for start, end in spans:
        if start <= pos < end:
            return True
        if start > pos:
            break
    return False


def _depth_before(events: list[tuple[int, int]], pos: int) -> int:
    """Return the paired-tag nesting depth at offset ``pos``.

    Depth > 0 means ``pos`` sits inside an open ``<whispers>``/``<rapidly>``/
    ``<slowly>`` span whose closing tag has not yet appeared.
    """
    depth = 0
    for start, delta in events:
        if start < pos:
            depth += delta
        else:
            break
    return depth


def _is_abbreviation(text: str, period_pos: int) -> bool:
    """Return True if the period at ``period_pos`` closes an abbreviation.

    Looks at the run of letters/dots ending just before ``period_pos`` and
    treats it as an abbreviation when it is a known abbreviation, a
    single-letter initial ("J."), or a dotted acronym ("U.S.", "A.B.C.").
    """
    start = period_pos
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    token = text[start:period_pos]
    if not token:
        return False

    low = token.lower().strip(".")
    if not low:
        return False
    if low in _ABBREVIATIONS:
        return True

    letters_only = low.replace(".", "")
    # Single-letter initial, e.g. "Barack H. Obama".
    if len(letters_only) == 1:
        return True
    # Dotted acronym where every dotted segment is a single letter.
    segments = [s for s in low.split(".") if s]
    if len(segments) > 1 and all(len(s) == 1 for s in segments):
        return True
    return False


def find_boundaries(text: str) -> list[int]:
    """Return the sorted cut offsets of real sentence boundaries in ``text``.

    A cut offset is the character index immediately past a sentence-final
    boundary. Boundaries are excluded when they fall inside a tag span, inside
    an open paired-tag span, or close an abbreviation.
    """
    spans, events = _tag_events(text)
    cuts: list[int] = []

    for m in _ENGLISH_BOUNDARY_RE.finditer(text):
        if _inside_tag(spans, m.start()):
            continue
        if _depth_before(events, m.start()) > 0:
            continue
        punct = m.group(1)
        if punct == ".":
            period_pos = m.start(1)  # the single '.' itself
            if _is_abbreviation(text, period_pos):
                continue
        cuts.append(m.end())

    for m in _CJK_BOUNDARY_RE.finditer(text):
        if _inside_tag(spans, m.start()):
            continue
        if _depth_before(events, m.start()) > 0:
            continue
        cuts.append(m.end())

    cuts.sort()
    return cuts


def segment(text: str) -> tuple[list[str], str]:
    """Split ``text`` into completed sentence units and a trailing remainder.

    Args:
        text: The full buffered text to segment.

    Returns:
        ``(units, remainder)``. ``units`` are the substrings terminated by a
        real sentence boundary (in order, with original spacing). ``remainder``
        is the text past the last boundary — possibly an incomplete sentence,
        an unclosed paired-tag span, or a partial tag — which the caller keeps
        buffered until more text (or a flush) arrives.
    """
    cuts = find_boundaries(text)
    if not cuts:
        return [], text

    units: list[str] = []
    prev = 0
    for cut in cuts:
        units.append(text[prev:cut])
        prev = cut
    return units, text[prev:]


# ---------------------------------------------------------------------------
# Secondary (sub-sentence) boundary discovery
# ---------------------------------------------------------------------------
# Used when a single sentence exceeds ``max_chunk_chars`` and there is no
# upcoming sentence terminator we can rely on — we fall back to the strongest
# clause-internal break we can find.
#
# Priority weights: higher = stronger natural break. The chunker picks the
# split candidate closest to ``target`` that respects ``min_offset`` /
# ``max_offset`` AND is not inside a tag span, then breaks ties by weight.
#
# We deliberately keep this list small and obvious — every entry here must
# correspond to a place a trained narrator would naturally take a micro-pause.
# Splitting at a weaker boundary (e.g. random whitespace) produces an
# out-of-distribution chunk shape.
_SECONDARY_BREAKS: tuple[tuple[re.Pattern, int], ...] = (
    # Hard clause breaks
    (re.compile(r";\s"), 90),
    (re.compile(r":\s"), 80),
    # Soft clause breaks
    (re.compile(r"—\s|\s—\s|\s--\s"), 70),
    (re.compile(r",\s"), 60),
    # Coordinating conjunctions — last resort. ``(?<=\S)`` requires a real
    # token before the conjunction, so we never split at "And ..." sentence
    # openings.
    (re.compile(r"(?<=\S)\s(?:and|but|so|or|because)\s"), 40),
)


def find_secondary_boundaries(
    text: str,
    min_offset: int,
    max_offset: int,
) -> list[tuple[int, int]]:
    """Return ``(cut_offset, weight)`` candidates inside ``[min_offset, max_offset]``.

    Cuts inside a recognized tag span are excluded.  Each match's cut offset is
    the position **after** the matched separator (so the punctuation stays with
    the left-hand chunk and the next chunk starts with the post-separator
    token).  The caller is responsible for picking the best candidate (e.g.
    closest to a target, highest weight on ties).

    Returns an empty list when no candidate exists in the window — the caller
    must then either widen the window, fall back to whitespace, or accept the
    oversized chunk.
    """
    if max_offset <= min_offset or max_offset > len(text):
        return []

    spans, events = _tag_events(text)
    out: list[tuple[int, int]] = []
    for pattern, weight in _SECONDARY_BREAKS:
        for m in pattern.finditer(text, min_offset, max_offset):
            cut = m.end()
            if _inside_tag(spans, cut - 1):
                continue
            # A cut BETWEEN a paired open/close tag (`<rapidly=2>fast, and
            # faster</rapidly>`) splits the span and orphans the halves into
            # separately-synthesized chunks with malformed tags — the paired
            # INTERIOR is atomic, not just the tag tokens. `_inside_tag` only
            # covers the tokens; `_depth_before` (the same guard the primary
            # path uses) covers the span.
            if _depth_before(events, cut) > 0:
                continue
            out.append((cut, weight))
    out.sort()
    return out
