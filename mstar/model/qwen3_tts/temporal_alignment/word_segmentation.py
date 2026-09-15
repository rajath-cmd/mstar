"""Group BPE text-token ids into word spans for the alignment pointer.

Ġ/▁ leading-space markers delimit words; bracket/angle tags are emitted
as their own ``is_tag=True`` spans so callers can skip them. Word strings
are assembled via the tokenizer's ``convert_tokens_to_string`` (byte
decoder) so accented chars decode correctly (``ständig`` not ``stÃ¤ndig``);
falls back to a naive join for minimal tokenizers in unit tests.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class WordSpan(NamedTuple):
    word: str
    token_start: int
    token_end: int
    is_tag: bool


_WS = ("Ċ", "\n", "\\n")


def _is_tag_open(s: str) -> bool:
    return s.startswith("[") or s.startswith("<")


def _tag_close_in(s: str) -> bool:
    return "]" in s or ">" in s


def segment_words(token_ids: list[int], tokenizer) -> list[WordSpan]:
    toks = tokenizer.convert_ids_to_tokens(token_ids)
    n = len(toks)
    spans: list[WordSpan] = []
    i = 0
    cur: list[str] = []
    cur_start = -1

    def _decode(pieces: list[str]) -> str:
        if hasattr(tokenizer, "convert_tokens_to_string"):
            return tokenizer.convert_tokens_to_string(pieces).strip()
        return "".join(p.lstrip("Ġ▁") for p in pieces).strip()

    def flush() -> None:
        nonlocal cur, cur_start
        if cur and cur_start >= 0:
            w = _decode(cur)
            if w:
                spans.append(WordSpan(w, cur_start, cur_start + len(cur), False))
        cur = []
        cur_start = -1

    while i < n:
        raw = toks[i]
        stripped = raw.lstrip("Ġ▁")
        if _is_tag_open(stripped):
            flush()
            j = i
            buf: list[str] = []
            while j < n:
                buf.append(toks[j])
                if _tag_close_in(toks[j].lstrip("Ġ▁")):
                    break
                j += 1
            j = min(j, n - 1)
            text = _decode(buf)
            m = re.search(r"\[[^\]]+\]|<[^>]+>", text)
            spans.append(WordSpan(m.group(0) if m else text, i, j + 1, True))
            i = j + 1
            continue
        if raw in _WS:
            i += 1
            continue
        clean = raw.lstrip("Ġ▁")
        if not clean:
            # A bare marker-only token (e.g. a standalone "Ġ") still carries a
            # word-boundary signal even though it has no text of its own --
            # flush before discarding it, or the next token (which has no
            # marker of its own, since the tokenizer put it here instead)
            # silently glues onto the current word instead of starting a new
            # one. Seen empirically: digit runs like "1, 2, 3" tokenize with
            # isolated "Ġ" tokens rather than fused "Ġ2"-style tokens.
            flush()
            i += 1
            continue
        is_start = raw.startswith("Ġ") or raw.startswith("▁") or i == 0
        if is_start and cur:
            flush()
            cur = [raw]
            cur_start = i
        elif not cur:
            cur = [raw]
            cur_start = i
        else:
            cur.append(raw)
        i += 1
    flush()
    spans.sort(key=lambda s: s.token_start)
    return spans


def spoken_words(spans: list[WordSpan]) -> list[WordSpan]:
    return [s for s in spans if not s.is_tag]
