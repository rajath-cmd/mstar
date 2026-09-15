"""Per-request word-alignment capture + finalize.

Lifecycle: ``register`` (talker preprocess on first prefill chunk) →
``accumulate_prefill`` / ``accumulate_decode`` (runner post-forward) →
``finalize`` (runner on request completion) → ``pop_word_alignment``
(serving layer after generation completes). ``cancel`` cleans up on
client barge-in. Both tables are TTL-bounded: expired ``_inflight``
registrations and unread ``_finalized`` alignments are evicted on the
``register``/``finalize`` heartbeat so an aborted request cannot retain
memory for the life of the process.

vllm-omni runs the talker worker in a different OS process from the
APIServer, so the in-memory registry can't alone bridge them. ``finalize``
also writes a small JSON drop in ``/tmp/qwen3_tts_alignments/`` (override
via ``QWEN3_TTS_ALIGNMENT_IPC_DIR``); ``pop_word_alignment`` reads from
both surfaces.

Incremental publication: waiting for ``finalize`` means no alignment exists
until a chunk's audio is fully synthesised, so a barge-in landing mid-chunk
has no word boundary to splice on. ``accumulate_decode`` therefore also drives
a :class:`StreamingAligner` and, every ``VLLM_TTS_WORD_TS_PARTIAL_FRAMES``
frames, republishes the words its commit horizon has settled to a SECOND disk
drop (``<rid>.partial.json``, read by ``read_partial_alignment`` and never
popped). ``finalize`` clears that drop and re-checks the committed prefix
against the batch decode.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from mstar.api_server.openai.protocol import WordAlignment
from mstar.model.qwen3_tts.temporal_alignment.frame_rate import (
    CODEC_FRAME_RATE_HZ,
)
from mstar.model.qwen3_tts.temporal_alignment.pointer_head import (
    AlignmentPointerHead,
    pool_word_keys,
)
from mstar.model.qwen3_tts.temporal_alignment.streaming_aligner import (
    DEFAULT_COMMIT_MARGIN,
    StreamingAligner,
)
from mstar.model.qwen3_tts.temporal_alignment.viterbi import (
    word_timestamps_viterbi_full_coverage,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    segment_words,
)

# Soft-import metrics: the worker process may have a slimmer install where
# prometheus_client isn't available. Never let a missing metric break finalize.
try:
    from mstar.model.qwen3_tts.temporal_alignment._metrics import (
        TTS_ALIGNMENT_CANCEL_TOTAL,
        TTS_ALIGNMENT_COMMIT_REVISION_TOTAL,
        TTS_ALIGNMENT_EVICT_TOTAL,
        TTS_ALIGNMENT_FINALIZE_TOTAL,
        TTS_ALIGNMENT_FINALIZED,
        TTS_ALIGNMENT_INFLIGHT,
        TTS_ALIGNMENT_PARTIAL_TOTAL,
        TTS_ALIGNMENT_PARTIAL_WORDS,
        TTS_ALIGNMENT_POP_TOTAL,
        TTS_ALIGNMENT_POP_WAIT,
        TTS_ALIGNMENT_REGISTER_TOTAL,
    )

    _METRICS_ENABLED = True
except Exception:  # pragma: no cover — exercised on slim worker images
    _METRICS_ENABLED = False


def _inc(metric, **labels) -> None:
    if not _METRICS_ENABLED:
        return
    try:
        if labels:
            metric.labels(**labels).inc()
        else:
            metric.inc()
    except Exception:
        pass


def _observe(metric, value: float) -> None:
    if not _METRICS_ENABLED:
        return
    try:
        metric.observe(value)
    except Exception:
        pass


def _set_gauge(metric, value: float) -> None:
    if not _METRICS_ENABLED:
        return
    try:
        metric.set(value)
    except Exception:
        pass


logger = logging.getLogger(__name__)


_REGISTRY_TTL_SECONDS: float = 120.0

_DISK_IPC_DIR = Path(os.environ.get("QWEN3_TTS_ALIGNMENT_IPC_DIR", "/tmp/qwen3_tts_alignments"))
try:
    _DISK_IPC_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


def _alignment_disk_path(request_id: str) -> Path:
    safe = request_id.replace("/", "_").replace("\\", "_")
    return _DISK_IPC_DIR / f"{safe}.json"


def _write_alignment_to_disk(
    request_id: str,
    alignment: WordAlignment,
    *,
    commit_revision: bool = False,
) -> None:
    """``commit_revision`` rides along because the counter is useless where it
    is raised. Only the APIServer process's Prometheus registry is scraped —
    worker-process counters (``tts_alignment_register_total``,
    ``..._finalize_total``, and this one) read 0 on ``/metrics`` no matter what
    the worker did. The revision verdict is computed in the worker and consumed
    in the APIServer, so it travels on the drop that already crosses that
    boundary and is counted on arrival in ``_read_alignment_from_disk``."""
    try:
        path = _alignment_disk_path(request_id)
        payload = {
            "words": list(alignment.words),
            "word_start_time_seconds": [float(s) for s in alignment.word_start_time_seconds],
            "word_end_time_seconds": [float(e) for e in alignment.word_end_time_seconds],
            "commit_revision": bool(commit_revision),
            "written_at": time.time(),
        }
        fd, tmp_path = tempfile.mkstemp(dir=str(_DISK_IPC_DIR), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("disk-IPC write for %s failed: %s", request_id, e)


def _read_alignment_from_disk(request_id: str) -> WordAlignment | None:
    path = _alignment_disk_path(request_id)
    try:
        with open(path) as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("disk-IPC read for %s failed: %s", request_id, e)
        return None
    try:
        path.unlink()
    except OSError:
        pass
    if payload.get("commit_revision"):
        # Raised in the worker, counted here — see _write_alignment_to_disk.
        _inc(TTS_ALIGNMENT_COMMIT_REVISION_TOTAL)
    try:
        return WordAlignment(
            words=list(payload.get("words", [])),
            word_start_time_seconds=[float(s) for s in payload.get("word_start_time_seconds", [])],
            word_end_time_seconds=[float(e) for e in payload.get("word_end_time_seconds", [])],
        )
    except Exception as e:
        logger.warning("disk-IPC parse for %s failed: %s", request_id, e)
        return None


def _delete_alignment_from_disk(request_id: str) -> None:
    try:
        _alignment_disk_path(request_id).unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("disk-IPC delete for %s failed: %s", request_id, e)


# --- incremental publication ------------------------------------------------

# How often the commit horizon is re-evaluated, in codec frames (12.5 Hz, so
# 80 ms each). 4 frames = 320 ms: fine enough that a barge-in rarely waits on
# the next snapshot, coarse enough that the backtrace walk is noise next to the
# talker's own decode step. 0 disables incremental publication entirely, which
# restores the pre-existing "one alignment at finalize" behaviour exactly.
_PARTIAL_INTERVAL_FRAMES: int = int(os.environ.get("VLLM_TTS_WORD_TS_PARTIAL_FRAMES", "4"))
# Beam width for the commit horizon; see StreamingAligner._merge.
_COMMIT_MARGIN: float = float(os.environ.get("VLLM_TTS_WORD_TS_COMMIT_MARGIN", DEFAULT_COMMIT_MARGIN))


def _partial_disk_path(request_id: str) -> Path:
    safe = request_id.replace("/", "_").replace("\\", "_")
    return _DISK_IPC_DIR / f"{safe}.partial.json"


def _write_partial_to_disk(request_id: str, payload: dict) -> None:
    """Atomically republish the committed-so-far words.

    Whole-file replace, not append: the payload is cumulative, so a reader that
    misses a revision simply sees the next one, and ``os.replace`` means a
    reader never observes a half-written snapshot.
    """
    try:
        path = _partial_disk_path(request_id)
        fd, tmp_path = tempfile.mkstemp(dir=str(_DISK_IPC_DIR), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("partial disk-IPC write for %s failed: %s", request_id, e)


def _delete_partial_from_disk(request_id: str) -> None:
    try:
        _partial_disk_path(request_id).unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("partial disk-IPC delete for %s failed: %s", request_id, e)


def read_partial_alignment(request_id: str) -> dict | None:
    """Read the committed-so-far words for an IN-FLIGHT request.

    Called from the API-server process while audio is still streaming. Unlike
    ``pop_word_alignment`` this is a peek — the drop stays in place so the next
    poll sees the next revision; ``finalize`` / ``cancel`` remove it.

    Returns ``{"seq", "words", "word_start_time_seconds",
    "word_end_time_seconds", "committed_frames"}`` or ``None`` when nothing has
    been published yet. Times are CHUNK-LOCAL seconds, exactly like
    ``pop_word_alignment``'s — the caller adds the turn offset.
    """
    try:
        with open(_partial_disk_path(request_id)) as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("partial disk-IPC read for %s failed: %s", request_id, e)
        return None
    if not isinstance(payload, dict) or "words" not in payload:
        return None
    return payload


@dataclass
class _RequestState:
    text_token_ids: list[int]
    layer: int
    text_token_start: int
    text_token_end: int
    tokenizer: Any
    head: AlignmentPointerHead
    rate: float = CODEC_FRAME_RATE_HZ
    registered_at: float = field(default_factory=time.monotonic)
    prefill_text_chunks: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    decode_frames: list[torch.Tensor] = field(default_factory=list)
    # --- incremental publication state (worker process only) ---
    aligner: StreamingAligner | None = None
    # None until the first snapshot attempt; False once the head/spans turn out
    # to be unusable, so it is not retried every interval.
    partial_ok: bool | None = None
    frames_at_last_snapshot: int = 0
    frames_pushed: int = 0
    partial_seq: int = 0
    committed_words: list[dict] = field(default_factory=list)


@dataclass
class _FinalizedAlignment:
    """A finalized alignment plus when it was finalized, for TTL eviction.

    ``pop_word_alignment`` is the only reader, and it pops — so an alignment
    that is never read (aborted client, dropped stream) would be retained for
    the life of the process without an expiry bound.
    """

    alignment: WordAlignment
    finalized_at: float = field(default_factory=time.monotonic)


class WordAlignmentRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, _RequestState] = {}
        self._finalized: dict[str, _FinalizedAlignment] = {}

    def register(
        self,
        request_id: str,
        *,
        text_token_ids: list[int],
        text_token_start: int,
        text_token_end: int,
        layer: int,
        tokenizer: Any,
        head: AlignmentPointerHead,
        rate: float = CODEC_FRAME_RATE_HZ,
    ) -> None:
        with self._lock:
            self._inflight[request_id] = _RequestState(
                text_token_ids=list(text_token_ids),
                layer=int(layer),
                text_token_start=int(text_token_start),
                text_token_end=int(text_token_end),
                tokenizer=tokenizer,
                head=head,
                rate=float(rate),
            )
            self._finalized.pop(request_id, None)
            self._evict_expired_locked()
            inflight = len(self._inflight)
            finalized = len(self._finalized)
        _inc(TTS_ALIGNMENT_REGISTER_TOTAL)
        _set_gauge(TTS_ALIGNMENT_INFLIGHT, inflight)
        _set_gauge(TTS_ALIGNMENT_FINALIZED, finalized)

    def is_registered(self, request_id: str) -> bool:
        with self._lock:
            return self._resolve_locked(request_id, self._inflight) is not None

    def _resolve_locked(self, request_id: str, table: dict) -> str | None:
        """Resolve a request_id that may have a vLLM stage suffix.

        vLLM v1 appends a per-stage suffix to request_ids for multi-stage
        models (e.g. ``speech-abc123`` arrives at the talker batch slot as
        ``speech-abc123-stageXYZ``). We register under the API-level id;
        accumulate / finalize accept either form. Caller holds the lock.
        """
        if request_id in table:
            return request_id
        for rid in table.keys():
            if request_id.startswith(rid + "-") or rid.startswith(request_id + "-"):
                return rid
        return None

    def accumulate_prefill(
        self,
        request_id: str,
        chunk_aux_hidden: torch.Tensor,
        chunk_global_start: int,
        chunk_global_end: int,
    ) -> None:
        with self._lock:
            resolved = self._resolve_locked(request_id, self._inflight)
            if resolved is None:
                return
            state = self._inflight[resolved]
            overlap_start = max(chunk_global_start, state.text_token_start)
            overlap_end = min(chunk_global_end, state.text_token_end)
            if overlap_start >= overlap_end:
                return
            local_start = overlap_start - chunk_global_start
            local_end = overlap_end - chunk_global_start
            # Move to CPU so we don't pin VRAM for the full audio duration
            # of every concurrent request; finalize bounces back to GPU.
            chunk = chunk_aux_hidden[local_start:local_end].detach().to(device="cpu", copy=True)
            state.prefill_text_chunks.append((overlap_start, chunk))

    def accumulate_decode(self, request_id: str, frame_aux_hidden: torch.Tensor) -> None:
        with self._lock:
            resolved = self._resolve_locked(request_id, self._inflight)
            if resolved is None:
                return
            state = self._inflight[resolved]
            row = frame_aux_hidden
            if row.dim() == 2:
                row = row.squeeze(0)
            state.decode_frames.append(row.detach().to(device="cpu", copy=True))
            # Snapshot cadence is decided here (cheap, under the lock); the
            # projection + backtrace walk runs below, OUTSIDE it. The registry
            # lock is process-global, and holding it across the readout would
            # serialise every concurrent request's decode step behind one
            # request's alignment work.
            due = (
                _PARTIAL_INTERVAL_FRAMES > 0
                and state.partial_ok is not False
                and len(state.decode_frames) - state.frames_at_last_snapshot >= _PARTIAL_INTERVAL_FRAMES
            )
            if due:
                state.frames_at_last_snapshot = len(state.decode_frames)
        if due:
            # Safe without the lock: register / accumulate / finalize all run on
            # the single model-runner thread of this process, so nothing else
            # mutates `state`. The API server only ever reads the disk drop.
            self._publish_partial(resolved, state)

    @staticmethod
    def _build_aligner(state: _RequestState) -> StreamingAligner | None:
        """Word keys are prompt-fixed, so this is built once, after prefill."""
        if not StreamingAligner.supports(state.head):
            return None
        if not state.prefill_text_chunks:
            return None
        spans = segment_words(state.text_token_ids, state.tokenizer)
        if not spans:
            return None
        chunks_sorted = sorted(state.prefill_text_chunks, key=lambda t: t[0])
        text_hidden = torch.cat([c[1] for c in chunks_sorted], dim=0)
        head_device = next(state.head.parameters()).device
        head_dtype = next(state.head.parameters()).dtype
        text_hidden = text_hidden.to(device=head_device, dtype=head_dtype)
        with torch.no_grad():
            word_keys = pool_word_keys(text_hidden, spans)
        if word_keys.shape[0] == 0:
            return None
        return StreamingAligner(
            state.head,
            word_keys,
            spans,
            rate=state.rate,
            commit_margin=_COMMIT_MARGIN,
        )

    def _publish_partial(self, request_id: str, state: _RequestState) -> None:
        """Advance the commit horizon and republish it if it moved.

        Never raises: a failure here disables incremental publication for THIS
        request and leaves the batch ``finalize`` path untouched, because a
        broken mid-flight readout must not cost the caller its end-of-chunk
        timestamps.
        """
        try:
            if state.aligner is None:
                aligner = self._build_aligner(state)
                if aligner is None:
                    state.partial_ok = False
                    _inc(TTS_ALIGNMENT_PARTIAL_TOTAL, outcome="unsupported_head")
                    return
                state.aligner = aligner
                state.partial_ok = True
            n_have = len(state.decode_frames)
            if n_have > state.frames_pushed:
                new_frames = torch.stack(state.decode_frames[state.frames_pushed : n_have], dim=0)
                state.aligner.push(new_frames)
                state.frames_pushed = n_have
            words = state.aligner.committed_words()
            if len(words) <= len(state.committed_words):
                _inc(TTS_ALIGNMENT_PARTIAL_TOTAL, outcome="no_new_words")
                return
            n_new = len(words) - len(state.committed_words)
            state.committed_words = words
            state.partial_seq += 1
            _write_partial_to_disk(
                request_id,
                {
                    "seq": state.partial_seq,
                    "words": [w["word"] for w in words],
                    "word_start_time_seconds": [float(w["start"]) for w in words],
                    "word_end_time_seconds": [float(w["end"]) for w in words],
                    "committed_frames": state.aligner.n_committed_frames,
                    "written_at": time.time(),
                },
            )
            _inc(TTS_ALIGNMENT_PARTIAL_TOTAL, outcome="published")
            _observe(TTS_ALIGNMENT_PARTIAL_WORDS, n_new)
        except Exception as e:
            state.partial_ok = False
            _inc(TTS_ALIGNMENT_PARTIAL_TOTAL, outcome="error")
            logger.warning(
                "partial snapshot for %s failed; incremental timestamps disabled for this request: %s",
                request_id,
                e,
            )

    @staticmethod
    def _check_commit_prefix(request_id: str, committed: list[dict], final_words: list[dict]) -> bool:
        """Did anything published mid-synthesis get contradicted at the end?

        The commit horizon is beam-pruned, so this is the guard that turns a
        theoretical revision into an observable one. Reported, never corrected:
        the words are already on the wire. Returns True on a disagreement.
        """
        n = len(committed)
        if final_words[:n] == committed:
            return False
        _inc(TTS_ALIGNMENT_COMMIT_REVISION_TOTAL)
        first = next(
            (i for i in range(n) if i >= len(final_words) or final_words[i] != committed[i]),
            n,
        )
        logger.warning(
            "commit-horizon revision for %s: published %d words, final decode has %d; "
            "first disagreement at index %d (published=%s final=%s). "
            "Raise VLLM_TTS_WORD_TS_COMMIT_MARGIN (currently %.1f) if this recurs.",
            request_id,
            n,
            len(final_words),
            first,
            committed[first] if first < n else None,
            final_words[first] if first < len(final_words) else None,
            _COMMIT_MARGIN,
        )
        return True

    def finalize(self, request_id: str) -> WordAlignment | None:
        with self._lock:
            resolved = self._resolve_locked(request_id, self._inflight)
            if resolved is None:
                _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="not_registered")
                return None
            state = self._inflight.pop(resolved)
            registered_id = resolved
            inflight_after_pop = len(self._inflight)
        _set_gauge(TTS_ALIGNMENT_INFLIGHT, inflight_after_pop)
        # The request is over: the serving layer takes the rest of the words
        # from pop_word_alignment, so the in-flight drop has no further reader.
        _delete_partial_from_disk(registered_id)

        if not state.decode_frames or not state.prefill_text_chunks:
            _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="empty_frames")
            return None

        all_spans = segment_words(state.text_token_ids, state.tokenizer)
        if not all_spans:
            _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="empty_spans")
            return None

        chunks_sorted = sorted(state.prefill_text_chunks, key=lambda t: t[0])
        text_hidden = torch.cat([c[1] for c in chunks_sorted], dim=0)
        frame_hidden = torch.stack(state.decode_frames, dim=0)

        head_device = next(state.head.parameters()).device
        head_dtype = next(state.head.parameters()).dtype
        text_hidden = text_hidden.to(device=head_device, dtype=head_dtype)
        frame_hidden = frame_hidden.to(device=head_device, dtype=head_dtype)

        try:
            with torch.no_grad():
                word_keys = pool_word_keys(text_hidden, all_spans)
                if word_keys.shape[0] == 0:
                    _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="no_words")
                    return None
                scores = state.head(frame_hidden, word_keys)
                if os.environ.get("MSTAR_ALIGN_DEBUG"):
                    fh, th = frame_hidden.float(), text_hidden.float()
                    logger.warning(
                        "ALIGN_DEBUG %s frames=%s text=%s keys=%s | "
                        "frame mean=%.4f std=%.4f rowstd=%.4f | "
                        "text mean=%.4f std=%.4f rowstd=%.4f | "
                        "scores %s min=%.3f max=%.3f argmax=%s",
                        request_id, tuple(frame_hidden.shape), tuple(text_hidden.shape),
                        tuple(word_keys.shape),
                        fh.mean().item(), fh.std().item(), fh.std(dim=0).mean().item(),
                        th.mean().item(), th.std().item(), th.std(dim=0).mean().item(),
                        tuple(scores.shape), scores.float().min().item(),
                        scores.float().max().item(),
                        scores.float().argmax(dim=-1).tolist()[:40],
                    )
                word_dicts = word_timestamps_viterbi_full_coverage(scores, all_spans, rate=state.rate)
        except Exception as e:
            logger.exception("finalize(%s) raised — skipping: %s", request_id, e)
            _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="error")
            return None

        if not word_dicts:
            _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="no_words")
            return None

        commit_revision = bool(
            state.committed_words and self._check_commit_prefix(registered_id, state.committed_words, word_dicts)
        )

        alignment = WordAlignment(
            words=[d["word"] for d in word_dicts],
            word_start_time_seconds=[float(d["start"]) for d in word_dicts],
            word_end_time_seconds=[float(d["end"]) for d in word_dicts],
        )

        with self._lock:
            self._evict_expired_locked()
            self._finalized[registered_id] = _FinalizedAlignment(alignment)
            finalized_after = len(self._finalized)
        _set_gauge(TTS_ALIGNMENT_FINALIZED, finalized_after)
        _inc(TTS_ALIGNMENT_FINALIZE_TOTAL, outcome="success")
        _write_alignment_to_disk(registered_id, alignment, commit_revision=commit_revision)
        return alignment

    def pop_word_alignment(
        self,
        request_id: str,
        wait_timeout_s: float = 2.0,
        poll_interval_s: float = 0.01,
    ) -> WordAlignment | None:
        """Return + remove the finalized alignment, polling briefly.

        The serving layer calls this immediately after the engine
        generator's final yield, but the runner's finalize fires one
        scheduler tick later (in ``_update_states.finished_req_ids``).
        Short poll bridges the gap. Returns ``None`` on timeout —
        request was aborted or no frames were captured.
        """
        start = time.monotonic()
        deadline = start + max(0.0, wait_timeout_s)
        while True:
            with self._lock:
                resolved = self._resolve_locked(request_id, self._finalized)
                if resolved is not None:
                    alignment = self._finalized.pop(resolved).alignment
                    finalized_after = len(self._finalized)
                    _set_gauge(TTS_ALIGNMENT_FINALIZED, finalized_after)
                    _inc(TTS_ALIGNMENT_POP_TOTAL, outcome="found_memory")
                    _observe(TTS_ALIGNMENT_POP_WAIT, time.monotonic() - start)
                    # finalize() writes the alignment to both memory and a disk
                    # IPC drop. Serving the pop from memory must also drain the
                    # disk copy, else a later pop re-reads the stale drop and
                    # returns the same alignment again instead of None.
                    _delete_alignment_from_disk(resolved)
                    return alignment
            alignment = _read_alignment_from_disk(request_id)
            if alignment is not None:
                _inc(TTS_ALIGNMENT_POP_TOTAL, outcome="found_disk")
                _observe(TTS_ALIGNMENT_POP_WAIT, time.monotonic() - start)
                return alignment
            if time.monotonic() >= deadline:
                logger.warning("pop_word_alignment(%s) timed out after %.2fs", request_id, wait_timeout_s)
                _inc(TTS_ALIGNMENT_POP_TOTAL, outcome="timeout")
                _observe(TTS_ALIGNMENT_POP_WAIT, time.monotonic() - start)
                return None
            time.sleep(poll_interval_s)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            resolved_in = self._resolve_locked(request_id, self._inflight)
            if resolved_in is not None:
                self._inflight.pop(resolved_in, None)
            resolved_fin = self._resolve_locked(request_id, self._finalized)
            if resolved_fin is not None:
                self._finalized.pop(resolved_fin, None)
            inflight = len(self._inflight)
            finalized = len(self._finalized)
        _delete_alignment_from_disk(request_id)
        _delete_partial_from_disk(request_id)
        _inc(TTS_ALIGNMENT_CANCEL_TOTAL)
        _set_gauge(TTS_ALIGNMENT_INFLIGHT, inflight)
        _set_gauge(TTS_ALIGNMENT_FINALIZED, finalized)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"inflight": len(self._inflight), "finalized": len(self._finalized)}

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [rid for rid, st in self._inflight.items() if now - st.registered_at > _REGISTRY_TTL_SECONDS]
        for rid in expired:
            self._inflight.pop(rid, None)
            _delete_partial_from_disk(rid)
        # Unread finalized alignments get the same bound: the only reader is
        # pop_word_alignment, so without this an alignment nobody reads (the
        # client hung up before the timestamps frame) is retained forever —
        # as is its disk-IPC drop.
        expired_finalized = [
            rid for rid, entry in self._finalized.items() if now - entry.finalized_at > _REGISTRY_TTL_SECONDS
        ]
        for rid in expired_finalized:
            self._finalized.pop(rid, None)
            _delete_alignment_from_disk(rid)
        if expired or expired_finalized:
            logger.warning(
                "evicted %d expired registrations and %d unread alignments: %s",
                len(expired),
                len(expired_finalized),
                expired + expired_finalized,
            )
            if _METRICS_ENABLED:
                try:
                    TTS_ALIGNMENT_EVICT_TOTAL.inc(len(expired) + len(expired_finalized))
                except Exception:
                    pass


_GLOBAL_REGISTRY: WordAlignmentRegistry | None = None


def get_registry() -> WordAlignmentRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = WordAlignmentRegistry()
    return _GLOBAL_REGISTRY
