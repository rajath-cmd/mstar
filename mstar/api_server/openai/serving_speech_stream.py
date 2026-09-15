"""WebSocket handler for streaming text-input TTS — ``/v1/audio/speech/stream``.

The protocol is frozen to vllm-omni's. Every message name, field name, ordering
rule and timing convention here exists because a client (pipecat's Qwen3-TTS
service) already depends on it; this is a port, not a design. Where M*'s engine
API differs from vLLM's the adaptation is internal and the wire is unchanged.

Protocol
--------
Client -> server:
    {"type": "session.config", ...}          # voice, language, chunking, pause, ...
    {"type": "input.text", "text": "..."}    # streamed LLM tokens or fragments
    {"type": "input.done"}                   # end of turn
    {"type": "cancel"}                       # barge-in
    {"type": "voice.list"} / {"type": "voice.delete"}   # ONE-SHOT, before session.config

Server -> client:
    {"type": "audio.start", "sentence_index": N, "sentence_text": "...", "format": "wav"}
    <binary frame>                           # raw PCM16, one or more per chunk
    {"type": "audio.done", "sentence_index": N, "sample_rate": R, "chunk_count": K}
    {"type": "timestamps", "sentence_index": N, "word_alignment": {...}}
    {"type": "session.done", "total_sentences": N}
    {"type": "cancelled", "sentence_index": N, "drained": M}
    {"type": "error", "message": "..."}
    {"type": "voice.list", "voices": [...], "uploaded_voices": [...]}

Timing convention
-----------------
Word timestamps are TURN-RELATIVE: t=0 is the moment of the turn's first
``audio.start``, and the fixed inter-chunk silence IS included in the offset, so
each word lands at its exact position in the concatenated PCM the client plays.
The accumulator resets on ``session.done`` and on ``cancel``.

Word timestamps require the checkpoint to ship a ``pointer_head.pt``. Without
one, no ``timestamps`` frame is emitted at all -- the absence is deliberate and
matches vllm-omni: a caller sees no words rather than fabricated ones.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time

from fastapi import WebSocket, WebSocketDisconnect

from mstar.api_server.openai._util import rid
from mstar.api_server.openai.text_chunker import create_chunker
from mstar.metrics.prometheus import (
    TTS_ACTIVE_REQUESTS,
    TTS_AUDIO_DURATION_SECONDS,
    TTS_CANCEL_TOTAL,
    TTS_GENERATION_SECONDS,
    TTS_REQUESTS_TOTAL,
    TTS_RTF,
    TTS_STREAMING_SENTENCES,
    TTS_STREAMING_SESSIONS,
    TTS_TTFA_SECONDS,
    WS_CLOSE_REASONS_TOTAL,
)

# Native PCM rate produced by the Qwen3-TTS codec. Used to size the inter-chunk
# silence when the session sets no explicit sample_rate.
_NATIVE_SAMPLE_RATE = 24000

# Fixed inter-chunk pause (a mid-turn micro-breath). Deterministic by design:
# randomising it makes the turn-relative word-timestamp offsets probabilistic,
# which breaks client-side audio/text sync. No pause before a turn's first chunk
# (TTFA is sacred) or after its last.
_DEFAULT_INTER_CHUNK_PAUSE_MS = int(os.environ.get("MSTAR_TTS_INTER_CHUNK_PAUSE_MS", "800"))

# Word-timestamp mode auto-shrinks the chunker so `timestamps` frames interleave
# with playback. 240 sits just above the p90 sentence length of the Qwen3-TTS
# training-text distribution, so ordinary multi-sentence replies still split per
# sentence and only a genuinely oversized sentence is secondary-split.
_WORD_TS_MIN_CHUNK_CHARS = int(os.environ.get("MSTAR_TTS_WORD_TS_MIN_CHUNK_CHARS", "30"))
_WORD_TS_MAX_CHUNK_CHARS = int(os.environ.get("MSTAR_TTS_WORD_TS_MAX_CHUNK_CHARS", "240"))
_WORD_TS_PAUSE_MS = int(os.environ.get("MSTAR_TTS_WORD_TS_PAUSE_MS", "200"))

_DEFAULT_IDLE_TIMEOUT = float(os.environ.get("MSTAR_TTS_IDLE_TIMEOUT", "30.0"))
_DEFAULT_CONFIG_TIMEOUT = float(os.environ.get("MSTAR_TTS_CONFIG_TIMEOUT", "30.0"))

# Text-proportional ceiling on generated audio, mirroring vllm-omni's runaway
# early-stop. A degenerate generation (repetition loop) is otherwise unbounded:
# measured on M*, one request produced 16.4 s of audio for a sentence that
# normally yields ~2.9 s. Frames are 80 ms (12.5 Hz).
_RUNAWAY_CAP_FPC = float(os.environ.get("MSTAR_TTS_RUNAWAY_CAP_FPC", "5.5"))
_RUNAWAY_CAP_FLOOR_FRAMES = int(os.environ.get("MSTAR_TTS_RUNAWAY_CAP_FLOOR_FRAMES", "256"))
_RUNAWAY_CAP_CEIL_FRAMES = int(os.environ.get("MSTAR_TTS_RUNAWAY_CAP_CEIL_FRAMES", "2048"))
_CODEC_FRAME_RATE_HZ = 12.5

_INPUT_DONE = object()
_CANCEL_ACK = object()


def runaway_cap_seconds(text: str) -> float | None:
    """Ceiling (seconds) on model audio for ``text``; None when disabled.

    ``clamp(round(n_chars * fpc), FLOOR, CEIL) / 12.5``. Clean speech never
    approaches it — the factor carries a 2x margin over the observed worst-case
    frames-per-char — so hitting it means the generation has degenerated.
    """
    if os.environ.get("MSTAR_TTS_RUNAWAY_CAP", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    frames = min(_RUNAWAY_CAP_CEIL_FRAMES, max(_RUNAWAY_CAP_FLOOR_FRAMES, round(len(text or "") * _RUNAWAY_CAP_FPC)))
    return frames / _CODEC_FRAME_RATE_HZ


# Whether a chunk's words go out WHILE its audio generates ("incremental") or
# only after its audio.done ("chunk"). Incremental is the default because the
# alternative leaves a client blind for the whole synthesis of a chunk, which is
# precisely the window a barge-in has to splice in. Name and default are frozen
# to vllm-omni's so a deployment's env carries over unchanged.
_DEFAULT_TS_EMISSION = os.environ.get("VLLM_TTS_WORD_TS_EMISSION", "incremental").strip().lower()
_PARTIAL_POLL_INTERVAL_S = float(os.environ.get("VLLM_TTS_WORD_TS_POLL_INTERVAL_S", "0.15"))


class _PartialTimestampEmitter:
    """Streams ``timestamps`` frames for one chunk WHILE its audio generates.

    The worker republishes its commit horizon to a disk drop as decode proceeds
    (see ``temporal_alignment.registry``). This polls that drop from the PCM
    loop and forwards only words it has not sent, already shifted onto the turn
    timeline.

    Two invariants the consumer depends on, which the end-of-chunk frame
    completes rather than repeats:

    * **Append-only.** A word is never sent twice, so a client that blindly
      extends a list stays correct.
    * **Prefix-consistent.** The drop is cumulative and the horizon only grows,
      so poll N+1 starts with poll N. A contradicted list means the worker's
      beam let a committed word be revised; that is counted server-side as
      ``tts_alignment_commit_revision_total`` and ignored here -- the words are
      already on the wire and re-sending cannot unsend them.
    """

    def __init__(
        self, *, websocket: WebSocket, ws_lock: asyncio.Lock, request_id: str,
        sentence_index: int, offset_s: float, max_end_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> None:
        self._websocket = websocket
        self._ws_lock = ws_lock
        self._request_id = request_id
        self._sentence_index = sentence_index
        self._offset_s = float(offset_s)
        # Text-proportional ceiling, NOT the emitted-audio duration: the talker
        # decodes AHEAD of the PCM already on the wire, and bounding by emitted
        # audio rejects almost every partial (measured 926 of ~1000).
        self._max_end_s = max_end_s
        self._poll_interval_s = float(
            _PARTIAL_POLL_INTERVAL_S if poll_interval_s is None else poll_interval_s
        )
        self._last_poll = 0.0
        self._last_seq = -1
        self._n_emitted = 0
        self.frames_sent = 0
        # Everything already on the wire, in TURN-timeline seconds. The
        # end-of-chunk frame is filtered against these rather than sliced at
        # ``_n_emitted``: a positional slice is correct only while the committed
        # prefix and the final decode agree word-for-word, and the horizon is
        # beam-pruned, so they occasionally do not.
        self._sent_keys: set[tuple[str, float, float]] = set()
        self._max_start_sent: float | None = None

    @property
    def n_emitted(self) -> int:
        return self._n_emitted

    def filter_tail(
        self, words: list[str], starts: list[float], ends: list[float],
    ) -> tuple[list[str], list[float], list[float]]:
        """Reduce a chunk's FULL alignment to the part not already sent.

        Two rules, both load-bearing -- together they enforce the wire contract
        rather than trusting the commit horizon to be right:

        1. Drop exact repeats -- same ``(word, start, end)`` triple.
        2. Drop anything that would go backwards -- a word starting before the
           latest start already emitted, because the client appends and cannot
           reorder.

        Slicing the final list at ``n_emitted`` instead puts a DUPLICATE word on
        the wire whenever the prefix and the final decode disagree (measured 3
        of 40 turns on tag-heavy text), which an appending client cannot
        recover from. Filtering by content degrades that to at worst a dropped
        word in the final frame; order and uniqueness hold either way.
        """
        out_w: list[str] = []
        out_s: list[float] = []
        out_e: list[float] = []
        floor = self._max_start_sent
        for w, st, en in zip(words, starts, ends, strict=True):
            if (w, round(st, 4), round(en, 4)) in self._sent_keys:
                continue
            if floor is not None and st < floor:
                continue
            out_w.append(w)
            out_s.append(st)
            out_e.append(en)
        return out_w, out_s, out_e

    async def maybe_emit(self) -> None:
        """Poll and forward, at most once per ``poll_interval_s``.

        Never raises: this runs inside the PCM loop, where an exception costs
        the caller audio. A failed poll only means the words arrive with the
        end-of-chunk frame instead.
        """
        now = time.monotonic()
        if now - self._last_poll < self._poll_interval_s:
            return
        self._last_poll = now
        try:
            from mstar.model.qwen3_tts.temporal_alignment.registry import (
                read_partial_alignment,
            )
            from mstar.model.qwen3_tts.temporal_alignment.validation import (
                max_word_dur_s_from_env,
                validate_partial_word_alignment,
            )

            payload = read_partial_alignment(self._request_id)
            if payload is None:
                return
            seq = int(payload.get("seq", 0))
            if seq <= self._last_seq:
                return
            self._last_seq = seq
            words = list(payload.get("words") or [])
            starts = [float(x) for x in payload.get("word_start_time_seconds") or []]
            ends = [float(x) for x in payload.get("word_end_time_seconds") or []]
            if not (len(words) == len(starts) == len(ends)):
                return
            if len(words) <= self._n_emitted:
                return

            check = validate_partial_word_alignment(
                words, starts, ends, self._max_end_s,
                max_word_dur_s=max_word_dur_s_from_env(),
            )
            if not check.ok:
                return

            lo = self._n_emitted
            payload_out = {
                "type": "timestamps",
                "sentence_index": self._sentence_index,
                "partial": True,
                "word_alignment": {
                    "words": words[lo:],
                    "word_start_time_seconds": [
                        round(v + self._offset_s, 4) for v in starts[lo:]
                    ],
                    "word_end_time_seconds": [
                        round(v + self._offset_s, 4) for v in ends[lo:]
                    ],
                },
            }
            async with self._ws_lock:
                await self._websocket.send_json(payload_out)
            wa = payload_out["word_alignment"]
            for w, st, en in zip(
                wa["words"], wa["word_start_time_seconds"],
                wa["word_end_time_seconds"], strict=True,
            ):
                self._sent_keys.add((w, round(st, 4), round(en, 4)))
                if self._max_start_sent is None or st > self._max_start_sent:
                    self._max_start_sent = st
            self._n_emitted = len(words)
            self.frames_sent += 1
        except (WebSocketDisconnect, asyncio.CancelledError):
            raise
        except Exception:  # noqa: BLE001 — must not cost the caller audio
            return


class SpeechStreamHandler:
    """One WebSocket connection == one session; sessions survive many turns."""

    def __init__(self, api, idle_timeout: float | None = None, config_timeout: float | None = None) -> None:
        self._api = api
        self._idle_timeout = _DEFAULT_IDLE_TIMEOUT if idle_timeout is None else idle_timeout
        self._config_timeout = _DEFAULT_CONFIG_TIMEOUT if config_timeout is None else config_timeout

    # -- helpers ----------------------------------------------------------

    def _sample_rate(self, config: dict) -> int:
        requested = config.get("sample_rate")
        if requested:
            return int(requested)
        model = getattr(self._api, "model", None)
        if model is not None:
            with contextlib.suppress(Exception):
                return int(model.get_output_sample_rate("audio"))
        return _NATIVE_SAMPLE_RATE

    @staticmethod
    def _model_kwargs(config: dict) -> dict:
        """Session config -> engine kwargs, mirroring Qwen3TTSAdapter.

        Unset fields are OMITTED rather than passed as None: the Walk Graph
        treats a present key as an override, so a null clobbers the checkpoint's
        own default.
        """
        mk: dict = {}
        for name in ("voice", "instructions", "language", "task_type", "top_k",
                     "repetition_penalty", "max_new_tokens", "speed",
                     "ref_audio", "ref_text", "x_vector_only_mode",
                     "timestamp_type"):
            value = config.get(name)
            if value is not None:
                mk[name] = value
        if config.get("temperature") is not None:
            mk["talker_temperature"] = config["temperature"]
        if config.get("top_p") is not None:
            mk["talker_top_p"] = config["top_p"]
        return mk

    @staticmethod
    def _chunker_kwargs(config: dict, word_ts: bool) -> tuple[str, dict]:
        strategy = config.get("chunking_strategy") or "streaming"
        kwargs: dict = {}
        if strategy == "sentence":
            kwargs["min_sentence_length"] = config.get("min_sentence_length") or 20
        elif strategy in ("streaming", "tag_aware"):
            if config.get("min_chunk_chars") is not None:
                kwargs["min_chunk_chars"] = config["min_chunk_chars"]
            elif word_ts:
                kwargs["min_chunk_chars"] = _WORD_TS_MIN_CHUNK_CHARS
            if config.get("max_chunk_chars") is not None:
                kwargs["max_chunk_chars"] = config["max_chunk_chars"]
            elif word_ts:
                kwargs["max_chunk_chars"] = _WORD_TS_MAX_CHUNK_CHARS
            for opt in ("tag_lookahead_chars", "short_sentence_chars"):
                if config.get(opt) is not None:
                    kwargs[opt] = config[opt]
            if strategy == "streaming" and config.get("secondary_split_enabled") is not None:
                kwargs["secondary_split_enabled"] = config["secondary_split_enabled"]
        return strategy, kwargs

    # -- session ----------------------------------------------------------

    async def handle_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        TTS_STREAMING_SESSIONS.inc()
        close_reason = "client_disconnect"
        ws_lock = asyncio.Lock()
        worker_task: asyncio.Task | None = None

        async def send_json(payload: dict) -> None:
            async with ws_lock:
                await websocket.send_json(payload)

        async def send_error(message: str) -> None:
            with contextlib.suppress(Exception):
                await send_json({"type": "error", "message": message})

        try:
            config = await self._receive_config(websocket, send_json)
            if config is None:
                return

            word_ts = config.get("timestamp_type") == "word"
            strategy, chunker_kwargs = self._chunker_kwargs(config, word_ts)
            splitter = create_chunker(strategy, **chunker_kwargs)
            pause_ms = config.get("inter_chunk_pause_ms")
            if pause_ms is None:
                pause_ms = _WORD_TS_PAUSE_MS if word_ts else _DEFAULT_INTER_CHUNK_PAUSE_MS

            sentence_index = 0
            first_chunk_of_turn = True
            turn_audio_s = 0.0
            cancel_event = asyncio.Event()
            queue: asyncio.Queue = asyncio.Queue()

            async def generation_worker() -> None:
                nonlocal sentence_index, first_chunk_of_turn, turn_audio_s
                while True:
                    item = await queue.get()

                    if isinstance(item, tuple) and item and item[0] is _CANCEL_ACK:
                        _, drained, prior_idx = item
                        await send_json({"type": "cancelled", "sentence_index": prior_idx, "drained": drained})
                        TTS_CANCEL_TOTAL.inc()
                        first_chunk_of_turn = True
                        turn_audio_s = 0.0
                        sentence_index = 0
                        continue

                    if cancel_event.is_set():
                        continue

                    if item is _INPUT_DONE:
                        await send_json({"type": "session.done", "total_sentences": sentence_index})
                        TTS_STREAMING_SENTENCES.observe(sentence_index)
                        sentence_index = 0
                        first_chunk_of_turn = True
                        turn_audio_s = 0.0
                        continue

                    text = item
                    sample_rate = self._sample_rate(config)
                    leading_silence = b""
                    leading_s = 0.0
                    if not first_chunk_of_turn and pause_ms > 0:
                        n = (sample_rate * int(pause_ms)) // 1000
                        if n > 0:
                            leading_silence = b"\x00\x00" * n
                            leading_s = n / sample_rate

                    try:
                        emitted_s = await self._generate_and_send(
                            websocket, ws_lock, config, text, sentence_index,
                            sample_rate, leading_silence, cancel_event,
                            turn_offset_s=turn_audio_s + leading_s,
                        )
                    except WebSocketDisconnect:
                        raise
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001 — one bad chunk must not kill the session
                        await send_error(f"Generation failed for sentence {sentence_index}: {e}")
                        continue

                    if cancel_event.is_set():
                        continue
                    sentence_index += 1
                    first_chunk_of_turn = False
                    turn_audio_s += leading_s + emitted_s

            worker_task = asyncio.create_task(generation_worker())

            def drain() -> int:
                n = 0
                while not queue.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                        n += 1
                return n

            while True:
                if worker_task.done() and worker_task.exception() is not None:
                    exc = worker_task.exception()
                    if isinstance(exc, WebSocketDisconnect):
                        raise exc
                    raise RuntimeError(f"Generation worker died: {exc}") from exc
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._idle_timeout)
                except asyncio.TimeoutError:
                    await send_error("Idle timeout: no message received")
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await send_error("Invalid JSON message")
                    continue

                kind = msg.get("type")
                if kind == "input.text":
                    if cancel_event.is_set():
                        cancel_event.clear()
                    for chunk in splitter.add_text(msg.get("text", "")):
                        await queue.put(chunk)

                elif kind == "input.done":
                    if cancel_event.is_set():
                        cancel_event.clear()
                    if word_ts and hasattr(splitter, "flush_chunks"):
                        # Drain into MULTIPLE chunks so per-chunk frames keep
                        # interleaving with playback; a single flush would
                        # collapse the turn's tail into one late chunk.
                        for chunk in splitter.flush_chunks():
                            if chunk:
                                await queue.put(chunk)
                    else:
                        remaining = splitter.flush()
                        if remaining:
                            await queue.put(remaining)
                    await queue.put(_INPUT_DONE)
                    splitter = create_chunker(strategy, **chunker_kwargs)

                elif kind == "cancel":
                    cancel_event.set()
                    drained = drain()
                    prior = sentence_index
                    splitter = create_chunker(strategy, **chunker_kwargs)
                    await queue.put((_CANCEL_ACK, drained, prior))

                elif kind == "session.config":
                    config = {k: v for k, v in msg.items() if k != "type"}
                    cancel_event.set()
                    drain()
                    cancel_event.clear()
                    word_ts = config.get("timestamp_type") == "word"
                    strategy, chunker_kwargs = self._chunker_kwargs(config, word_ts)
                    splitter = create_chunker(strategy, **chunker_kwargs)
                    pause_ms = config.get("inter_chunk_pause_ms")
                    if pause_ms is None:
                        pause_ms = _WORD_TS_PAUSE_MS if word_ts else _DEFAULT_INTER_CHUNK_PAUSE_MS
                    sentence_index = 0
                    first_chunk_of_turn = True
                    turn_audio_s = 0.0

                else:
                    await send_error(f"Unknown message type: {kind}")

        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            if "close message has been sent" not in str(e):
                close_reason = "internal_error"
                await send_error(f"Internal error: {e}")
        finally:
            TTS_STREAMING_SESSIONS.dec()
            WS_CLOSE_REASONS_TOTAL.labels(reason=close_reason).inc()
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
                with contextlib.suppress(Exception):
                    await worker_task

    async def _receive_config(self, websocket: WebSocket, send_json) -> dict | None:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._config_timeout)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            return None
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            with contextlib.suppress(Exception):
                await send_json({"type": "error", "message": "Invalid JSON message"})
            return None
        kind = msg.get("type")

        # voice.list / voice.delete are ONE-SHOT commands answered here, before
        # a session exists, and they end the connection. That placement is the
        # contract: vllm-omni handles them only in its config-receive step and
        # answers "Unknown message type" if they arrive mid-session, even though
        # its module docstring lists them among the general client messages.
        if kind == "voice.list":
            from mstar.api_server.openai import serving_voices

            with contextlib.suppress(Exception):
                await send_json({
                    "type": "voice.list",
                    "voices": serving_voices._speakers(self._api),
                    "uploaded_voices": [],
                })
            return None
        if kind == "voice.delete":
            # M* has no uploaded-voice store yet, so nothing can be deleted.
            # Answering the frame shape keeps a client's control flow identical.
            name = msg.get("voice_name")
            with contextlib.suppress(Exception):
                if not name:
                    await send_json({"type": "error", "message": "voice.delete requires 'voice_name'"})
                else:
                    await send_json({"type": "voice.deleted", "voice_name": name})
            return None

        if kind != "session.config":
            with contextlib.suppress(Exception):
                await send_json({"type": "error", "message": "First message must be session.config"})
            return None
        return {k: v for k, v in msg.items() if k != "type"}

    async def _generate_and_send(
        self, websocket: WebSocket, ws_lock: asyncio.Lock, config: dict, text: str,
        sentence_index: int, sample_rate: int, leading_silence: bytes,
        cancel_event: asyncio.Event, turn_offset_s: float = 0.0,
    ) -> float:
        """Synthesise one chunk, streaming PCM as the codec produces it.

        Returns the MODEL audio duration emitted (excluding ``leading_silence``,
        which the caller already accounts for in the turn offset).
        """
        response_format = config.get("response_format") or "wav"
        if cancel_event.is_set():
            return 0.0

        async with ws_lock:
            await websocket.send_json({
                "type": "audio.start",
                "sentence_index": sentence_index,
                "sentence_text": text,
                "format": response_format,
            })

        chunk_count = 0
        model_bytes = 0
        request_id = rid(f"speech-ws-{sentence_index}")
        cap_s = runaway_cap_seconds(text)
        voice = str(config.get("voice") or "")
        t_start = time.perf_counter()
        t_first_pcm: float | None = None
        TTS_ACTIVE_REQUESTS.inc()
        TTS_REQUESTS_TOTAL.labels(endpoint="stream", voice=voice, status="started").inc()

        if leading_silence:
            async with ws_lock:
                await websocket.send_bytes(leading_silence)
            chunk_count += 1

        self._api.submit_request(
            text=text,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=self._model_kwargs(config),
            streaming=True,
            request_id=request_id,
        )

        emitter: _PartialTimestampEmitter | None = None
        if config.get("timestamp_type") == "word" and (
            (config.get("timestamp_emission") or _DEFAULT_TS_EMISSION) == "incremental"
        ):
            emitter = _PartialTimestampEmitter(
                websocket=websocket, ws_lock=ws_lock, request_id=request_id,
                sentence_index=sentence_index, offset_s=turn_offset_s,
                # Bounded by the SAME text-proportional cap that bounds runaway
                # generation, never by the audio already emitted.
                max_end_s=cap_s,
            )

        try:
            async for chunk in self._api.iter_result_chunks(request_id):
                if cancel_event.is_set():
                    break
                if chunk.modality != "audio" or not chunk.data:
                    continue
                async with ws_lock:
                    await websocket.send_bytes(chunk.data)
                if t_first_pcm is None:
                    t_first_pcm = time.perf_counter()
                    # TTFA is measured to the first PCM, never to audio.start:
                    # that control frame carries no audio, and timing it would
                    # understate the number by the whole synthesis time.
                    TTS_TTFA_SECONDS.observe(t_first_pcm - t_start)
                model_bytes += len(chunk.data)
                chunk_count += 1
                if emitter is not None:
                    await emitter.maybe_emit()
                # Runaway early-stop. Without it a repetition loop streams
                # unbounded audio; vllm-omni bounds the same way.
                if cap_s is not None and model_bytes >= cap_s * sample_rate * 2:
                    break
        finally:
            TTS_ACTIVE_REQUESTS.dec()
            elapsed = time.perf_counter() - t_start
            audio_s = model_bytes / 2 / sample_rate if sample_rate else 0.0
            TTS_GENERATION_SECONDS.observe(elapsed)
            if audio_s > 0:
                TTS_AUDIO_DURATION_SECONDS.observe(audio_s)
                TTS_RTF.observe(elapsed / audio_s)
            TTS_REQUESTS_TOTAL.labels(
                endpoint="stream", voice=voice,
                status="cancelled" if cancel_event.is_set() else "success",
            ).inc()
            if cancel_event.is_set():
                with contextlib.suppress(Exception):
                    self._api.abort_request(request_id)

        if cancel_event.is_set():
            # The receive loop's `cancelled` ack is the turn's terminal frame;
            # an audio.done here would claim a chunk that did not complete.
            return 0.0

        async with ws_lock:
            await websocket.send_json({
                "type": "audio.done",
                "sentence_index": sentence_index,
                "sample_rate": sample_rate,
                "chunk_count": chunk_count,
            })

        if config.get("timestamp_type") == "word":
            await self._send_timestamps(
                websocket, ws_lock, request_id, sentence_index, turn_offset_s,
                emitter,
            )
        return model_bytes / 2 / sample_rate if sample_rate else 0.0

    async def _send_timestamps(
        self, websocket: WebSocket, ws_lock: asyncio.Lock, request_id: str,
        sentence_index: int, turn_offset_s: float,
        emitter: "_PartialTimestampEmitter | None" = None,
    ) -> None:
        """Emit this chunk's ``timestamps`` frame, after its ``audio.done``.

        Times arrive from the aligner relative to the chunk's own audio and are
        shifted to TURN-relative here: t=0 is the turn's first ``audio.start``,
        and the inter-chunk silence the server inserted is part of the offset.
        A client concatenating the binary frames it received can therefore seek
        to a word by its timestamp without tracking chunk boundaries itself.

        Silent on failure. A missing timestamps frame costs a caller word
        timing; an exception here would cost them the rest of the session.
        """
        try:
            from mstar.model.qwen3_tts.temporal_alignment.registry import get_registry

            alignment = get_registry().pop_word_alignment(request_id)
        except Exception:  # noqa: BLE001 — a sidecar must not kill the session
            return
        if alignment is None or not alignment.words:
            return
        words = list(alignment.words)
        starts = [round(v + turn_offset_s, 4) for v in alignment.word_start_time_seconds]
        ends = [round(v + turn_offset_s, 4) for v in alignment.word_end_time_seconds]
        if emitter is not None:
            words, starts, ends = emitter.filter_tail(words, starts, ends)
            if not words:
                # Every word already went out incrementally. An empty frame
                # between audio.done and the next audio.start helps nobody.
                return
        payload = {
            "type": "timestamps",
            "sentence_index": sentence_index,
            "word_alignment": {
                "words": words,
                "word_start_time_seconds": starts,
                "word_end_time_seconds": ends,
            },
        }
        with contextlib.suppress(Exception):
            async with ws_lock:
                await websocket.send_json(payload)
