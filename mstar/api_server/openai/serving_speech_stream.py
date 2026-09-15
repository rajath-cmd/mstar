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

Not yet implemented here, and deliberately visible as absence rather than as a
wrong answer: word timestamps (M* has no alignment head — Phase 3). No
``timestamps`` frame is emitted, which is exactly what vllm-omni does for a
checkpoint shipping no ``pointer_head.pt``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os

from fastapi import WebSocket, WebSocketDisconnect

from mstar.api_server.openai._util import rid
from mstar.api_server.openai.text_chunker import create_chunker

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
                     "ref_audio", "ref_text", "x_vector_only_mode"):
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
                        first_chunk_of_turn = True
                        turn_audio_s = 0.0
                        sentence_index = 0
                        continue

                    if cancel_event.is_set():
                        continue

                    if item is _INPUT_DONE:
                        await send_json({"type": "session.done", "total_sentences": sentence_index})
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
                await send_error(f"Internal error: {e}")
        finally:
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
        cancel_event: asyncio.Event,
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

        try:
            async for chunk in self._api.iter_result_chunks(request_id):
                if cancel_event.is_set():
                    break
                if chunk.modality != "audio" or not chunk.data:
                    continue
                async with ws_lock:
                    await websocket.send_bytes(chunk.data)
                model_bytes += len(chunk.data)
                chunk_count += 1
                # Runaway early-stop. Without it a repetition loop streams
                # unbounded audio; vllm-omni bounds the same way.
                if cap_s is not None and model_bytes >= cap_s * sample_rate * 2:
                    break
        finally:
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
        return model_bytes / 2 / sample_rate if sample_rate else 0.0
