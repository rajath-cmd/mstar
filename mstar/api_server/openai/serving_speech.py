"""/v1/audio/speech handler (text-to-speech).

Non-streaming returns the full audio as a container blob (WAV by default).
Streaming returns a single open-ended WAV response (header + PCM16 frames) as
the audio is produced.
"""

from __future__ import annotations

import asyncio
import base64

from fastapi.responses import JSONResponse, Response, StreamingResponse

from mstar.api_server import media_io
from mstar.api_server.openai._util import rid


def _sample_rate(api) -> int:
    return api.model.get_output_sample_rate("audio") if api.model is not None else 24000


def _word_alignment(request_id: str) -> dict | None:
    """Collect this request's word alignment from the engine worker.

    The worker process computes the alignment in ``check_stop`` and publishes
    it to a disk drop; this process can only see that drop, so the read polls
    briefly to cover the gap between the last audio chunk reaching us and the
    worker finishing its Viterbi walk.

    ``None`` means no words -- the checkpoint ships no ``pointer_head.pt``, the
    request never asked for timestamps, or capture produced nothing. All three
    are answered the same way vllm-omni answers them: a null ``timestamp_info``
    rather than an error.
    """
    try:
        from mstar.model.qwen3_tts.temporal_alignment.registry import get_registry

        alignment = get_registry().pop_word_alignment(request_id)
    except Exception:  # noqa: BLE001 — timestamps must never fail the audio
        return None
    if alignment is None or not alignment.words:
        return None
    return {
        "word_alignment": {
            "words": list(alignment.words),
            "word_start_time_seconds": [round(v, 4) for v in alignment.word_start_time_seconds],
            "word_end_time_seconds": [round(v, 4) for v in alignment.word_end_time_seconds],
        }
    }


def _envelope(
    audio_bytes: bytes,
    pcm_len: int,
    fmt: str,
    sample_rate: int,
    index: int | None = None,
    request_id: str | None = None,
) -> dict:
    """The JSON body ``timestamp_type='word'`` switches the response to.

    Shape is frozen to vllm-omni's, so one client code path parses either
    server. ``timestamp_info`` is null when no words were produced.
    """
    body = {
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
        "format": fmt,
        "sample_rate": sample_rate,
        "duration_seconds": round(pcm_len / 2 / sample_rate, 3) if sample_rate else None,
        "timestamp_info": _word_alignment(request_id) if request_id else None,
    }
    if index is not None:
        body = {"index": index, **body}
    return body


async def _synthesize(api, adapter, req, text: str, request_id: str, raw_request) -> tuple[bytes, int, str, int]:
    """One scalar synthesis. Returns (container_bytes, pcm_len, fmt, sample_rate)."""
    single = req if isinstance(req.input, str) else req.model_copy(update={"input": text})
    args = adapter.speech_to_request(single, api.upload_dir)
    sample_rate = _sample_rate(api)
    fmt = (req.response_format or "wav").lower()
    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        streaming=False,
        request_id=request_id,
    )
    chunks = await api.collect_results(request_id, raw_request)
    pcm = b"".join(c.data for c in chunks if c.modality == "audio")
    audio_bytes, _mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)
    return audio_bytes, len(pcm), fmt, sample_rate


async def create_speech(api, model_name, adapter, req, raw_request=None):  # noqa: ARG001
    # A list input fans out into one index-aligned result per item.
    if isinstance(req.input, list):
        async def one(index: int, text: str) -> dict:
            item_id = rid(f"speech-batch-{index}")
            audio_bytes, pcm_len, fmt, sr = await _synthesize(
                api, adapter, req, text, item_id, raw_request
            )
            return _envelope(
                audio_bytes, pcm_len, fmt, sr, index=index, request_id=item_id
            )

        results = await asyncio.gather(*[one(i, t) for i, t in enumerate(req.input)])
        return JSONResponse({"results": list(results)})

    args = adapter.speech_to_request(req, api.upload_dir)
    request_id = rid("speech")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    fmt = (req.response_format or "wav").lower()

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        streaming=bool(req.stream),
        request_id=request_id,
    )

    if req.stream:
        return StreamingResponse(
            _stream_wav(api, request_id, sample_rate),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    chunks = await api.collect_results(request_id, raw_request)
    pcm = b"".join(c.data for c in chunks if c.modality == "audio")
    audio_bytes, mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)

    # Word timestamps switch the response from raw container bytes to a JSON
    # envelope. Without them the body stays bytes, which is the legacy shape.
    if getattr(req, "timestamp_type", None) == "word":
        return JSONResponse(
            _envelope(audio_bytes, len(pcm), fmt, sample_rate, request_id=request_id)
        )
    return Response(content=audio_bytes, media_type=mime)


async def _stream_wav(api, request_id, sample_rate):
    yield media_io.wav_stream_header(sample_rate)
    async for c in api.iter_result_chunks(request_id):
        if c.modality == "audio" and c.data:
            yield c.data
