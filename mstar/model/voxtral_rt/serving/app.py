"""OpenAI-compatible transcription server for Voxtral-Realtime on M*.

Scope, stated plainly: this is the BRING-UP server. It serves one request at a
time against the verified model core in ``mstar.model.voxtral_rt.model``. It is
not yet on M*'s Walk Graph engine, so it has no continuous batching, no paged
attention and no CUDA graphs -- the three things that gave Qwen3-TTS its
advantage under load. Expect single-stream latency to be competitive and
concurrency to be flat.

The API surface is fixed now so that moving onto the engine later is an
internal change: ``POST /v1/audio/transcriptions``, OpenAI's shape, matching
what the vllm-realtime fork serves.
"""

import asyncio
import io
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from mstar.model.voxtral_rt.audio_frontend import prepare
from mstar.model.voxtral_rt.model import VoxtralRealtime

logger = logging.getLogger(__name__)

_RESPONSE_FORMATS = {"json", "text", "verbose_json"}


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resample.

    Deliberately simple. The model only ever sees 16 kHz, and a client sending
    another rate is a convenience case, not the hot path -- but accepting the
    audio at the wrong rate silently would shift every timestamp, so it is
    resampled rather than reinterpreted.
    """
    if sr_in == sr_out:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * sr_out / sr_in))
    x_in = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, audio).astype(np.float32)


def read_audio(data: bytes, target_sr: int) -> tuple[np.ndarray, float]:
    """Decode a container to mono float32 at ``target_sr``."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    duration = len(audio) / sr
    return _resample(audio, sr, target_sr), duration


def build_app(model_dir: str, device: str = "cuda", dtype=torch.bfloat16):
    app = FastAPI(title="Voxtral-Realtime on M*")
    state: dict = {}
    # One GPU, one model, no batching: serialise so concurrent callers queue
    # instead of racing each other's CUDA work.
    lock = asyncio.Lock()

    @app.on_event("startup")
    async def _load() -> None:
        t0 = time.perf_counter()
        logger.info("loading Voxtral-Realtime from %s", model_dir)
        state["model"] = VoxtralRealtime.from_pretrained(
            model_dir, device=device, dtype=dtype
        )
        state["ready"] = True
        logger.info("model ready in %.1fs", time.perf_counter() - t0)

    @app.get("/health")
    async def health():
        if not state.get("ready"):
            raise HTTPException(status_code=503, detail="model still loading")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": Path(model_dir).name, "object": "model", "owned_by": "mstar"}],
        }

    # NOTE: this module must NOT use `from __future__ import annotations`.
    # It turns every annotation into a string, and FastAPI resolves a route's
    # annotations against the MODULE namespace -- so `UploadFile` on a route
    # defined inside this factory becomes an unresolvable ForwardRef and every
    # multipart request fails with a pydantic "not fully defined" 500. That is
    # why the FastAPI imports above are module-level.
    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        response_format: str = Form(default="json"),
        language: str = Form(default=""),
    ):
        del model, language  # accepted for OpenAI compatibility
        if not state.get("ready"):
            raise HTTPException(status_code=503, detail="model still loading")
        if response_format not in _RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"response_format must be one of {sorted(_RESPONSE_FORMATS)}",
            )
        m = state["model"]
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty audio file")
        try:
            audio, source_seconds = read_audio(raw, m.config.sampling_rate)
        except Exception as exc:  # noqa: BLE001 — a bad upload is a 400, not a 500
            raise HTTPException(
                status_code=400, detail=f"could not decode audio: {exc}"
            ) from exc

        async with lock:
            result = await asyncio.to_thread(_run, m, audio, model_dir)

        if response_format == "text":
            return PlainTextResponse(result.text)
        body = {"text": result.text}
        if response_format == "verbose_json":
            body = {
                "task": "transcribe",
                # The checkpoint has no language head, and unlike the
                # vllm-realtime fork we do not guess one from the output
                # script. An empty string is honest; a wrong ISO code is not.
                "language": "",
                "duration": round(source_seconds, 2),
                "text": result.text,
                "segments": None,
                "words": None,
            }
        return JSONResponse(body)

    def _run(m, audio: np.ndarray, model_dir: str):
        inputs = prepare(audio, m.config.sampling_rate, model_dir, m.config)
        return m.transcribe(inputs.input_ids, inputs.input_features)

    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default=os.environ.get("VOXTRAL_MODEL_PATH", ""),
        help="checkpoint directory (or $VOXTRAL_MODEL_PATH)",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("set --model or VOXTRAL_MODEL_PATH")
    for required in ("config.json", "model.safetensors", "tekken.json"):
        if not (Path(args.model) / required).is_file():
            raise SystemExit(f"missing {required} in {args.model}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(build_app(args.model, device=args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
