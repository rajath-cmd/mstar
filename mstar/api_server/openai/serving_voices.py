"""``/v1/audio/voices`` — the speaker list a client picks ``voice`` from.

Shape is frozen to vllm-omni's: ``voices`` are the checkpoint's built-in
speakers, ``uploaded_voices`` are runtime voice clones (always empty until M*
grows an upload path; the field is present so a client can read it either way).
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


def _speakers(api) -> list[str]:
    """Read the loaded checkpoint's speakers, lowercased and sorted.

    Never hardcoded: the set is per-checkpoint (the stock 0.6B ships different
    names from our fine-tunes), and the Qwen3-TTS preprocessor already rejects
    an unknown speaker with the supported list, so these must be the same names.
    """
    model = getattr(api, "model", None)
    if model is None:
        return []
    config = getattr(model, "config", None)
    for owner in (getattr(config, "talker", None), config):
        raw = getattr(owner, "speakers", None)
        if raw:
            return sorted({str(s).lower() for s in raw})
    return []


async def list_voices(api) -> JSONResponse:
    return JSONResponse({"voices": _speakers(api), "uploaded_voices": []})
