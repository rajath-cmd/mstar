"""``/v1/audio/voices`` — the speaker list a client picks ``voice`` from.

Shape is frozen to vllm-omni's: ``voices`` are the checkpoint's built-in
speakers, ``uploaded_voices`` are runtime voice clones (always empty until M*
grows an upload path; the field is present so a client can read it either way).
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


def _speakers(api) -> list[str]:
    """Read the loaded checkpoint's speakers, lowercased and sorted.

    Deliberately the SAME source the preprocessor validates against —
    ``config.talker.spk_id``, whose keys it lists verbatim when it rejects an
    unknown speaker (``qwen3_tts_model.py``). Reading anything else would let
    this endpoint advertise a voice that /v1/audio/speech then refuses, which is
    worse than advertising none.

    Never hardcoded: the set is per-checkpoint. The stock 0.6B ships "vivian";
    our 14-voice fine-tunes ship alexandra..steven.
    """
    config = getattr(getattr(api, "model", None), "config", None)
    spk_id = getattr(getattr(config, "talker", None), "spk_id", None) or {}
    return sorted({str(name).lower() for name in spk_id})


async def list_voices(api) -> JSONResponse:
    return JSONResponse({"voices": _speakers(api), "uploaded_voices": []})
