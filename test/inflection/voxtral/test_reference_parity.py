"""Token-level parity against the HF reference implementation.

These are the tests that decide whether the port is correct. They compare
GENERATED TOKEN IDS, not transcripts, because text comparison hides the failure
mode that actually occurred during bring-up: a corrupt time-conditioning buffer
made the model emit the right words two frames early, which still decoded to
identical text on six of eight clips while being measurably wrong.

Reference output was recorded from
``transformers.VoxtralRealtimeForConditionalGeneration`` at bfloat16 on a B200.
Regenerate with ``scripts/inflection/voxtral_record_reference.py`` if the
checkpoint changes.
"""

import json
from pathlib import Path

import pytest
import soundfile as sf

REFERENCE = Path(__file__).parent / "reference"


def _cases():
    data = json.loads((REFERENCE / "hf_reference.json").read_text())
    return [(name, payload) for name, payload in sorted(data.items())]


@pytest.mark.parametrize(("name", "expected"), _cases())
def test_tokens_match_the_reference(model, checkpoint, name, expected):
    from mstar.model.voxtral_rt.audio_frontend import prepare

    audio, sr = sf.read(REFERENCE / "audio" / name)
    inputs = prepare(audio, sr, checkpoint, model.config)
    result = model.transcribe(inputs.input_ids, inputs.input_features)

    assert result.token_ids == expected["new_token_ids"], (
        f"{name}: token sequence diverged from the reference. Compare where: a "
        f"pure SHIFT (same non-pad tokens, different pad positions) points at "
        f"time conditioning; different non-pad tokens point at weights or "
        f"attention."
    )
    assert result.text.strip() == expected["text"].strip()


@pytest.mark.parametrize(("name", "expected"), _cases())
def test_decode_budget_equals_the_audio(model, checkpoint, name, expected):
    """Transcription length is set by the audio, never by a sampling cap."""
    from mstar.model.voxtral_rt.audio_frontend import prepare

    audio, sr = sf.read(REFERENCE / "audio" / name)
    inputs = prepare(audio, sr, checkpoint, model.config)
    result = model.transcribe(inputs.input_ids, inputs.input_features)

    n_prompt = inputs.input_ids.shape[0]
    assert len(result.token_ids) == inputs.n_audio_tokens - n_prompt
    assert result.n_audio_tokens == inputs.n_audio_tokens


def test_time_conditioning_buffer_is_initialised(model):
    """Guards the bring-up bug that no weight check could catch.

    ``inv_freq`` is a NON-PERSISTENT buffer, so no checkpoint key restores it,
    and loading materialises the module with ``to_empty()`` -- which allocates
    buffer storage without initialising it. The result was a model that loaded
    cleanly, reported no missing weights, and transcribed at the wrong times.
    """
    import math

    import torch

    inv = model.time_embedding.inv_freq
    assert torch.isfinite(inv).all(), "inv_freq contains uninitialised memory"
    assert inv.shape[0] == model.config.text.hidden_size // 2
    # First entry is exp(0) == 1 for any theta; the last is theta^-((n-1)/n).
    assert math.isclose(float(inv[0]), 1.0, rel_tol=1e-3)
    assert float(inv[-1]) < 1e-3
    assert float(inv.min()) > 0.0


def test_audio_embeddings_are_one_per_80ms(model, checkpoint):
    from mstar.model.voxtral_rt.audio_frontend import prepare

    audio, sr = sf.read(REFERENCE / "audio" / "en_0.wav")
    inputs = prepare(audio, sr, checkpoint, model.config)
    embeds = model.encode_audio(inputs.input_features)
    assert embeds.shape[1] == inputs.n_audio_tokens
    assert embeds.shape[2] == model.config.text.hidden_size
