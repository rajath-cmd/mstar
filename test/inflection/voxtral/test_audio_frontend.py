"""Front end: prompt construction and mel extraction.

The mel is asserted structurally here. Its bit-identity against the HF
reference processor was verified out of band (max abs diff 0.000e+00); pinning
that in CI would mean installing a transformers newer than the one M* pins,
which is the dependency risk the port exists to avoid.
"""

import numpy as np
import pytest

from mstar.model.voxtral_rt.audio_frontend import log_mel, prepare


def _tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_log_mel_shape_and_range(config):
    mel = log_mel(_tone(1.0), config)
    assert mel.shape[0] == config.audio.num_mel_bins
    # 1 s at a 160-sample hop, centred, minus the reflection tail.
    assert mel.shape[1] == 100
    # The floor is global_log_mel_max - 8, rescaled by (x + 4) / 4.
    assert float(mel.min()) >= (config.global_log_mel_max - 8.0 + 4.0) / 4.0 - 1e-6


def test_log_mel_floor_is_absolute_not_per_utterance(config):
    """A quiet clip must not be renormalised louder.

    Whisper takes the dynamic-range ceiling from the utterance's own maximum.
    Voxtral uses a fixed one, which is what lets a streaming chunk be
    normalised identically to the whole file. Scaling the input down must
    therefore LOWER the features, not leave them unchanged.
    """
    loud = log_mel(_tone(1.0), config)
    quiet = log_mel(_tone(1.0) * 0.01, config)
    assert float(quiet.mean()) < float(loud.mean()) - 0.1


def test_prompt_is_bos_then_streaming_pad(config, checkpoint):
    """The prompt carries no text -- only positions."""
    out = prepare(_tone(3.0), 16000, checkpoint, config)
    ids = out.input_ids.tolist()
    assert ids[0] == 1, "first token should be <s>"
    assert set(ids[1:]) == {32}, "the rest should all be [STREAMING_PAD]"


def test_token_budget_is_the_audio_length(config, checkpoint):
    """One decode step per 80 ms, after the encoder's padding."""
    out = prepare(_tone(4.0), 16000, checkpoint, config)
    assert out.n_audio_tokens == out.input_features.shape[-1] // config.audio_length_per_tok
    # Padding only ever ADDS frames, so the budget covers the real audio.
    assert out.n_audio_tokens >= 4.0 * config.frame_rate_hz


def test_wrong_sample_rate_is_rejected_not_reinterpreted(config, checkpoint):
    """Silently accepting 24 kHz would shift every timestamp by 1.5x."""
    with pytest.raises(ValueError, match="16000 Hz"):
        prepare(_tone(1.0, sr=24000), 24000, checkpoint, config)


def test_stereo_is_mixed_down(config, checkpoint):
    mono = _tone(2.0)
    stereo = np.stack([mono, mono], axis=-1)
    assert prepare(stereo, 16000, checkpoint, config).n_audio_tokens == \
        prepare(mono, 16000, checkpoint, config).n_audio_tokens
