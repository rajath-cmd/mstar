"""Waveform -> (prompt tokens, log-mel) for Voxtral-Realtime.

Two jobs, and the split matters:

* **Tokenisation and audio padding** come from ``mistral_common``, the
  reference tokeniser the checkpoint was trained with. Reimplementing the pad
  arithmetic here would be a parity risk for no benefit -- the padding sets the
  alignment between mel frames and text positions, and being one token out
  shifts the entire transcript against the audio.
* **Mel extraction** is ported, because the reference lives in a transformers
  version newer than the one M* pins, and upgrading transformers to obtain one
  feature extractor would put the Qwen3-TTS stack at risk. It is ~15 lines of
  STFT and is verified against the reference bit-for-bit in
  ``test/inflection/voxtral/test_audio_frontend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers.audio_utils import mel_filter_bank

from mstar.model.voxtral_rt.config import VoxtralRealtimeConfig


@dataclass
class VoxtralInputs:
    """One utterance, ready for the graph."""

    input_ids: torch.Tensor      # [n_prompt]
    input_features: torch.Tensor  # [n_mel_bins, n_frames]
    num_delay_tokens: int
    n_audio_tokens: int
    audio_seconds: float


@lru_cache(maxsize=4)
def _mel_filters(n_fft: int, n_mels: int, sampling_rate: int) -> np.ndarray:
    return mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=sampling_rate / 2,
        sampling_rate=sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    )


def log_mel(
    waveform: np.ndarray | torch.Tensor,
    cfg: VoxtralRealtimeConfig,
    *,
    center: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Log-mel spectrogram, ``[n_mel_bins, n_frames]``.

    Whisper's normalisation with one deliberate change: the dynamic-range floor
    is taken from a FIXED ``global_log_mel_max`` rather than this utterance's
    own maximum. That is what lets a streaming chunk be normalised identically
    to the whole file -- a per-utterance max would make the same audio produce
    different features depending on how it was cut.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.to(device=device, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(cfg.n_fft, device=waveform.device)
    stft = torch.stft(
        waveform, cfg.n_fft, cfg.hop_length,
        window=window, return_complex=True, center=center,
    )
    # Drop the final frame: with center=True it is the reflection tail and
    # carries no new audio.
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(
        _mel_filters(cfg.n_fft, cfg.audio.num_mel_bins, cfg.sampling_rate)
    ).to(waveform.device, torch.float32)
    mel = filters.T @ magnitudes

    log_spec = torch.clamp(mel, min=1e-10).log10()
    ceiling = torch.tensor(
        cfg.global_log_mel_max, device=log_spec.device, dtype=log_spec.dtype
    )
    log_spec = torch.maximum(log_spec, ceiling - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.squeeze(0).cpu()


@lru_cache(maxsize=2)
def _tokenizer(model_dir: str):
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tekken = Path(model_dir) / "tekken.json"
    if not tekken.is_file():
        raise FileNotFoundError(
            f"{tekken} not found. Voxtral-Realtime needs the Tekken tokenizer "
            "shipped with the checkpoint; without it the prompt cannot be built "
            "and output token ids cannot be decoded."
        )
    return MistralTokenizer.from_file(str(tekken))


def prepare(
    waveform: np.ndarray,
    sampling_rate: int,
    model_dir: str,
    cfg: VoxtralRealtimeConfig,
) -> VoxtralInputs:
    """Build the offline-transcription prompt and features for one utterance.

    The returned ``input_ids`` is ``<s>`` followed by ``[STREAMING_PAD]``: the
    prompt carries no text, only positions. Voxtral-Realtime is synchronous --
    every position is one 80 ms audio frame, and the transcript is whatever the
    model emits at those positions instead of a pad. That is why the total
    sequence length is decided by the audio, not by a sampling parameter, and
    why ``n_audio_tokens`` is an exact decode-step budget rather than a cap.
    """
    from mistral_common.protocol.transcription.request import (
        StreamingMode,
        TranscriptionRequest,
    )
    from mistral_common.tokens.tokenizers.audio import Audio

    if sampling_rate != cfg.sampling_rate:
        raise ValueError(
            f"Voxtral-Realtime expects {cfg.sampling_rate} Hz audio, got "
            f"{sampling_rate}. Resample before calling; the mel front end has "
            "no resampler and mismatched rates silently shift every timestamp."
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:  # mixdown; the tower is single channel
        waveform = waveform.mean(axis=-1)

    tk = _tokenizer(model_dir)
    obj = Audio(audio_array=waveform, sampling_rate=sampling_rate, format="wav")
    request = TranscriptionRequest(
        audio=obj.to_base64("wav"), streaming=StreamingMode.OFFLINE, language=None
    )
    encoded = tk.encode_transcription(request)
    padded = np.asarray(encoded.audios[0].audio_array, dtype=np.float32)

    features = log_mel(padded, cfg, center=True)
    n_audio_tokens = features.shape[-1] // cfg.audio_length_per_tok
    return VoxtralInputs(
        input_ids=torch.tensor(encoded.tokens, dtype=torch.long),
        input_features=features,
        num_delay_tokens=cfg.default_num_delay_tokens,
        n_audio_tokens=n_audio_tokens,
        audio_seconds=len(waveform) / sampling_rate,
    )


def decode(model_dir: str, token_ids: list[int]) -> str:
    """Token ids -> transcript, special tokens stripped."""
    tk = _tokenizer(model_dir)
    return tk.decode(list(token_ids))
