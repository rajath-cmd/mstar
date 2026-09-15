"""Config and resource labels for Voxtral-Realtime on M*.

The checkpoint ships BOTH a HF-style ``config.json`` and a Mistral-style
``params.json``; they describe the same weights. We read ``config.json``
because it is the one the HF reference implementation loads, so a field we
disagree with is a bug on our side rather than a question of which file wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Resource labels. One attention/KV stream for the audio tower and one for the
# text decoder: they have different widths, different head counts and different
# sliding windows, so they cannot share a cache.
AUDIO_ATTN = "audio_attn"
AUDIO_KV = "audio_kv_cache"
AUDIO_POS = "audio_pos"
TEXT_ATTN = "text_attn"
TEXT_KV = "text_kv_cache"
TEXT_POS = "text_pos"
SAMPLER = "sampler"

AUDIO_LABEL = "audio"
TEXT_LABEL = "text"


@dataclass
class VoxtralAudioConfig:
    """The 32-layer causal audio tower (Whisper-shaped, but RoPE + sliding)."""

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 64
    intermediate_size: int = 5120
    num_mel_bins: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 750


@dataclass
class VoxtralTextConfig:
    """The 26-layer Mistral decoder, with AdaRMSNorm time conditioning."""

    hidden_size: int = 3072
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    vocab_size: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 8192
    tie_word_embeddings: bool = True


@dataclass
class VoxtralRealtimeConfig:
    """Top-level model config.

    ``audio_length_per_tok`` (8 mel frames) and ``downsample_factor`` (4) fix
    the model's cadence: 8 mel frames at a 160-sample hop is 80 ms, so the
    decoder runs at exactly 12.5 Hz and emits exactly one token per 80 ms of
    audio. That is not a tuning knob -- the transcript length is the audio
    length, and every timing claim downstream rests on it.
    """

    audio: VoxtralAudioConfig = field(default_factory=VoxtralAudioConfig)
    text: VoxtralTextConfig = field(default_factory=VoxtralTextConfig)
    audio_length_per_tok: int = 8
    downsample_factor: int = 4
    default_num_delay_tokens: int = 3
    ada_rms_norm_t_cond_dim: int = 32
    projector_hidden_act: str = "gelu"

    # Feature extraction. Matches processor_config.json; the global max makes
    # the log-mel floor absolute rather than per-utterance, so a chunk of a
    # stream is normalised identically to the whole file.
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    win_length: int = 400
    global_log_mel_max: float = 1.5

    @property
    def frame_rate_hz(self) -> float:
        return self.sampling_rate / (self.hop_length * self.audio_length_per_tok)

    @property
    def encoder_frames_per_token(self) -> int:
        """Audio-tower output frames consumed per decoded text token.

        The embedder's stride-2 conv halves the mel rate, and the tower's
        output is then reshaped by ``downsample_factor``. Both together are
        what turn ``audio_length_per_tok`` mel frames into one text position.
        """
        return self.downsample_factor

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> VoxtralRealtimeConfig:
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        a, t = raw.get("audio_config", {}), raw.get("text_config", {})

        def rope(d: dict, default: float) -> float:
            return float((d.get("rope_parameters") or {}).get("rope_theta", default))

        cfg = cls(
            audio=VoxtralAudioConfig(
                hidden_size=a.get("hidden_size", 1280),
                num_hidden_layers=a.get("num_hidden_layers", 32),
                num_attention_heads=a.get("num_attention_heads", 32),
                num_key_value_heads=a.get("num_key_value_heads", 32),
                head_dim=a.get("head_dim", 64),
                intermediate_size=a.get("intermediate_size", 5120),
                num_mel_bins=a.get("num_mel_bins", 128),
                rms_norm_eps=a.get("rms_norm_eps", 1e-5),
                rope_theta=rope(a, 1_000_000.0),
                sliding_window=a.get("sliding_window", 750),
            ),
            text=VoxtralTextConfig(
                hidden_size=t.get("hidden_size", 3072),
                num_hidden_layers=t.get("num_hidden_layers", 26),
                num_attention_heads=t.get("num_attention_heads", 32),
                num_key_value_heads=t.get("num_key_value_heads", 8),
                head_dim=t.get("head_dim", 128),
                intermediate_size=t.get("intermediate_size", 9216),
                vocab_size=t.get("vocab_size", 131072),
                rms_norm_eps=t.get("rms_norm_eps", 1e-5),
                rope_theta=rope(t, 1_000_000.0),
                sliding_window=t.get("sliding_window", 8192),
                tie_word_embeddings=t.get("tie_word_embeddings", True),
            ),
            audio_length_per_tok=raw.get("audio_length_per_tok", 8),
            downsample_factor=raw.get("downsample_factor", 4),
            default_num_delay_tokens=raw.get("default_num_delay_tokens", 3),
            projector_hidden_act=raw.get("projector_hidden_act", "gelu"),
        )
        params = Path(model_dir) / "params.json"
        if params.is_file():
            p = json.loads(params.read_text())
            cfg.ada_rms_norm_t_cond_dim = int(p.get("ada_rms_norm_t_cond_dim", 32))
        return cfg
