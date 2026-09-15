"""The cadence arithmetic, which every timing claim downstream rests on."""

from mstar.model.voxtral_rt.config import VoxtralRealtimeConfig


def test_config_loads_the_shipped_shapes(config):
    assert config.audio.num_hidden_layers == 32
    assert config.audio.hidden_size == 1280
    assert config.audio.sliding_window == 750
    assert config.text.num_hidden_layers == 26
    assert config.text.hidden_size == 3072
    # GQA: 32 query heads over 8 KV heads.
    assert config.text.num_attention_heads == 32
    assert config.text.num_key_value_heads == 8
    assert config.text.tie_word_embeddings is True


def test_frame_rate_is_exactly_12_5_hz(config):
    """Not 12, not 12.6. The transcript length IS the audio length.

    16000 / (160 * 8) = 12.5. If this drifts, every token lands at the wrong
    time and the decode-step budget stops matching the audio.
    """
    assert config.frame_rate_hz == 12.5
    assert config.audio_length_per_tok == 8
    assert config.downsample_factor == 4


def test_defaults_match_the_checkpoint(config):
    fresh = VoxtralRealtimeConfig()
    assert fresh.frame_rate_hz == config.frame_rate_hz
    assert fresh.audio.num_mel_bins == config.audio.num_mel_bins == 128
