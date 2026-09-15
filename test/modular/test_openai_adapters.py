"""Unit tests for the OpenAI request adapters (pydantic + numpy; no fastapi)."""

import base64
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("numpy")

from mstar.api_server.openai import adapters  # noqa: E402
from mstar.api_server.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ImageGenerationRequest,
    SpeechRequest,
)


def test_bagel_chat_maps_max_tokens(tmp_path):
    req = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "hi"}], max_tokens=32)
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert sa.text == "hi"
    assert sa.output_modalities == ["text"]
    assert sa.model_kwargs["max_output_tokens"] == 32


def test_bagel_chat_decodes_image_input(tmp_path):
    raw = b"\x89PNG-data"
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    req = ChatCompletionRequest(
        model="bagel",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    )
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert "image" in sa.file_paths
    assert Path(sa.file_paths["image"][0]).read_bytes() == raw
    assert "image" in sa.input_modalities and "text" in sa.input_modalities


def test_qwen3_chat_audio_and_sampling(tmp_path):
    req = ChatCompletionRequest(
        model="qwen3_omni",
        messages=[{"role": "user", "content": "say hi"}],
        modalities=["text", "audio"],
        audio={"voice": "Ethan"},
        temperature=0.3,
        top_p=0.8,
    )
    sa = adapters.Qwen3OmniAdapter().chat_to_request(req, tmp_path)
    assert sa.output_modalities == ["text", "audio"]
    assert sa.model_kwargs["thinker_temperature"] == 0.3
    assert sa.model_kwargs["thinker_top_p"] == 0.8
    assert sa.model_kwargs["voice"] == "Ethan"


def test_orpheus_speech_maps_voice_and_sampling(tmp_path):
    req = SpeechRequest(model="orpheus", input="hi", voice="tara", temperature=0.7, top_p=0.5, seed=11)
    sa = adapters.OrpheusAdapter().speech_to_request(req, tmp_path)
    assert sa.output_modalities == ["audio"]
    assert sa.model_kwargs == {"voice": "tara", "temperature": 0.7, "top_p": 0.5, "seed": 11}


def test_qwen3_speech_maps_talker_sampling(tmp_path):
    # talker_top_p was previously missing; the shared sampling helper adds it.
    req = SpeechRequest(model="qwen3_omni", input="hi", voice="Ethan", temperature=0.5, top_p=0.7, seed=123)
    sa = adapters.Qwen3OmniAdapter().speech_to_request(req, tmp_path)
    assert sa.output_modalities == ["text", "audio"]
    assert sa.model_kwargs["talker_temperature"] == 0.5
    assert sa.model_kwargs["talker_top_p"] == 0.7
    assert sa.model_kwargs["seed"] == 123
    assert sa.model_kwargs["voice"] == "Ethan"


def test_chat_and_image_honor_seed(tmp_path):
    chat = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "x"}], seed=7)
    assert adapters.BagelAdapter().chat_to_request(chat, tmp_path).model_kwargs["seed"] == 7
    img = ImageGenerationRequest(model="bagel", prompt="a cat", seed=9)
    assert adapters.BagelAdapter().image_to_request(img, tmp_path).model_kwargs["seed"] == 9


def test_extra_body_non_standard_knobs_pass_through(tmp_path):
    # top_k / repetition_penalty aren't OpenAI fields → flow via extra_body verbatim.
    req = ChatCompletionRequest(
        model="qwen3_omni", messages=[{"role": "user", "content": "x"}],
        talker_top_k=50, talker_repetition_penalty=1.05,
    )
    mk = adapters.Qwen3OmniAdapter().chat_to_request(req, tmp_path).model_kwargs
    assert mk["talker_top_k"] == 50 and mk["talker_repetition_penalty"] == 1.05


def test_bagel_image(tmp_path):
    req = ImageGenerationRequest(model="bagel", prompt="a cat")
    sa = adapters.BagelAdapter().image_to_request(req, tmp_path)
    assert sa.text == "a cat" and sa.output_modalities == ["image"]


def test_extra_body_passthrough(tmp_path):
    # Unknown fields (OpenAI client extra_body) flow through as model_kwargs.
    req = ChatCompletionRequest(model="bagel", messages=[{"role": "user", "content": "x"}], think_mode=True)
    sa = adapters.BagelAdapter().chat_to_request(req, tmp_path)
    assert sa.model_kwargs.get("think_mode") is True


def test_qwen3_chat_audio_url_input(tmp_path):
    raw = b"RIFFfakewavbytes"
    url = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    req = ChatCompletionRequest(
        model="qwen3_omni",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "audio_url", "audio_url": {"url": url}},
        ]}],
    )
    sa = adapters.Qwen3OmniAdapter().chat_to_request(req, tmp_path)
    assert "audio" in sa.file_paths
    assert Path(sa.file_paths["audio"][0]).read_bytes() == raw
    assert "audio" in sa.input_modalities


def test_bagel_image_edit(tmp_path):
    image_path = str(tmp_path / "in.png")
    sa = adapters.BagelAdapter().image_edit_to_request("make it neon", image_path, {"cfg_img_scale": 2.0})
    assert sa.text == "make it neon"
    assert sa.file_paths == {"image": [image_path]}
    assert sa.input_modalities == ["image", "text"]
    assert sa.output_modalities == ["image"]
    assert sa.model_kwargs == {"cfg_img_scale": 2.0}


def test_registry():
    assert {"bagel", "qwen3_omni", "orpheus"} <= set(adapters.ADAPTER_REGISTRY)
    assert adapters.get_adapter("pi05") is None
    assert adapters.get_adapter("bagel").supports_chat


# --- Qwen3-TTS ---------------------------------------------------------------
# The request schema and adapter mapping are frozen to vllm-omni's so an
# M*-served Qwen3-TTS is a drop-in replacement. These are the CPU-checkable half
# of the parity contract; the live differential suite is test/parity/.


def test_qwen3_tts_adapter_is_registered():
    """qwen3_tts was absent from ADAPTER_REGISTRY, so /v1/audio/speech 404'd for
    it and the base speech_to_request raised NotImplementedError."""
    assert adapters.get_adapter("qwen3_tts") is not None


def test_speech_request_accepts_the_full_qwen3_tts_surface():
    req = SpeechRequest(
        input="hi", model="qwen3_tts", voice="alexandra",
        instructions="Speak calmly.", response_format="wav", speed=1.0,
        stream_format="audio", task_type="CustomVoice", language="Auto",
        ref_audio=None, ref_text=None, x_vector_only_mode=False,
        max_new_tokens=512, stream=False, temperature=0.7, top_k=30,
        top_p=0.95, repetition_penalty=1.05, timestamp_type="word",
    )
    assert req.top_k == 30
    assert req.repetition_penalty == 1.05
    assert req.timestamp_type == "word"
    assert req.task_type == "CustomVoice"


def test_speech_request_accepts_a_batch_input():
    assert SpeechRequest(input=["a", "b"], voice="alexandra").input == ["a", "b"]


def test_qwen3_tts_speech_maps_the_full_surface(tmp_path):
    req = SpeechRequest(
        input="Hello there.", model="qwen3_tts", voice="alexandra",
        language="Auto", instructions="Speak calmly.", task_type="CustomVoice",
        temperature=0.7, top_k=30, top_p=0.95, repetition_penalty=1.05,
        max_new_tokens=512, timestamp_type="word",
    )
    sa = adapters.get_adapter("qwen3_tts").speech_to_request(req, tmp_path)
    assert sa.text == "Hello there."
    assert sa.input_modalities == ["text"]
    assert sa.output_modalities == ["audio"]
    mk = sa.model_kwargs
    assert mk["voice"] == "alexandra"
    assert mk["language"] == "Auto"
    assert mk["instructions"] == "Speak calmly."
    assert mk["task_type"] == "CustomVoice"
    assert mk["top_k"] == 30
    assert mk["repetition_penalty"] == 1.05
    assert mk["timestamp_type"] == "word"
    # Talker sampling is namespaced, matching the Qwen3-Omni convention.
    assert mk["talker_temperature"] == 0.7
    assert mk["talker_top_p"] == 0.95


def test_qwen3_tts_omits_unset_fields(tmp_path):
    """An unset field must not reach model_kwargs as None: the Walk Graph treats
    a present key as an override, so a null clobbers the checkpoint default."""
    req = SpeechRequest(input="Hi.", voice="alexandra")
    mk = adapters.get_adapter("qwen3_tts").speech_to_request(req, tmp_path).model_kwargs
    for absent in ("instructions", "language", "top_k", "repetition_penalty", "timestamp_type"):
        assert absent not in mk, f"{absent} leaked into model_kwargs as None"
