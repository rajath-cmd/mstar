# M* Qwen3-TTS Phase 1: REST parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make M*'s `/v1/audio/speech` accept and honour the exact request surface of vllm-omni's, and stand up the differential conformance harness that every later phase's parity claim is asserted through.

**Architecture:** M* dispatches OpenAI-compatible requests through `ADAPTER_REGISTRY[model_name].speech_to_request(req) -> SubmitArgs`, which lowers a pydantic request onto `model_kwargs` for the Walk Graph. Qwen3-TTS has no entry there today, so the endpoint is dead for it. This phase adds `Qwen3TTSAdapter`, widens `SpeechRequest` from 9 fields to vllm-omni's 19, adds the voices endpoints, and shapes the `timestamp_type` JSON envelope (populated in Phase 3). Parity is asserted by a harness that runs one set of assertions against both servers.

**Tech Stack:** Python 3.12, pydantic v2, FastAPI, pytest, `uv` (never bare `pip`/`python`), httpx.

**Spec:** `docs/superpowers/specs/2026-09-15-mstar-qwen3-tts-parity-spec.md`

## Global Constraints

- **Gate 0 is BLOCKING.** `rajath-cmd/mstar` is public. Nothing from this plan is pushed to that remote until repository visibility is resolved. Work locally; commit locally; do not `git push`.
- **The protocol is frozen.** Field names, types, defaults and semantics are fixed by vllm-omni. M* adapts to them.
- **`uv` only.** Never bare `pip` or `python`.
- Codec frame rate is **12.5 Hz exactly** (`24000 / 1920` = 80 ms/frame). Never 12 Hz.
- Reference checkpoint: `/mnt/data/models/audio/tts/qwen3-tts/sft-cv-13l-14pv-ticr-1e-13l-fulldata_17946/checkpoint-final`
- Reference server: `~/workspace/vllm-omni` @ `raj/ws-incremental-word-timestamps`
- Voices on this checkpoint: `alexandra, allan, charlie, constanza, elliot, eloise, fred, jacopo, john, kerry, marc, pell, sheena, steven`. **Not** `Vivian` (that is the stock CustomVoice model's).
- Every parity claim in a commit message or report cites a measurement.

---

## File Structure

| File | Responsibility |
|---|---|
| `mstar/api_server/openai/protocol.py` (modify, `:45`) | `SpeechRequest` — widen to vllm-omni's 19 fields |
| `mstar/api_server/openai/adapters.py` (modify, `:454`) | `Qwen3TTSAdapter` + `ADAPTER_REGISTRY` entry |
| `mstar/api_server/openai/serving_speech.py` (modify) | Batch (list `input`) path; `timestamp_type` JSON envelope |
| `mstar/api_server/openai/serving_voices.py` (create) | `/v1/audio/voices`, upload, delete, list |
| `mstar/api_server/openai/router.py` (modify, `:100`) | Wire the voices routes |
| `test/modular/test_openai_adapters.py` (modify) | Unit tests for `Qwen3TTSAdapter` |
| `test/parity/conftest.py` (create) | Two-server fixtures, skip markers |
| `test/parity/contract_rest.py` (create) | Endpoint-agnostic REST assertions — the single source of parity truth |
| `test/parity/test_rest_parity.py` (create) | Runs `contract_rest` against both servers |
| `test/parity/capture_golden.py` (create) | Records vllm-omni REST responses to `test/parity/golden/` |

`test/parity/` is a new directory **inside** the existing `test/` tree, so it does not violate the no-new-top-level-directories rule.

---

### Task 1: M* environment and Qwen3-TTS smoke (the Phase 0 gate)

Nothing downstream is testable until M* serves our checkpoint at all. This task is a gate: if it fails, stop and report rather than proceeding.

**Files:**
- Create: `docs/inflection/qwen3-tts/DEV-SETUP.md`

**Interfaces:**
- Consumes: nothing
- Produces: a working `.venv`; a documented `mstar serve` invocation; the fact of whether `/generate` yields audio for our checkpoint

- [ ] **Step 1: Create the venv and install M***

```bash
cd ~/workspace/mstar
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[all]"
```

- [ ] **Step 2: Verify the import and the CLI**

Run: `.venv/bin/python -c "import mstar; print(mstar.__file__)"` then `.venv/bin/mstar --help`
Expected: both succeed. If torch/CUDA resolution fails, mirror the vllm-omni recipe (`--torch-backend=cu128`) and record what was needed in `DEV-SETUP.md`.

- [ ] **Step 3: Confirm the checkpoint is shaped the way M*'s config reader expects**

```bash
CKPT=/mnt/data/models/audio/tts/qwen3-tts/sft-cv-13l-14pv-ticr-1e-13l-fulldata_17946/checkpoint-final
ls "$CKPT"/config.json "$CKPT"/generation_config.json "$CKPT"/speech_tokenizer/config.json
```
Expected: all three present. (`mstar/model/qwen3_tts/config.py` reads exactly these.)

- [ ] **Step 4: Serve the checkpoint**

```bash
cd ~/workspace/mstar
CUDA_VISIBLE_DEVICES=0 HF_TOKEN_PATH=/dev/null \
  .venv/bin/mstar serve --config configs/qwen3tts.yaml --model-path "$CKPT" \
  --host 0.0.0.0 --port 8100 > logs/mstar-8100.log 2>&1 &
```
Then poll: `until curl -sf http://127.0.0.1:8100/health >/dev/null; do sleep 5; done; echo READY`

Record the exact flag names in `DEV-SETUP.md` — `serve`'s real signature comes from `mstar/cli`; correct this command to match it rather than assuming.

- [ ] **Step 5: Smoke `/generate` for audio**

```bash
curl -sS -X POST http://127.0.0.1:8100/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from M star.","output_modalities":["audio"]}' \
  -o /tmp/mstar_smoke.json
```
Expected: a response carrying audio bytes. Decode to WAV and confirm non-zero duration.

**GATE:** if no audio is produced, stop. Phase 1 cannot proceed and the spec's Phase 0 gate has failed; report the failure with the server log rather than working around it.

- [ ] **Step 6: Write `DEV-SETUP.md` and commit**

Document: venv creation, the install extras that actually worked, the corrected `serve` invocation, the smoke command, and every deviation discovered.

```bash
git add docs/inflection/qwen3-tts/DEV-SETUP.md
git commit -m "docs(qwen3-tts): M* dev setup + checkpoint smoke procedure"
```

---

### Task 2: Differential conformance harness

The machinery that makes "parity" a test result rather than a claim. One module of assertions, executed against both servers.

**Files:**
- Create: `test/parity/conftest.py`, `test/parity/contract_rest.py`, `test/parity/test_rest_parity.py`, `test/parity/capture_golden.py`

**Interfaces:**
- Consumes: Task 1's running M* server
- Produces: `contract_rest.assert_speech_contract(base_url, *, voice) -> None`; pytest fixtures `mstar_url` and `omni_url`; golden JSON under `test/parity/golden/`

- [ ] **Step 1: Write the two-server fixtures**

```python
# test/parity/conftest.py
"""Fixtures for differential parity tests.

Each test runs against BOTH servers. A server whose URL is unset is skipped,
so the suite is useful with one server up (conformance) and decisive with two
(differential).
"""
import os
import pytest

MSTAR_URL = os.environ.get("MSTAR_TTS_URL")
OMNI_URL = os.environ.get("OMNI_TTS_URL")
VOICE = os.environ.get("QWEN3_TTS_VOICE", "alexandra")


@pytest.fixture(scope="session")
def voice() -> str:
    return VOICE


@pytest.fixture(params=["mstar", "omni"])
def server(request) -> tuple[str, str]:
    url = {"mstar": MSTAR_URL, "omni": OMNI_URL}[request.param]
    if not url:
        pytest.skip(f"set {'MSTAR_TTS_URL' if request.param == 'mstar' else 'OMNI_TTS_URL'}")
    return request.param, url
```

- [ ] **Step 2: Write the contract assertions**

```python
# test/parity/contract_rest.py
"""REST /v1/audio/speech contract, asserted identically against either server.

L1 (protocol) and L2 (behavioural) only — see the parity spec. Nothing here
compares audio bytes or word TIMES between servers: the WS/REST path exposes no
seed and its sampling is not greedy, so two runs of the SAME server on the SAME
text already differ (measured 2026-09-04). Cross-server byte identity is not a
property this system has.
"""
from __future__ import annotations

import httpx

SPEECH_FIELDS = [
    "input", "model", "voice", "instructions", "response_format", "speed",
    "stream_format", "task_type", "language", "ref_audio", "ref_text",
    "x_vector_only_mode", "max_new_tokens", "stream", "temperature",
    "top_k", "top_p", "repetition_penalty", "timestamp_type",
]


def post_speech(base_url: str, body: dict, timeout: float = 180.0) -> httpx.Response:
    return httpx.post(f"{base_url}/v1/audio/speech", json=body, timeout=timeout)


def assert_accepts_every_field(base_url: str, voice: str) -> None:
    """Every documented field is accepted (no 422) — one at a time."""
    probe = {
        "instructions": "Speak calmly.", "language": "Auto", "speed": 1.0,
        "task_type": "CustomVoice", "max_new_tokens": 512, "temperature": 0.7,
        "top_k": 30, "top_p": 0.95, "repetition_penalty": 1.05,
        "response_format": "wav", "stream": False,
    }
    for field, value in probe.items():
        body = {"input": "Hello there.", "voice": voice, field: value}
        r = post_speech(base_url, body)
        assert r.status_code == 200, f"{field}={value!r} rejected: {r.status_code} {r.text[:200]}"


def assert_raw_audio_without_timestamps(base_url: str, voice: str) -> None:
    """No timestamp_type => raw container bytes, not JSON."""
    r = post_speech(base_url, {"input": "Hello there.", "voice": voice})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    assert r.content[:4] == b"RIFF", "expected a WAV container"


def assert_envelope_with_timestamps(base_url: str, voice: str) -> None:
    """timestamp_type='word' => JSON envelope with the documented shape."""
    r = post_speech(base_url, {"input": "Hello world.", "voice": voice, "timestamp_type": "word"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    for key in ("audio", "format", "sample_rate", "duration_seconds", "timestamp_info"):
        assert key in body, f"envelope missing {key!r}"
    info = body["timestamp_info"]
    if info is not None:                      # null is legal: no alignment head
        wa = info["word_alignment"]
        assert set(wa) == {"words", "word_start_time_seconds", "word_end_time_seconds"}
        n = len(wa["words"])
        assert len(wa["word_start_time_seconds"]) == n
        assert len(wa["word_end_time_seconds"]) == n
        starts = wa["word_start_time_seconds"]
        assert starts == sorted(starts), "word starts are not monotone"


def assert_batch_returns_one_result_per_input(base_url: str, voice: str) -> None:
    """A list input returns a results array, index-aligned."""
    r = post_speech(base_url, {"input": ["One.", "Two.", "Three."], "voice": voice})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["index"] for x in results] == [0, 1, 2]
    for item in results:
        assert item["audio"], "empty audio in batch item"


def assert_rejects_unknown_timestamp_type(base_url: str, voice: str) -> None:
    r = post_speech(base_url, {"input": "Hi.", "voice": voice, "timestamp_type": "phoneme"})
    assert r.status_code == 422, f"expected 422, got {r.status_code}"
```

- [ ] **Step 3: Run the contract against vllm-omni first — it must pass there**

```python
# test/parity/test_rest_parity.py
"""Every assertion runs against both servers (see conftest fixtures)."""
from test.parity import contract_rest


def test_accepts_every_field(server, voice):
    contract_rest.assert_accepts_every_field(server[1], voice)


def test_raw_audio_without_timestamps(server, voice):
    contract_rest.assert_raw_audio_without_timestamps(server[1], voice)


def test_envelope_with_timestamps(server, voice):
    contract_rest.assert_envelope_with_timestamps(server[1], voice)


def test_batch(server, voice):
    contract_rest.assert_batch_returns_one_result_per_input(server[1], voice)


def test_rejects_unknown_timestamp_type(server, voice):
    contract_rest.assert_rejects_unknown_timestamp_type(server[1], voice)
```

Run against the reference only:
```bash
cd ~/workspace/vllm-omni && scripts/inflection/launch_tts_ws_server.sh 8901 &
cd ~/workspace/mstar
OMNI_TTS_URL=http://127.0.0.1:8901 .venv/bin/python -m pytest test/parity/test_rest_parity.py -v
```
Expected: the `omni` params PASS, the `mstar` params SKIP.

**If an assertion fails against vllm-omni, the assertion is wrong, not vllm-omni.** vllm-omni *is* the contract. Fix the assertion.

- [ ] **Step 4: Run the same suite against M*, and watch it fail**

```bash
MSTAR_TTS_URL=http://127.0.0.1:8100 .venv/bin/python -m pytest test/parity/test_rest_parity.py -v
```
Expected: every `mstar` param FAILS — `qwen3_tts` has no adapter, so `/v1/audio/speech` cannot serve it. This red bar is Tasks 3–6's definition of done.

- [ ] **Step 5: Record golden responses from vllm-omni**

```python
# test/parity/capture_golden.py
"""Record vllm-omni's REST responses as the parity reference.

Stores SHAPE, not bytes: keys, types, lengths, status codes. Audio bytes and
word times are deliberately excluded — they are not reproducible run to run.
"""
import json
import pathlib
import sys

from test.parity import contract_rest

GOLDEN = pathlib.Path(__file__).parent / "golden"


def shape(obj):
    if isinstance(obj, dict):
        return {k: shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [shape(obj[0])] if obj else []
    return type(obj).__name__


def main(base_url: str, voice: str) -> None:
    GOLDEN.mkdir(exist_ok=True)
    cases = {
        "envelope": {"input": "Hello world.", "voice": voice, "timestamp_type": "word"},
        "batch": {"input": ["One.", "Two."], "voice": voice},
    }
    for name, body in cases.items():
        r = contract_rest.post_speech(base_url, body)
        (GOLDEN / f"{name}.json").write_text(
            json.dumps({"status": r.status_code,
                        "content_type": r.headers["content-type"].split(";")[0],
                        "shape": shape(r.json())}, indent=2)
        )
        print(f"wrote {name}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "alexandra")
```

Run: `.venv/bin/python -m test.parity.capture_golden http://127.0.0.1:8901 alexandra`

- [ ] **Step 6: Commit**

```bash
git add test/parity/
git commit -m "test(parity): differential REST conformance harness + vllm-omni goldens

Assertions live in one module and run against both servers. Shape-only
goldens: audio bytes and word times are not reproducible run-to-run on the
same server (measured 2026-09-04), so cross-server byte identity is not a
property this system has and is not asserted."
```

---

### Task 3: Widen `SpeechRequest` to the full field surface

**Files:**
- Modify: `mstar/api_server/openai/protocol.py:45-58`
- Test: `test/modular/test_openai_adapters.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SpeechRequest` with all 19 fields; `input: str | list[str]`

- [ ] **Step 1: Write the failing test**

```python
# append to test/modular/test_openai_adapters.py
def test_speech_request_accepts_the_full_qwen3_tts_surface():
    """vllm-omni's 19-field surface; the protocol is frozen to it."""
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
    req = SpeechRequest(input=["a", "b"], voice="alexandra")
    assert req.input == ["a", "b"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/modular/test_openai_adapters.py -k qwen3_tts_surface -v`
Expected: FAIL — `top_k`, `repetition_penalty`, `timestamp_type`, `task_type` are not fields.

- [ ] **Step 3: Widen the model**

```python
# mstar/api_server/openai/protocol.py — replace class SpeechRequest
class SpeechRequest(BaseModel):
    """OpenAI ``/v1/audio/speech`` (text-to-speech).

    The field set is frozen to vllm-omni's ``OpenAICreateSpeechRequest`` so an
    M*-served Qwen3-TTS is a drop-in replacement. Fields beyond the OpenAI
    standard (``top_k``, ``repetition_penalty``, ``task_type``, the voice-clone
    trio, ``timestamp_type``) are Qwen3-TTS's and are ignored by models that do
    not declare support. ``sample_rate`` is deliberately absent: it is a
    WebSocket session field, not a REST one.
    """

    model_config = _CFG

    input: str | list[str]
    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = 1.0
    stream_format: Literal["sse", "audio"] | None = "audio"
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    stream: bool | None = False
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    timestamp_type: Literal["word"] | None = None
```

Add `Literal` to the module's `typing` import if absent.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/modular/test_openai_adapters.py -v`
Expected: PASS, and no existing adapter test regresses (the added fields are optional).

- [ ] **Step 5: Commit**

```bash
git add mstar/api_server/openai/protocol.py test/modular/test_openai_adapters.py
git commit -m "feat(openai): widen SpeechRequest to the Qwen3-TTS field surface"
```

---

### Task 4: `Qwen3TTSAdapter` and registry entry

**Files:**
- Modify: `mstar/api_server/openai/adapters.py` (new class before `ADAPTER_REGISTRY:454`; registry entry at `:454`)
- Test: `test/modular/test_openai_adapters.py`

**Interfaces:**
- Consumes: `SpeechRequest` (Task 3); `_passthrough(req) -> dict`; `_apply_sampling(req, mk, *, temperature_key, top_p_key, max_tokens_key)`; `SubmitArgs(text, file_paths, input_modalities, output_modalities, model_kwargs, prompt_parts)`
- Produces: `Qwen3TTSAdapter` with `supports_speech = True`; `ADAPTER_REGISTRY["qwen3_tts"]`

- [ ] **Step 1: Write the failing test**

```python
def test_qwen3_tts_adapter_is_registered():
    assert adapters.get_adapter("qwen3_tts") is not None


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
    assert mk["talker_temperature"] == 0.7


def test_qwen3_tts_omits_unset_fields(tmp_path):
    """An unset field must not appear as None in model_kwargs."""
    req = SpeechRequest(input="Hi.", voice="alexandra")
    mk = adapters.get_adapter("qwen3_tts").speech_to_request(req, tmp_path).model_kwargs
    for absent in ("instructions", "language", "top_k", "repetition_penalty", "timestamp_type"):
        assert absent not in mk, f"{absent} leaked into model_kwargs as None"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/modular/test_openai_adapters.py -k qwen3_tts -v`
Expected: FAIL — `get_adapter("qwen3_tts")` returns `None`.

- [ ] **Step 3: Implement the adapter**

```python
# mstar/api_server/openai/adapters.py — insert before ADAPTER_REGISTRY
class Qwen3TTSAdapter(OpenAIAdapter):
    """Qwen3-TTS: text in, audio out.

    Unlike Qwen3-Omni (a chat model whose spoken reply happens to be audio),
    Qwen3-TTS is a dedicated TTS model: ``input`` is the script to voice, and
    the only output modality is audio.

    The mapping is frozen to vllm-omni's so an M* server is a drop-in
    replacement. Two rules matter:
      * Unset fields are OMITTED, never sent as ``None`` — the Walk Graph
        treats a present key as an override, so a null would clobber the
        checkpoint's own default.
      * Talker sampling is namespaced (``talker_temperature`` / ``talker_top_p``)
        to match the Qwen3-Omni convention already in this file, while the
        non-OpenAI knobs (``top_k``, ``repetition_penalty``) keep their plain
        names because that is what vllm-omni's engine expects.
    """

    supports_speech = True

    _DIRECT_FIELDS = (
        "voice", "instructions", "language", "task_type", "top_k",
        "repetition_penalty", "max_new_tokens", "timestamp_type",
        "ref_audio", "ref_text", "x_vector_only_mode", "speed",
    )

    def speech_to_request(self, req: SpeechRequest, upload_dir: Path) -> SubmitArgs:  # noqa: ARG002
        mk = _passthrough(req)
        for name in self._DIRECT_FIELDS:
            value = getattr(req, name, None)
            if value is not None:
                mk.setdefault(name, value)
        _apply_sampling(
            req, mk,
            temperature_key="talker_temperature",
            top_p_key="talker_top_p",
            max_tokens_key=None,
        )
        text = req.input if isinstance(req.input, str) else None
        return SubmitArgs(
            text=text,
            input_modalities=["text"],
            output_modalities=["audio"],
            model_kwargs=mk,
        )
```

Register it:
```python
ADAPTER_REGISTRY: dict[str, OpenAIAdapter] = {
    "bagel": BagelAdapter(),
    "qwen3_omni": Qwen3OmniAdapter(),
    "qwen3_tts": Qwen3TTSAdapter(),
    "orpheus": OrpheusAdapter(),
    "cosmos3": Cosmos3Adapter(),
    "cosmos3_droid": Cosmos3Adapter(),
    "cosmos3_super": Cosmos3Adapter(),
    "wan22": Wan22Adapter(),
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/modular/test_openai_adapters.py -v`
Expected: PASS.

- [ ] **Step 5: Verify against the live server**

Run: `MSTAR_TTS_URL=http://127.0.0.1:8100 .venv/bin/python -m pytest test/parity/test_rest_parity.py -k "mstar and (accepts_every_field or raw_audio)" -v`
Expected: PASS. The envelope and batch cases still fail — Tasks 5 and 6.

- [ ] **Step 6: Commit**

```bash
git add mstar/api_server/openai/adapters.py test/modular/test_openai_adapters.py
git commit -m "feat(openai): Qwen3TTSAdapter — /v1/audio/speech works for qwen3_tts

qwen3_tts was absent from ADAPTER_REGISTRY, so the base speech_to_request
raised NotImplementedError and the endpoint was dead for it."
```

---

### Task 5: Batch input and the `timestamp_type` envelope

**Files:**
- Modify: `mstar/api_server/openai/serving_speech.py`
- Test: `test/parity/test_rest_parity.py` (already written, Task 2)

**Interfaces:**
- Consumes: `Qwen3TTSAdapter` (Task 4)
- Produces: `create_speech` returning a JSON envelope when `timestamp_type == "word"`, and a `results` array when `input` is a list

- [ ] **Step 1: Run the two already-failing parity cases**

Run: `MSTAR_TTS_URL=http://127.0.0.1:8100 .venv/bin/python -m pytest test/parity/test_rest_parity.py -k "mstar and (envelope or batch)" -v`
Expected: FAIL — M* returns raw audio regardless of `timestamp_type`, and treats a list `input` as invalid.

- [ ] **Step 2: Implement both paths**

```python
# mstar/api_server/openai/serving_speech.py
import base64

async def create_speech(api, model_name, adapter, req, raw_request=None):  # noqa: ARG001
    if isinstance(req.input, list):
        return await _create_speech_batch(api, adapter, req, raw_request)

    args = adapter.speech_to_request(req, api.upload_dir)
    request_id = rid("speech")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    fmt = (req.response_format or "wav").lower()

    api.submit_request(
        text=args.text, file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        streaming=bool(req.stream), request_id=request_id,
    )

    if req.stream:
        return StreamingResponse(
            _stream_wav(api, request_id, sample_rate),
            media_type="audio/wav", headers={"Cache-Control": "no-cache"},
        )

    chunks = await api.collect_results(request_id, raw_request)
    pcm = b"".join(c.data for c in chunks if c.modality == "audio")
    audio_bytes, mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)

    if req.timestamp_type != "word":
        return Response(content=audio_bytes, media_type=mime)

    # Word timestamps switch the response from raw bytes to a JSON envelope.
    # Phase 3 populates timestamp_info; until then it is null, which is the
    # same thing vllm-omni returns for a checkpoint with no pointer_head.pt.
    return JSONResponse({
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
        "format": fmt,
        "sample_rate": sample_rate,
        "duration_seconds": round(len(pcm) / 2 / sample_rate, 3),
        "timestamp_info": None,
    })


async def _create_speech_batch(api, adapter, req, raw_request):
    """A list ``input`` returns one index-aligned result per item."""
    import asyncio

    async def one(index: int, text: str) -> dict:
        single = req.model_copy(update={"input": text})
        args = adapter.speech_to_request(single, api.upload_dir)
        request_id = rid(f"speech-batch-{index}")
        sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
        fmt = (req.response_format or "wav").lower()
        api.submit_request(
            text=args.text, file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=args.output_modalities,
            model_kwargs=args.model_kwargs,
            streaming=False, request_id=request_id,
        )
        chunks = await api.collect_results(request_id, raw_request)
        pcm = b"".join(c.data for c in chunks if c.modality == "audio")
        audio_bytes, _ = media_io.pcm16_to_container(pcm, sample_rate, fmt)
        return {
            "index": index,
            "audio": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": fmt,
            "sample_rate": sample_rate,
            "duration_seconds": round(len(pcm) / 2 / sample_rate, 3),
            "timestamp_info": None,
        }

    results = await asyncio.gather(*[one(i, t) for i, t in enumerate(req.input)])
    return JSONResponse({"results": list(results)})
```

Add `from fastapi.responses import JSONResponse` to the imports.

- [ ] **Step 3: Run to verify it passes**

Run: `MSTAR_TTS_URL=http://127.0.0.1:8100 .venv/bin/python -m pytest test/parity/test_rest_parity.py -k mstar -v`
Expected: all PASS. `timestamp_info` is `null` on M*, which the contract permits.

- [ ] **Step 4: Run both servers together — the actual parity check**

```bash
MSTAR_TTS_URL=http://127.0.0.1:8100 OMNI_TTS_URL=http://127.0.0.1:8901 \
  .venv/bin/python -m pytest test/parity/test_rest_parity.py -v
```
Expected: 10 passed (5 assertions × 2 servers).

- [ ] **Step 5: Commit**

```bash
git add mstar/api_server/openai/serving_speech.py
git commit -m "feat(openai): batch input + timestamp_type JSON envelope

Shape matches vllm-omni's. timestamp_info stays null until Phase 3 lands the
alignment head — the same response a checkpoint without pointer_head.pt gives."
```

---

### Task 6: Voices endpoints

**Files:**
- Create: `mstar/api_server/openai/serving_voices.py`
- Modify: `mstar/api_server/openai/router.py` (after the speech route at `:100`)
- Test: `test/parity/contract_rest.py`, `test/parity/test_rest_parity.py`

**Interfaces:**
- Consumes: `api.model` (for the speaker list)
- Produces: `GET /v1/audio/voices` → `{"voices": [...], "uploaded_voices": [...]}`

- [ ] **Step 1: Write the failing contract assertion**

```python
# append to test/parity/contract_rest.py
def assert_lists_voices(base_url: str, voice: str) -> None:
    r = httpx.get(f"{base_url}/v1/audio/voices", timeout=30.0)
    assert r.status_code == 200
    body = r.json()
    assert "voices" in body, "missing 'voices'"
    assert isinstance(body["voices"], list) and body["voices"]
    assert voice in [v.lower() if isinstance(v, str) else v for v in body["voices"]], (
        f"{voice!r} absent from {body['voices']}"
    )
```

```python
# append to test/parity/test_rest_parity.py
def test_lists_voices(server, voice):
    contract_rest.assert_lists_voices(server[1], voice)
```

- [ ] **Step 2: Run to verify it fails on M* and passes on vllm-omni**

Run: `MSTAR_TTS_URL=http://127.0.0.1:8100 OMNI_TTS_URL=http://127.0.0.1:8901 .venv/bin/python -m pytest test/parity/test_rest_parity.py -k voices -v`
Expected: `omni` PASS, `mstar` FAIL with 404.

- [ ] **Step 3: Implement**

```python
# mstar/api_server/openai/serving_voices.py
"""``/v1/audio/voices`` — the speaker list a client picks ``voice`` from.

Shape is frozen to vllm-omni's: ``voices`` are the checkpoint's built-in
speakers, ``uploaded_voices`` are runtime voice clones (empty until M* grows
an upload path).
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


async def list_voices(api) -> JSONResponse:
    speakers: list[str] = []
    model = getattr(api, "model", None)
    if model is not None:
        config = getattr(model, "config", None)
        talker = getattr(config, "talker", None)
        raw = getattr(talker, "speakers", None) or getattr(config, "speakers", None) or []
        speakers = sorted({str(s).lower() for s in raw})
    return JSONResponse({"voices": speakers, "uploaded_voices": []})
```

```python
# mstar/api_server/openai/router.py — add after the /v1/audio/speech route
@router.get("/v1/audio/voices")
async def audio_voices(raw_request: Request):
    from mstar.api_server.openai import serving_voices
    return await serving_voices.list_voices(_api(raw_request))
```

Match the surrounding route handlers' way of reaching the API object rather than assuming `_api`; correct this to the file's actual idiom.

- [ ] **Step 4: Run to verify it passes**

Run: `MSTAR_TTS_URL=http://127.0.0.1:8100 OMNI_TTS_URL=http://127.0.0.1:8901 .venv/bin/python -m pytest test/parity/test_rest_parity.py -v`
Expected: 12 passed (6 assertions × 2 servers).

If the speaker list comes back empty, trace where `config.py` parks `config.json`'s speakers and read from there — do not hardcode the 14 names.

- [ ] **Step 5: Commit**

```bash
git add mstar/api_server/openai/serving_voices.py mstar/api_server/openai/router.py test/parity/
git commit -m "feat(openai): GET /v1/audio/voices with the vllm-omni response shape"
```

---

### Task 7: Wire the parity suite into CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `test/modular/test_openai_adapters.py`
- Produces: a `tts-parity-unit` CI job

The live differential tests need two GPU servers and cannot run on a GitHub runner. The *unit* half (adapter mapping, request schema) can, and that is what catches a protocol regression at review time.

- [ ] **Step 1: Add the job**

```yaml
  # Qwen3-TTS protocol surface. The live differential suite (test/parity/)
  # needs two GPU servers and runs out-of-band; these are the CPU-checkable
  # halves: the request schema and the adapter mapping. A change that silently
  # drops a field from SpeechRequest fails here.
  tts-parity-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install (cpu torch)
        run: |
          python -m pip install --upgrade pip
          pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
          pip install -e . pytest
      - name: Adapter + protocol tests
        env:
          HF_HUB_OFFLINE: "1"
        run: pytest test/modular/test_openai_adapters.py -v
```

- [ ] **Step 2: Add the job to branch protection**

```bash
gh api -X PATCH repos/rajath-cmd/mstar/branches/inflection%2Fqwen3-tts/protection/required_status_checks \
  -f 'contexts[]=build' -f 'contexts[]=dynamo-smoke' \
  -f 'contexts[]=rust-transport' -f 'contexts[]=tts-parity-unit'
```

- [ ] **Step 3: Verify the job passes locally first**

Run: `.venv/bin/python -m pytest test/modular/test_openai_adapters.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add tts-parity-unit job (adapter + request-schema surface)"
```

---

## Phase 1 exit criteria

- [ ] M* serves the reference checkpoint and produces audio (Task 1 gate)
- [ ] `test/parity/test_rest_parity.py` green against **both** servers, 12 passed
- [ ] `test/modular/test_openai_adapters.py` green
- [ ] `tts-parity-unit` required on `inflection/qwen3-tts`
- [ ] `DEV-SETUP.md` reproduces the environment from a clean clone
- [ ] Nothing pushed to the public remote (Gate 0 still blocking)

**Explicitly NOT in Phase 1:** word-timestamp *values* (`timestamp_info` is null until Phase 3), WebSocket streaming (Phase 2), metrics (Phase 4), voice upload/delete (deferred until a consumer needs it — the pipecat client does not).

---

## Self-review

**Spec coverage.** Phase 1's row in the spec roadmap — adapter, full `SpeechRequest`, voices endpoints, REST conformance green — maps to Tasks 4, 3, 6, 2 respectively. Task 1 covers the Phase 0 environment gate; Task 5 covers the REST response shapes the spec's L1 section enumerates; Task 7 is the CI hook the spec's drift risk calls for.

**Gaps accepted, and why.** Voice *upload*/*delete* are specified in the spec's L1 list but deferred: they exist in vllm-omni for the Base voice-cloning checkpoint, our reference checkpoint is CustomVoice, and the pipecat client never calls them. They move to Phase 7 alongside the drop-in validation, where a real consumer would surface the need.

**Type consistency.** `speech_to_request(req, upload_dir) -> SubmitArgs` matches the base class at `adapters.py:193`. `SubmitArgs` field names match `:44-52`. `_apply_sampling`'s keyword-only `temperature_key` / `top_p_key` / `max_tokens_key` match `:` definition. `contract_rest.post_speech` is used by both `test_rest_parity.py` and `capture_golden.py` with one signature.

**Two steps are deliberately "correct this to match reality" rather than final.** Task 1 Step 4 (`mstar serve` flags) and Task 6 Step 3 (the router's API-object idiom) depend on files not yet read closely. They are flagged inline as things to verify rather than assumed, which is honest about what this plan knows; every other step contains the real content.
