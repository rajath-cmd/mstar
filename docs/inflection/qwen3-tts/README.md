# Qwen3-TTS on M* — build, run, verify

A drop-in replacement for the vllm-omni Qwen3-TTS server: same WebSocket
protocol, same HTTP API, same request surface. Point an existing client at it
and change nothing else.

- [Quick start](#quick-start)
- [Docker](#docker)
- [Bare metal](#bare-metal)
- [Configuration profiles](#configuration-profiles) — **the one choice that matters**
- [API](#api)
- [Benchmarks](#benchmarks) — reproduce every number
- [Parity test suite](#parity-test-suite)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
CKPT=/path/to/qwen3-tts/checkpoint-final

docker build -f docker/Dockerfile.qwen3-tts -t inflection/qwen3-tts-mstar:latest .

docker run -d --name qwen3-tts --gpus '"device=0"' --ipc=host -p 8100:8100 \
  -v "$CKPT":/checkpoint:ro -e MSTAR_MODEL_PATH=/checkpoint \
  inflection/qwen3-tts-mstar:latest

until curl -sf http://127.0.0.1:8100/health >/dev/null; do sleep 5; done

curl -X POST http://127.0.0.1:8100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from M star.","voice":"alexandra"}' --output out.wav
```

First start takes 4–7 minutes: weight load plus CUDA-graph capture. The
healthcheck allows for it (`start-period=420s`).

---

## Docker

### Build

```bash
docker build -f docker/Dockerfile.qwen3-tts -t inflection/qwen3-tts-mstar:latest .
```

CUDA 13 host? Override the three build args together — a mismatch between the
CUDA image, the torch index and the flash-attn wheel fails at import, not at
build:

```bash
docker build -f docker/Dockerfile.qwen3-tts \
  --build-arg CUDA_IMAGE=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
  --build-arg FLASH_ATTN_WHEEL=<matching cu13/torch2.9/cp312 wheel> \
  -t inflection/qwen3-tts-mstar:cu13 .
```

The CUDA 12.8 default is a host-compatibility floor: a cu12.8 image runs on r570
and r580+ drivers, a CUDA 13 image needs r580+.

The build fails fast if the image cannot import `mstar`, if `qwen3_tts` is not
in the adapter registry, or if the WebSocket handler is missing — all three have
broken silently before.

### Run

Weights are **not** baked in; one image serves any Qwen3-TTS fine-tune.

```bash
docker run -d --name qwen3-tts \
  --gpus '"device=0"' --ipc=host -p 8100:8100 \
  -v /path/to/checkpoint-final:/checkpoint:ro \
  -e MSTAR_MODEL_PATH=/checkpoint \
  -e MSTAR_TTS_CONFIG=/opt/mstar/src/configs/inflection_qwen3tts_lowlatency.yaml \
  --restart unless-stopped \
  inflection/qwen3-tts-mstar:latest
```

`--ipc=host` is required: M* moves tensors between the Talker and Codec nodes
through shared memory, and the default 64 MB `/dev/shm` is too small.

| env | default | meaning |
|---|---|---|
| `MSTAR_MODEL_PATH` | `/checkpoint` | Checkpoint directory (mounted) |
| `MSTAR_TTS_CONFIG` | low-latency profile | Deployment profile, see below |
| `MSTAR_TTS_PORT` | `8100` | Listen port |
| `MSTAR_TTS_GPUS` | `0` | `--gpus` value passed to `mstar serve` |
| `MSTAR_TTS_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

The checkpoint directory must contain `config.json`, `generation_config.json`,
`model.safetensors` and `speech_tokenizer/config.json`. The entrypoint checks all
four and exits with a readable message rather than failing inside worker startup.
A `pointer_head.pt` beside them is the TFA alignment head (word timestamps —
Phase 3; M* does not read it yet).

---

## Bare metal

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[all]"

MSTAR_MODEL_PATH=/path/to/checkpoint-final \
CONFIG=$PWD/configs/inflection_qwen3tts_lowlatency.yaml \
  scripts/inflection/launch_mstar_qwen3_tts.sh 8100
```

Use the launcher rather than calling `mstar serve` directly: it puts `.venv/bin`
on `PATH`, which FlashInfer's JIT needs to find `ninja`. Without it the server
logs `FileNotFoundError: ninja` on every kernel capture and never binds.

---

## Configuration profiles

**This is the one choice that changes behaviour.** Both profiles serve the same
model and the same API; they differ only in codec streaming cadence.

| | `inflection_qwen3tts_lowlatency.yaml` | `inflection_qwen3tts.yaml` |
|---|---|---|
| For | voice agents, barge-in | batch / offline synthesis |
| `codec_chunk_frames` | small (see sweep) | 300 (checkpoint default) |
| First audio after | a few frames | **300 frames = 24 s** |
| Throughput | lower | highest |

`codec_chunk_frames` is how many 12.5 Hz codec frames the Codec node buffers
before emitting any PCM, so it *is* the time-to-first-audio floor. The
checkpoint's stock 300 means short replies finish generating before a single byte
leaves the server.

**`chunk_frames` must exceed `left_context_frames`.** The chunk policy advances
by `chunk - left_context` on its first pop, so a violating pair advances
backwards and silently drops the start of every stream — audio that still sounds
fluent because it simply begins mid-sentence. `LeftContextChunkPolicy` now
rejects such a configuration at construction; to go lower, reduce **both**.

Tune with `scripts/inflection/sweep_codec_latency.sh`, which measures TTFA and
checks each arm's audio is complete.

---

## API

Identical to vllm-omni's. Full client guide: `InflectionAI/vllm-omni` issue #44.

### REST

`POST /v1/audio/speech` — 19 fields: `input` (str or list), `model`, `voice`,
`instructions`, `response_format`, `speed`, `stream_format`, `task_type`,
`language`, `ref_audio`, `ref_text`, `x_vector_only_mode`, `max_new_tokens`,
`stream`, `temperature`, `top_k`, `top_p`, `repetition_penalty`,
`timestamp_type`.

Returns raw container bytes; with `timestamp_type: "word"` it returns a JSON
envelope (`audio`, `format`, `sample_rate`, `duration_seconds`,
`timestamp_info`). A list `input` returns an index-aligned `results` array.

`GET /v1/audio/voices` — `{"voices": [...], "uploaded_voices": []}`, read from
the loaded checkpoint.

Validation failures answer **400** with `{"error": {message, type, param, code}}`
— not FastAPI's default 422 + `detail`, which would break a client branching on
the status or reading `error.message`.

### WebSocket

`/v1/audio/speech/stream` — the path a voice agent should use, and the only one
with meaningful TTFA.

```
-> {"type":"session.config","voice":"alexandra","language":"Auto"}
-> {"type":"input.text","text":"Hello there. "}
-> {"type":"input.done"}
<- {"type":"audio.start","sentence_index":0,"sentence_text":"...","format":"wav"}
<- <binary PCM16 frames>
<- {"type":"audio.done","sentence_index":0,"sample_rate":24000,"chunk_count":N}
<- {"type":"session.done","total_sentences":1}
```

`{"type":"cancel"}` barges in and answers `{"type":"cancelled","sentence_index":N,"drained":M}`.

`voice.list` and `voice.delete` are **one-shot commands sent before
`session.config`**; they answer and close. Sent mid-session they are rejected as
unknown — matching vllm-omni, whose own docstring lists them as general messages
but whose implementation only handles them pre-config.

---

## Benchmarks

Everything below is reproducible. Start two servers (M* and vllm-omni) on
separate GPUs with the **same checkpoint** — the comparison is void otherwise.

```bash
CKPT=/path/to/checkpoint-final
MSTAR_MODEL_PATH=$CKPT GPUS=0 \
  CONFIG=$PWD/configs/inflection_qwen3tts_lowlatency.yaml \
  scripts/inflection/launch_mstar_qwen3_tts.sh 8100
MODEL=$CKPT CUDA_VISIBLE_DEVICES=1 \
  <vllm-omni>/scripts/inflection/launch_tts_ws_server.sh 8901
```

### TTFA + throughput over the WebSocket

```bash
python -m benchmark.inflection.tts_ws_ttfa \
  --mstar http://127.0.0.1:8100 --omni http://127.0.0.1:8901 \
  --concurrencies 1,4,8,16,32 --reps 5 --out out/ttfa
```

TTFA is measured from `input.done` to the first **binary** frame — the first PCM
a client could play. The `audio.start` control frame is not counted; timing it
would flatter both servers by the whole synthesis time.

### Batch latency + throughput over REST

```bash
python -m benchmark.inflection.tts_ab \
  --mstar http://127.0.0.1:8100 --omni http://127.0.0.1:8901 \
  --concurrencies 1,4,8,16,32 --reps 5 --mode batch --out out/tts_ab
```

`--mode batch` is the default deliberately. Neither server's REST *streaming*
gives a usable TTFA — M* flushes at completion, vllm-omni re-sends cumulatively
(each chunk repeats all prior audio; 9 MB for ~5 s). The harness detects that
pattern and suppresses audio-derived metrics rather than reporting the
physically impossible throughput it would otherwise compute.

### Codec cadence sweep

```bash
MSTAR_MODEL_PATH=$CKPT scripts/inflection/sweep_codec_latency.sh
```

Restarts the server per `chunk:context` pair, measures TTFA, and **verifies the
audio is complete** by transcribing it — a truncating config wins on TTFA while
being wrong, so speed alone cannot pick the winner.

### Reading the output

Every harness writes `results.json`, an interactive Plotly `.html`, and a
matplotlib `.png`.

Two guards worth knowing, because both caught real bugs:

- **Fairness gate** — mean audio duration per arm. A server emitting shorter
  audio wins every derived metric while doing less work, so the comparison is
  void unless durations match (±15 %).
- **Cumulative-resend detection** — monotonically growing chunk sizes mean the
  stream re-sends from the start, making byte-derived duration meaningless.

---

## Parity test suite

One assertion module run against **both** servers, so a divergence appears as
one red bar next to one green one.

```bash
MSTAR_TTS_URL=http://127.0.0.1:8100 \
OMNI_TTS_URL=http://127.0.0.1:8901 \
  python -m pytest test/parity -v
```

| suite | scope |
|---|---|
| `test_rest_parity.py` | `/v1/audio/speech` field surface, response shapes, batch, error envelope, voices |
| `test_ws_parity.py` | frame types and ordering, `sentence_index`, cancel, errors, voice commands |
| `test_alignment_parity.py` | word timestamps — **red on M* until Phase 3**, by design |

A server whose URL is unset is skipped, so the suite is useful with one server
up and decisive with two.

**vllm-omni is the contract.** An assertion that fails against it is a wrong
assertion, not an upstream bug. That rule has already corrected this suite twice:
once on the validation-error status (422 vs 400) and once on where `voice.list`
is legal.

CPU-checkable halves (request schema, adapter mapping) run in CI as
`tts-parity-unit`; the live differential suite needs two GPU servers.

---

## Troubleshooting

**`FileNotFoundError: ninja` in a PYTEST run** — same cause. `pytest` does not
inherit the venv's `bin`, so FlashInfer's JIT cannot compile and ~34
attention/sampling tests fail spuriously. Run them as
`PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest ...`.

**`FileNotFoundError: ninja`, server never binds** — FlashInfer JIT-compiles
attention kernels and shells out to `ninja` via `PATH`. Use the launcher, or put
`.venv/bin` first yourself.

**`ValueError: M* currently requires equal Talker and CodePredictor hidden
sizes`** — a pre-Phase-0b M*. The 1.7B checkpoint runs a 2048 Talker into a 1024
depth decoder via `small_to_mtp_projection`; support for that landed in
`4ad3a07`.

**`ValueError: LeftContextChunkPolicy requires chunk > left_context`** — working
as intended. See [Configuration profiles](#configuration-profiles).

**Audio starts mid-sentence** — a codec cadence violating the rule above, on a
build predating the guard. Verify with
`scripts/inflection/check_audio_complete.py`.

**`Unsupported Qwen3-TTS speaker 'vivian'`** — speaker sets are per-checkpoint.
`GET /v1/audio/voices` lists the loaded checkpoint's own.

**`timestamp_info` is `null`** — expected on M* today (Phase 3). On vllm-omni it
means the checkpoint ships no `pointer_head.pt`; check the startup log for
`loaded AlignmentPointerHead`.

**Server exits during startup, no error** — usually OOM from co-tenancy. Each
instance reserves a large share of its GPU; give each its own device.
