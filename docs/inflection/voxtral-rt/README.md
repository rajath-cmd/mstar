# Voxtral-Realtime on M* — build, run, verify

Vanilla `Voxtral-Mini-4B-Realtime-2602` (no auxiliary heads) running on M*,
with an OpenAI-compatible transcription endpoint.

**Status: bring-up, verified correct.** The model core is bit-exact against the
HuggingFace reference — 8 of 8 test utterances token-identical, English and
Chinese, 3.5 s to 25 s. The server in front of it is single-stream and is not
yet on M*'s Walk Graph engine. See [What is and is not done](#what-is-and-is-not-done).

- [How the model works](#how-the-model-works) — read this first
- [Quick start](#quick-start)
- [Docker](#docker)
- [API](#api)
- [Verifying correctness](#verifying-correctness)
- [Benchmarks](#benchmarks)
- [What is and is not done](#what-is-and-is-not-done)
- [Troubleshooting](#troubleshooting)

---

## How the model works

Voxtral-Realtime looks like a LLaVA-style audio LLM from its config and is not
one. The difference decides everything downstream, so it is worth 60 seconds.

**Audio embeddings are ADDED to text embeddings, position by position.** They
are not spliced in as a separate span of tokens. Position *i* of the sequence
carries both the *i*-th 80 ms audio frame and the *i*-th text token, summed:

```
position   0      1      2      3      4      5   ...
audio    [80ms] [80ms] [80ms] [80ms] [80ms] [80ms]     <- always present
text      <s>    PAD    PAD    "You"  PAD   "went"     <- emitted or PAD
           |      |      |      |      |      |
           +------+------+------+------+------+---> summed, then decoded
```

Three consequences:

1. **The sequence length is the audio length.** 12.5 positions per second,
   exactly (16000 Hz / 160-sample hop / 8 mel frames per token). A 10 s clip is
   125 positions, always.
2. **Transcription is not "generate until EOS".** It is a fixed budget of
   `audio_seconds x 12.5` decode steps. The transcript is whatever is not
   `[STREAMING_PAD]`. There is no `max_tokens` to tune and no runaway to guard
   against — when the audio ends there is nothing left to condition on.
3. **The prompt carries no text**, only positions: `<s>` followed by
   `[STREAMING_PAD]`.

This is what makes the model realtime-capable — it never needs the end of the
audio to emit the beginning of the transcript.

A fourth piece is easy to miss and caused the one real bug during bring-up: each
decoder layer scales its post-attention state by `1 + ada_rms_norm(t_cond)`,
where `t_cond` is a sinusoidal embedding of `num_delay_tokens` (3). The model is
told how far behind the audio it is allowed to run. Corrupt that and it
transcribes the right words at the wrong *times*.

### Shape

| stage | in | out |
|---|---|---|
| log-mel | waveform @ 16 kHz | `[128, T]` @ 100 Hz |
| embedder (2 causal convs, stride 2) | `[128, T]` | `[T/2, 1280]` @ 50 Hz |
| audio tower (32 layers, causal, window 750) | `[T/2, 1280]` | `[T/2, 1280]` |
| downsample x4 + projector | `[T/2, 1280]` | `[T/8, 3072]` @ **12.5 Hz** |
| text decoder (26 layers, GQA 32/8) | `[T/8, 3072]` | one token per position |

The audio tower runs **once**, densely, over the whole utterance. That is not a
simplification of the streaming design: every layer is causal with a fixed
window, so a dense pass is numerically identical to an incremental one with a KV
cache, at one kernel launch per layer instead of one per 80 ms.

---

## Quick start

```bash
CKPT=/mnt/data/models/audio/stt/voxtral-rt/pretrained/Voxtral-Mini-4B-Realtime-2602

uv pip install --python .venv/bin/python --no-deps \
    'mistral-common==1.11.7' tiktoken jsonschema jsonschema-specifications \
    referencing rpds-py pycountry pydantic-extra-types

CUDA_VISIBLE_DEVICES=0 VOXTRAL_MODEL_PATH=$CKPT \
  .venv/bin/python -m mstar.model.voxtral_rt.serving.app --port 8200

curl -X POST http://127.0.0.1:8200/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "response_format=json"
```

Startup is ~3 s (weight load only; no CUDA-graph capture yet).

### Why `--no-deps` on mistral-common

`mistral-common` pins `numpy<2.4` and M* runs 2.5.x. Installing it normally
downgrades numpy under the whole repo, including the Qwen3-TTS stack. The
listed packages are its actual runtime needs; numpy stays where it is.

`mistral_common` is required, not optional: it is the reference tokenizer the
checkpoint was trained with, and it also computes the audio padding. The padding
sets the alignment between mel frames and text positions — being one token out
shifts the entire transcript against the audio — so it is not reimplemented.

**Mel extraction is ported rather than imported.** The reference feature
extractor lives in `transformers>=5.16`, and M* pins 4.57 for the Qwen3-TTS
path; upgrading transformers to obtain one feature extractor would put a working
production stack at risk. The port is ~15 lines of STFT and was verified
bit-identical to the reference (max abs diff `0.000e+00`).

---

## Docker

Verified end to end: built, run on a mounted checkpoint, and transcribing
correctly on English and Chinese.

```bash
docker build -f docker/Dockerfile.voxtral-rt -t inflection/voxtral-rt-mstar:latest .

CKPT=/mnt/data/models/audio/stt/voxtral-rt/pretrained/Voxtral-Mini-4B-Realtime-2602
docker run -d --name voxtral-rt --gpus '"device=0"' --ipc=host -p 8200:8200 \
  -v "$CKPT":/checkpoint:ro \
  --restart unless-stopped \
  inflection/voxtral-rt-mstar:latest

until curl -sf http://127.0.0.1:8200/health >/dev/null; do sleep 2; done

curl -X POST http://127.0.0.1:8200/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "response_format=text"
```

Weights are not baked in: one image serves any Voxtral-Realtime checkpoint.
The image is ~15 GB and the server is ready ~2 s after start (weight load only
— there is no CUDA-graph capture to wait for, unlike the Qwen3-TTS image's
4–7 minutes).

| env | default | meaning |
|---|---|---|
| `VOXTRAL_MODEL_PATH` | `/checkpoint` | checkpoint directory (mounted) |
| `VOXTRAL_PORT` | `8200` | listen port |

The entrypoint checks for `config.json`, `model.safetensors` and `tekken.json`
and exits with a readable message rather than failing inside model loading.
`tekken.json` is load-bearing twice: it is the tokenizer, and it drives the
audio padding that aligns mel frames to text positions.

The build itself asserts that `numpy` stayed on 2.x and that `mistral_common`,
`python-multipart` and `soundfile` import — each has broken quietly at least
once, and a missing multipart parser turns every upload into a 400 that says
nothing about the cause.

---

## API

### `POST /v1/audio/transcriptions`

OpenAI's shape, matching what the vllm-realtime fork serves, so an existing
client needs only a new base URL.

| field | values | notes |
|---|---|---|
| `file` | audio container | any rate; resampled to 16 kHz, mixed to mono |
| `response_format` | `json`, `text`, `verbose_json` | default `json` |
| `model`, `language` | any | accepted and ignored, for compatibility |

```json
{"task":"transcribe","language":"","duration":10.38,
 "text":"这并不是告别。这是一个篇章的结束，也是新篇章的开始。",
 "segments":null,"words":null}
```

**`language` is always `""`.** The checkpoint has no language head. The
vllm-realtime fork fills this field by classifying the output text's Unicode
block, which cannot separate any two Latin-script languages; an empty string is
honest and a wrong ISO code is not. Run LID on the text if you need it.

`words` is `null` — word timestamps are not ported yet (see below).

Unsupported `response_format`, an empty upload, or an undecodable container all
answer **400**.

### `GET /health`, `GET /v1/models`

`/health` answers 503 while the model loads, 200 after.

---

## Verifying correctness

Parity is enforced by tests, not claimed in prose.

```bash
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest test/inflection/voxtral -v
```

| tier | needs | asserts |
|---|---|---|
| `test_config_and_cadence.py` | checkpoint config | the 12.5 Hz cadence and shipped shapes |
| `test_audio_frontend.py` | tokenizer | prompt layout, absolute mel floor, rate rejection |
| `test_reference_parity.py` | GPU + checkpoint | **generated token ids** vs recorded HF output |

The parity tests compare **token ids, not transcripts**, deliberately. Text
comparison hides the failure mode that actually occurred: a corrupt
time-conditioning buffer made the model emit the right words two frames early,
which still decoded to identical text on six of eight clips.

Reference output is recorded from `transformers`, in a different environment, by
a script that does not import M*:

```bash
~/workspace/vllm-realtime/.venv/bin/python \
  scripts/inflection/voxtral_record_reference.py \
  --model $CKPT --audio-dir test/inflection/voxtral/reference/audio \
  --out test/inflection/voxtral/reference/hf_reference.json
```

Recording it from the implementation under test would be circular.

### Measured parity

8 clips, English and Chinese, 3.5 s–25 s, bfloat16 on one B200:

| | result |
|---|---|
| transcript identical to reference | **8 / 8** |
| generated token ids identical | **8 / 8** |
| mel vs reference processor | max abs diff `0.000e+00` |
| prompt token ids vs reference | identical |

---

## Benchmarks

```bash
python -m benchmark.inflection.voxtral_bench \
  --url http://127.0.0.1:8200 --audio-dir /path/to/wavs \
  --concurrencies 1,2,4,8 --reps 3 --out out/voxtral
```

Writes `results.json`, a Plotly HTML and a matplotlib PNG.

Measured, one B200, 8 clips / 84 s of audio:

| concurrency | latency p50 | RTF p50 | aggregate xRT | success |
|---|---|---|---|---|
| 1 | 1102 ms | 0.17 | 6.1x | 100% |
| 2 | 3897 ms | 0.30 | 4.4x | 100% |
| 4 | 6026 ms | 0.42 | 5.6x | 100% |
| 8 | 12733 ms | 0.74 | 6.4x | 100% |

**Read this as a baseline to beat, not a result.** RTF rises roughly linearly
with concurrency because the bring-up server holds a lock and serves one request
at a time; aggregate throughput is therefore flat at single-stream speed. The
engine path is what changes this shape, and these numbers are recorded so that
change is measurable.

Single-stream RTF of 0.17 means a 10-second clip transcribes in 1.7 seconds.

---

## What is and is not done

**Done and verified**
- Model core, bit-exact against the HF reference (8/8 token-identical)
- Front end: mistral-common tokenization, ported mel (bit-identical)
- `POST /v1/audio/transcriptions` — json / text / verbose_json, 400s
- 15 tests across CPU and GPU tiers, with recorded reference fixtures
- Latency/RTF benchmark with charts

**Not done**
- **Walk Graph engine integration.** This is the main item. Voxtral is not in
  `MODEL_REGISTRY` and does not run under `mstar serve`; it has its own
  process. Porting it brings continuous batching, paged attention and CUDA
  graphs — the three things that gave Qwen3-TTS 7.8x lower latency than
  vllm-omni at concurrency 32. The shape fits M*'s model well: the audio tower
  is a one-shot prefill node (like the existing `whisper` encoder) and the text
  decoder is a standard AR loop whose per-step audio conditioning is the same
  "add a precomputed per-step vector" pattern the Qwen3-TTS talker already uses
  for `trailing_text_hidden`.
- **Streaming.** The architecture supports it natively — that is the point of
  the causal tower and the conv padding cache — but only offline transcription
  is wired. A `/v1/realtime` WebSocket matching the vllm-realtime protocol is
  the natural next surface.
- **Word timestamps.** The vllm-realtime fork derives them from the synchronous
  alignment (position *i* is *i*/12.5 seconds), which is unusually easy here —
  the alignment is structural, not inferred. `words` is `null` until then.
- Prometheus metrics, Grafana, Docker image, multi-GPU.

---

## Troubleshooting

**Every multipart request 500s with a pydantic "not fully defined" error, while
`/health` works** — a route module used `from __future__ import annotations`.
It stringifies annotations, and FastAPI resolves a route's annotations against
the *module* namespace, so `UploadFile` on a route defined inside a factory
becomes an unresolvable ForwardRef. Keep the FastAPI imports module-level and
that file free of the future import.

**Transcript is right but shifted early, or drops the last word** — the time
conditioning is wrong. `TimeEmbedding.inv_freq` is a non-persistent buffer, so
no checkpoint key restores it, and loading materialises the module with
`to_empty()`, which allocates buffer storage *without initialising it*. Nothing
reports missing. `test_time_conditioning_buffer_is_initialised` guards it.

**`ValueError: N weights were not found in the checkpoint`** — working as
intended. A partially loaded Voxtral produces a fluent, wrong transcript rather
than an error, so the loader refuses to return one. Check the key prefixes: the
checkpoint nests the decoder one level deeper (`language_model.model.layers`)
than the module tree.

**`Invalid mX.strides[0] ... expected to be divisible by 8`** — a non-contiguous
tensor reached M*'s fused RMSNorm. The audio embedder's `permute` produces
exactly this; it is followed by `.contiguous()` for that reason.

**`mistral-common` downgraded numpy and Qwen3-TTS broke** — install it with
`--no-deps` and the explicit list above.

**Server binds then immediately exits with "address already in use"** — a
previous instance survived. `pgrep -f voxtral_rt.serving.app` and kill it;
confirm the port is free before relaunching.
