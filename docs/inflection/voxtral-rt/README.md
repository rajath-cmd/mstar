# Voxtral-Realtime on M* — build, run, verify

Vanilla `Voxtral-Mini-4B-Realtime-2602` (no auxiliary heads) running on M*,
with an OpenAI-compatible transcription endpoint.

**Status: running on M*'s Walk Graph engine, verified correct.** 8 of 8 test
utterances transcribe identically to the HuggingFace reference, English and
Chinese, 3.5 s to 25 s — through the real engine, with paged attention,
continuous batching and CUDA-graph decode. See
[What is and is not done](#what-is-and-is-not-done).

- [How the model works](#how-the-model-works) — read this first
- [Quick start](#quick-start)
- [Two ways to run it](#two-ways-to-run-it)
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

## Two ways to run it

There are two servers in this branch, and the difference matters.

| | engine path (**use this**) | bring-up server |
|---|---|---|
| launch | `scripts/inflection/launch_mstar_voxtral_rt.sh` | `python -m mstar.model.voxtral_rt.serving.app` |
| runs on | M*'s Walk Graph engine | a plain FastAPI loop |
| batching | continuous, paged attention, CUDA graphs | one request at a time behind a lock |
| endpoint | `POST /generate` (M*'s multimodal API) | `POST /v1/audio/transcriptions` (OpenAI shape) |
| throughput @ c=16 | **133x realtime** | 7x realtime |
| latency p50 @ c=16 | **1091 ms** | 24276 ms |

```bash
CKPT=/mnt/data/models/audio/stt/voxtral-rt/pretrained/Voxtral-Mini-4B-Realtime-2602
MSTAR_MODEL_PATH=$CKPT GPUS=0 scripts/inflection/launch_mstar_voxtral_rt.sh 8300

curl -X POST http://127.0.0.1:8300/generate \
  -F "files=@sample.wav" -F "input_modalities=audio" \
  -F "output_modalities=text" -F "streaming=false"
```

Chunks come back base64-encoded, one per decoded token; concatenate and decode
them for the transcript. The OpenAI-shaped `/v1/audio/transcriptions` wrapper
has not been ported onto the engine path yet — that is the next piece of work,
and it is a routing change, not a model one.

The bring-up server is kept because it is the trusted oracle. `model.py` is a
standalone PyTorch implementation of the same weights, token-identical to the
HF reference, and `weights.py` guarantees both paths load the checkpoint the
same way. Bisecting the engine port against it is what found the KV-cursor bug
below; without a reference implementation that bug is close to undiagnosable.

### Measured

One B200, 8-clip corpus (3.5 s – 25 s), 3 reps per level, 100% success at every
level for both arms:

| concurrency | latency p50 | | RTF p50 | | throughput | |
|---|---|---|---|---|---|---|
| | bring-up | **engine** | bring-up | **engine** | bring-up | **engine** |
| 1 | 1369 ms | **674 ms** | 0.14 | **0.06** | 7.0x | 6.6x |
| 2 | 2921 ms | **813 ms** | 0.22 | **0.07** | 6.9x | **29.6x** |
| 4 | 5170 ms | **820 ms** | 0.43 | **0.08** | 6.7x | **42.9x** |
| 8 | 11983 ms | **738 ms** | 0.85 | **0.07** | 7.0x | **97.7x** |
| 16 | 24276 ms | **1091 ms** | **1.74** | **0.11** | 7.0x | **132.9x** |

Two things to read off it. The serialized server crosses RTF 1.0 between c=8
and c=16 — past that it cannot keep up with the audio — while the engine stays
near 0.1 throughout. And single-stream latency HALVES (1369 → 674 ms), which is
not batching at all: that is CUDA-graph decode removing per-step launch
overhead.

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
- **Walk Graph engine integration** — registered as `voxtral_rt`, runs under
  `mstar serve`, with paged attention, continuous batching and CUDA-graph
  decode. 8/8 transcripts still identical to the reference through the engine;
  19x throughput and 22x lower latency at concurrency 16.
- `POST /v1/audio/transcriptions` on the bring-up server — json / text /
  verbose_json, 400s
- 15 tests across CPU and GPU tiers, with recorded reference fixtures
- Latency/RTF benchmarks and charts for both paths

**Not done**
- **The OpenAI-shaped endpoint on the engine path.** The engine serves
  `/generate`; `/v1/audio/transcriptions` still only exists on the bring-up
  server. This is a routing change, not a model one, and it is the next piece
  of work.
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

**The model loads, prefills perfectly, then transcribes nothing but padding**
— a decoder running M*'s shared `Attention` must advance the resource cursors
itself: `attend.bind_step()` once, then `attend.set_layer_idx()` per layer.
Forget the latter and all 26 layers read and write layer 0's KV pages. The
deception is that PREFILL still comes out bit-exact — with an empty cache the
attention is computed entirely from the q/k/v of that same call, so the layer
index never matters. Only the first DECODE step, the first read of cached keys,
goes wrong. Compare prefill AND decode logits against `model.py` (the oracle)
when this shape of bug appears.

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
