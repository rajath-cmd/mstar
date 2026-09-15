# Qwen3-TTS on M* — build, run, verify

A drop-in replacement for the vllm-omni Qwen3-TTS server: same WebSocket
protocol, same HTTP API, same request surface. Point an existing client at it
and change nothing else.

- [Quick start](#quick-start)
- [Docker](#docker)
- [Bare metal](#bare-metal)
- [Configuration profiles](#configuration-profiles) — **the one choice that matters**
- [API](#api)
- [Word timestamps](#word-timestamps)
- [Observability](#observability) — Prometheus + Grafana
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
A `pointer_head.pt` beside them is the TFA alignment head. When present, word
timestamps work; when absent, `timestamp_type: "word"` returns no words rather
than an error. See [Word timestamps](#word-timestamps).

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
<- {"type":"timestamps","sentence_index":0,"word_alignment":{...}}
<- {"type":"session.done","total_sentences":1}
```

`timestamps` frames appear only when the session asked for them; with the
default `incremental` emission they also arrive *during* synthesis, interleaved
with the PCM. See [Word timestamps](#word-timestamps).

`{"type":"cancel"}` barges in and answers `{"type":"cancelled","sentence_index":N,"drained":M}`.

`voice.list` and `voice.delete` are **one-shot commands sent before
`session.config`**; they answer and close. Sent mid-session they are rejected as
unknown — matching vllm-omni, whose own docstring lists them as general messages
but whose implementation only handles them pre-config.

---

## Word timestamps

Set `timestamp_type: "word"` on REST, or in `session.config` on the WebSocket.
Requires a `pointer_head.pt` in the checkpoint directory; the startup log says
`loaded AlignmentPointerHead ...` when one was found.

Capture is opt-in per request, and deliberately so: it copies every decoded
frame's hidden state to host memory and holds it for the life of the request.
Traffic that never asks for timestamps pays nothing.

### REST

```bash
curl -X POST http://127.0.0.1:8100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"The quick brown fox jumps over the lazy dog.",
       "voice":"alexandra","timestamp_type":"word"}' | jq .timestamp_info
```

```json
{"word_alignment": {
  "words": ["The","quick","brown","fox","jumps","over","the","lazy","dog."],
  "word_start_time_seconds": [0.0,0.24,0.56,0.96,1.44,1.92,2.16,2.24,2.72],
  "word_end_time_seconds":   [0.24,0.56,0.96,1.44,1.92,2.16,2.24,2.72,3.28]}}
```

Times are relative to the returned audio.

### WebSocket

Times are **turn-relative**: t=0 is the turn's first `audio.start`, and the
server-inserted inter-chunk silence is included. A client concatenating the
binary frames it received can seek to a word by its timestamp without tracking
chunk boundaries.

By default words go out **incrementally**, while the chunk's audio is still
generating:

```
audio.start
  timestamps {partial:true}   The quick brown fox jumps
audio.done
  timestamps                  over the lazy dog.
session.done
```

That matters for barge-in. With `chunk` emission the client is blind for the
whole synthesis of a chunk — which is exactly the window an interrupt has to
splice into. Two invariants hold on the wire, and a client may rely on both:

- **Append-only** — a word is never sent twice, so blindly extending a list is
  correct.
- **Monotone** — a word never starts before one already sent.

These are enforced at the boundary by filtering the final frame against what
already went out, *not* by trusting the commit horizon. The horizon is
beam-pruned, so the committed prefix and the final decode occasionally disagree
(measured 3 turns in 40 on tag-heavy text); a positional slice would put a
duplicate word on the wire, which an appending client cannot recover from.

| env | default | meaning |
|---|---|---|
| `VLLM_TTS_WORD_TS_EMISSION` | `incremental` | `incremental` or `chunk` |
| `VLLM_TTS_WORD_TS_POLL_INTERVAL_S` | `0.15` | partial poll cadence |
| `VLLM_TTS_WORD_TS_COMMIT_MARGIN` | `10.0` | beam margin before a word is published |

Names are vllm-omni's, unchanged, so a deployment's environment carries over.
Per-session override: `"timestamp_emission": "chunk"` in `session.config`.

`tts_alignment_commit_revision_total` **must stay zero**. Non-zero means a
published word was contradicted by the final decode. The wire stays uncorrupted
either way, but a revision can cost a word in the final frame; raise
`VLLM_TTS_WORD_TS_COMMIT_MARGIN` if it fires on ordinary prose.

### Accuracy

Not bit-identical to vllm-omni, and it cannot be — the model is sampled, and two
runs of the *same* server on the same text give different audio and different
word times. What matches is the timing structure. Same checkpoint, same
sentence, word durations in seconds:

| | The | quick | brown | fox | jumps | over | the | lazy | dog. |
|---|---|---|---|---|---|---|---|---|---|
| M* | 0.24 | 0.32 | 0.40 | 0.48 | 0.48 | 0.24 | 0.08 | 0.48 | 0.56 |
| vllm-omni | 0.16 | 0.32 | 0.24 | 0.48 | 0.48 | 0.24 | 0.08 | 0.48 | 0.48 |

`test_alignment_parity.py` asserts the statistical form of this: word count,
monotonicity, coverage of the script, and bounds against the audio duration.

---

## Observability

`GET /metrics` serves Prometheus exposition. **Series names are frozen to
vllm-omni's**, so an existing dashboard, alert or scrape config keeps working
across a backend swap — that is what makes the image drop-in rather than merely
API-compatible.

```bash
curl -s http://127.0.0.1:8100/metrics | grep -E '^tts_'
```

| series | type | use |
|---|---|---|
| `tts_ttfa_seconds` | histogram | time to first **PCM**, never to `audio.start` |
| `tts_rtf` | histogram | generation / audio duration; >1 means slower than realtime |
| `tts_generation_seconds` | histogram | end-to-end synthesis |
| `tts_audio_duration_seconds` | histogram | audio produced |
| `tts_requests_total` | counter | by `endpoint`, `voice`, `status` |
| `tts_active_requests` | gauge | in-flight |
| `tts_streaming_sessions` | gauge | open WebSockets |
| `tts_cancel_total` | counter | barge-ins |
| `ws_close_reasons_total` | counter | by `reason` |
| `tts_alignment_head_loaded` | gauge | 1 when word timestamps are possible |
| `tts_alignment_commit_revision_total` | counter | **must stay 0** |

### Grafana

Import `docs/inflection/qwen3-tts/grafana/qwen3-tts-mstar.json` and pick a
Prometheus datasource. Ten panels, ordered by what you check first during an
incident: TTFA and RTF, then load, then alignment.

```yaml
# prometheus.yml
scrape_configs:
  - job_name: qwen3-tts
    static_configs: [{targets: ['qwen3-tts:8100']}]
```

A note on where numbers come from: the engine runs in a **separate process**
from the API server, so a counter the worker increments is invisible to the
registry uvicorn scrapes. Gauges whose truth lives in the worker are restated
at scrape time from the same facts the worker acts on; worker-only counters
(`tts_alignment_register_total`) legitimately read 0 in the scrape. The
alignment *outcome* signals — revisions, pops, partials — cross the process
boundary through the same disk drop the words do, so they are real.

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


### Measured results

One B200 per arm, same checkpoint (`sft-cv-13l-14pv-ticr-1e-13l-fulldata_17946`),
arms interleaved per concurrency level, M* on the low-latency profile.

REST batch, `tts_ab`:

| concurrency | M* p50 | omni p50 | M* xRT | omni xRT | M* RTF | omni RTF |
|---|---|---|---|---|---|---|
| 1  | 376 ms | 1236 ms | 11.2x | 3.9x | 0.09 | 0.26 |
| 4  | 672 ms | 1784 ms | 21.6x | 6.8x | 0.14 | 0.41 |
| 8  | 639 ms | 3436 ms | 35.2x | 9.2x | 0.15 | 0.67 |
| 16 | 870 ms | 5214 ms | 54.7x | 11.3x | 0.19 | **1.20** |
| 32 | 1178 ms | 9147 ms | **86.1x** | 13.0x | 0.26 | **2.17** |

WebSocket TTFA, `tts_ws_ttfa`:

| concurrency | M* p50 | M* p90 | omni p50 | omni p90 |
|---|---|---|---|---|
| 1  | 44 ms | 52 ms | 39 ms | 45 ms |
| 4  | 93 ms | 113 ms | 85 ms | 92 ms |
| 8  | 120 ms | 136 ms | 105 ms | 146 ms |
| 16 | 134 ms | 161 ms | 202 ms | 224 ms |
| 32 | **180 ms** | 198 ms | 495 ms | 531 ms |

Read honestly:

- **At concurrency 1 the two are equivalent on TTFA** (39 vs 44 ms — inside
  run-to-run noise, and vllm-omni is nominally ahead). Anyone quoting a
  single-stream latency win is quoting noise.
- **The difference is what happens under load.** M* holds RTF under 1.0 through
  c=32; vllm-omni crosses 1.0 between c=8 and c=16, meaning it can no longer
  keep up with realtime playback. For a voice agent that is the capacity
  ceiling, and it arrives long before any error does.
- At c=32 M* is 7.8x lower batch latency, 6.6x throughput, 2.75x lower TTFA.
- Both arms 100% success at every level.

Raw data and charts for this run are committed under
`docs/inflection/qwen3-tts/bench-2026-09-15/` — `tts_ab-results.json`, `ttfa-results.json`
and the matplotlib PNGs. The interactive Plotly HTML is not committed (it is
regenerated by any run; it loads plotly.js from CDN rather than bundling it).
Codec-cadence sweep: `BENCH-2026-09-15.md`.

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
| `test_alignment_parity.py` | word timestamps — presence, monotonicity, coverage, bounds |

34 assertions, **34 green against both servers** — including against the
container image, which is the drop-in claim's actual proof rather than a claim
about the source tree.

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

**`timestamp_info` is `null` / no `timestamps` frames** — the checkpoint ships
no `pointer_head.pt`, or the request did not ask. Check the startup log for
`loaded AlignmentPointerHead` and `tts_alignment_head_loaded` in `/metrics`.
Both servers answer the same way — no words, not an error.

**WebSocket handshake returns 404 while REST works** — the image has no
WebSocket library, so uvicorn logs `Unsupported upgrade request` at WARNING and
refuses every upgrade. Fixed in this image (`websockets` is pinned, asserted at
build, and preflighted at container start, which now *fails to boot* rather than
serving a half-working API). If you see it on an older image, rebuild. A source
checkout never shows this, because pytest's own dependencies supply
`websockets`.

**Requests submit but never execute, GPU at 0%** — an orphaned worker from a
previous hard kill is still holding the conductor's IPC resources. `kill -9` on
a server's parent leaves its workers reparented to init. Kill the API process
*and* its children, then any leftover process still holding the GPU:

```bash
SRV=$(pgrep -f 'mstar serve qwen3_tts.*8100' | head -1)
kill -9 $(pgrep -P $SRV) $SRV
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  ps -o cmd= -p $p | grep -q 'workspace/mstar' && kill -9 $p
done
```

Never `pkill -f` a pattern that matches your own shell's command line — it kills
the shell (exit 144) and leaves exactly this mess.

**Server exits during startup, no error** — usually OOM from co-tenancy. Each
instance reserves a large share of its GPU; give each its own device.
