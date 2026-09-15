# Making M* the best engine for Qwen3-TTS and Voxtral-Realtime

What VoxServe does that M* does not, what it is worth, and what to build next.

Sources: [VoxServe: Streaming-Centric Serving System for Speech Language
Models](https://arxiv.org/abs/2602.00269) (Kamahori, Lee, Jha, Kadekodi, Wang,
Krishnamurthy, Kasikci), the [launch
post](https://vox-serve.github.io/2025/09/29/introducing-vox-serve.html), and
the source at [vox-serve/vox-serve](https://github.com/vox-serve/vox-serve).
Keisuke Kamahori is first author on both VoxServe and M*, so these are the same
people's conclusions about the same problem — worth weighting accordingly.
VoxServe reports **10–20x higher throughput than existing implementations at
comparable latency**, and ships a `qwen3-tts` model at **40 ms TTFA on an H100**.

All M* numbers below are measured on this machine, one B200 per server,
checkpoint `sft-cv-13l-14pv-ticr-1e-13l-fulldata_17946`.

---

## Where M* already stands

Worth stating plainly, because the remaining gaps are narrower than the paper's
headline numbers suggest against a naive baseline.

| | M* today | VoxServe |
|---|---|---|
| Continuous batching | yes (`can_batch`, batch 32) | yes |
| Paged attention | yes (FlashInfer) | yes |
| CUDA graphs, decode | yes, per batch size | yes |
| CUDA graphs, depth loop | yes, piecewise region | yes |
| CUDA graphs, vocoder | yes, one padded shape | yes, one graph per exact size |
| Async scheduling overlap | partial (`plan_executor`) | full (`gather(model, scheduling)`) |
| Short first chunk | **yes, as of this branch** | yes (`first_chunk_frames`) |
| Streaming-aware priority | **no** | yes (`is_pressing`) |
| Multi-chunk vocoder batching | **no** | yes |
| Delayed stop decision | **no** | yes (NanoFlow) |
| LM/vocoder disaggregation | no | yes (optional) |

Measured against vllm-omni, M* already wins at every concurrency:

| | M* | vllm-omni |
|---|---|---|
| WS TTFA p50, c=1 | **35 ms** | 41 ms |
| WS TTFA p50, c=32 | **162 ms** | 503 ms |
| RTF p50, c=32 | **0.26** | 2.12 |
| throughput, c=32 | **67–86x RT** | 13x RT |

---

## 1. Short first chunk — DONE, −33% TTFA

The one idea M* was missing outright. TTFA is set by how many frames must
decode before *any* audio leaves; throughput is set by the steady-state chunk.
A single `chunk_frames` conflates them.

Shipped in `configs/inflection_qwen3tts_lowlatency.yaml` as
`codec_first_chunk_frames: 1` + `codec_growing_left_context: true`:

| | TTFA p50 |
|---|---|
| chunk=2 (previous default) | 45.6 ms |
| chunk=2 + first_chunk=1 | **30.6 ms** |
| vllm-omni | 39.2 ms |

Throughput at c=32 unchanged (84.5x vs 86.1x realtime) because the steady-state
chunk never moved. Audio was transcribed to confirm the opening word is not
clipped.

This also required `GrowingLeftContextChunkPolicy`. The old policy took its full
left context from the first pop, so it had to advance by `chunk - left_context`
and thus required `chunk > left_context` — which is why `chunk_frames=1` was
unreachable (it forces `left_context=0`, a configuration that emits no audio at
all). The new policy grows the context from 0, exactly as vllm-omni's
`chunked_decode` does.

---

## 2. Streaming-aware priority scheduling — the big one

**The paper's central idea, and M* has no equivalent.**

> "speed improvements beyond playback rate have diminishing returns (except for
> the first chunk)"

A client that already holds 3 seconds of buffered audio gains *nothing* from
receiving the next chunk sooner. VoxServe turns that slack into throughput:

```python
# vox_serve/scheduler/online.py
req.is_pressing = current_time >= latest_chunk_start_time - 1.0  # 1s buffer
```

Then `_select_lm_requests` batches **prefill first** (always critical), then
**pressing decodes**, then *piggybacks* non-pressing decodes into any remaining
slots. Same for the detokenizer, which only runs at all if some pressing
request needs it.

The effect is that batch occupancy stops being a function of arrival times and
becomes a function of who actually needs service now. The blog post attributes
**~15% additional throughput** to scheduling alone, on top of everything else.

**What it needs in M\*:** the conductor currently selects work by readiness, with
no notion of a per-request playback deadline. Two pieces:

1. Per-request playback clock — first-chunk send time plus the sum of emitted
   chunk durations. The WS handler already tracks turn audio duration for
   word-timestamp offsets, so the state largely exists; it needs to reach the
   scheduler.
2. A priority input to batch selection, so `can_batch` / batch assembly can
   prefer pressing requests and backfill with the rest.

**Why it matters most for us:** it is the only change here that raises the
*concurrency ceiling* rather than trimming latency. M* crosses RTF 1.0 somewhere
past c=32; this is what pushes that out.

**Caveat worth measuring first:** the win scales with how much slack exists. At
RTF 0.26 M* generates ~4x faster than playback, so slack is large and the
headroom should be real — but it should be measured on a mixed arrival pattern,
not a synchronised ladder, because a synchronised ladder makes every request
pressing at the same moment and hides the effect entirely.

---

## 3. Batch multiple chunk positions per vocoder call

VoxServe's detokenizer scheduler builds, per request, a *list* of chunk indices
(`req.next_audio_decode_idx = audio_idx_list`) and proportionally allocates a
global detokenize budget across requests when oversubscribed. One vocoder
invocation covers several chunks and several requests.

M*'s Codec node decodes one chunk per request per step.

This matters precisely because of change #1: small chunks are good for latency
and bad for per-call efficiency, and batching chunk positions is what buys back
the efficiency. It is also the natural fix for the cost structure noted below in
§4 — the two should be designed together.

---

## 4. Exact-size vocoder CUDA graphs instead of padding to max

M* pads every codec input to `chunk + left_context` so a single captured graph
serves every call (`CodecSubmodule.prepare_inputs`). vllm-omni instead captures
one graph per exact size (`compute_exact_sizes`, T=1..21) and pads nothing.

With today's shipping profile the live sizes are just {1, 3}, so padding waste
is small. It becomes significant the moment the context is raised: uniform
`chunk=1, left_context=15` — what vllm-omni actually ships — would decode 16
frames for every 1 emitted under M*'s padding scheme. That is roughly 10x the
vocoder work per second of audio, and it is a large part of why vllm-omni
achieves 13x realtime where M* achieves 67–86x.

So this is not urgent, but it is the gate on ever adopting a *large* left
context for quality. Capture exact sizes, and the quality/latency/throughput
knobs become independent.

---

## 5. Delayed stop decision (NanoFlow)

VoxServe "adopts an asynchronous execution pipeline ... leveraging a delayed
stop-decision mechanism (as proposed in NanoFlow)".

M*'s `TalkerSubmodule.check_stop` does:

```python
token = int(outputs["layer0_codes"][0].item())   # GPU -> CPU sync
```

once per request per decode step. At 12.5 Hz across 32 concurrent requests that
is a lot of synchronisation on the critical path. The comment in `check_stop`
notes it runs off the GPU execution thread, which limits the damage — but the
sync still serialises against the stream.

Deferring the stop decision (let decode run ahead, reconcile stops a step or two
later, discard the overrun frames) removes the per-step sync. Contained change,
measurable win, no protocol impact. Good next item after #2.

---

## 6. Full async scheduling overlap

```python
model_result, scheduling_result = await asyncio.gather(run_model(), run_scheduling())
```

VoxServe selects the *next* batch while the GPU executes the *current* one, and
awaits the previous LM task inside the scheduling half so the GPU is never idle
waiting for Python.

M* has `plan_executor` ("speculative `plan()` pre-runs on a dedicated thread"),
so this is partly closed already. Worth profiling before building: measure the
gap between consecutive decode kernels at c=32 and see how much is CPU.

---

## 7. Disaggregation (optional, costs a GPU)

`DisaggregationScheduler` runs the LM on GPU 0 and the detokenizer on GPU 1 as
two independent async loops joined by queues. For Qwen3-TTS the codec is a 114M
decoder competing with the Talker for SMs; splitting them removes that
interference at the cost of a second GPU. Only worth it once #2 and #3 have
taken the single-GPU case as far as it goes.

---

## What this means for Voxtral-Realtime

Voxtral has **no vocoder**, so #3, #4 and #7 do not apply. Its ranking is
different and much simpler:

1. **Get it onto the Walk Graph at all.** This is the whole game. The bring-up
   server serves one request at a time behind a lock — no continuous batching,
   no paged attention, no CUDA graphs. Measured: RTF 0.17 single-stream, rising
   roughly linearly to 0.74 at concurrency 8, with aggregate throughput flat at
   ~6x realtime. Everything else is noise next to this.

   The shape fits M* well. The audio tower is a one-shot dense prefill — the
   same role `whisper`'s encoder already plays in this repo. The text decoder is
   a standard AR loop whose only unusual feature is that each step adds a
   precomputed per-step vector to the token embedding, which is exactly the
   pattern `TalkerSubmodule` already runs for `trailing_text_hidden`.

2. **Streaming-aware priority (#2) applies directly and for the same reason.**
   Streaming ASR has a realtime deadline too: a session only needs transcript
   out as fast as audio comes in. A request whose audio has not yet arrived is
   by definition not pressing. The `is_pressing` machinery is shared.

3. **Delayed stop decision (#5) does not apply.** Voxtral's decode length is
   fixed by the audio (`audio_seconds x 12.5`), not by an EOS token, so there is
   no stop decision to sync on — the loop bound is known before it starts. That
   is a genuine architectural advantage over TTS and should be exploited: the
   whole decode can be planned up front.

4. **Input streaming.** VoxServe has an `InputStreamingScheduler`; Voxtral's
   causal tower and conv padding cache are built for it. A `/v1/realtime`
   WebSocket matching the vllm-realtime protocol is the natural surface once the
   engine path lands.

---

## Recommended order

1. ~~Short first chunk~~ — done, −33% TTFA, shipped.
2. **Voxtral onto the Walk Graph.** Largest absolute win available (flat →
   batched concurrency), and it is a port rather than a research question.
3. **Streaming-aware priority scheduling.** Shared by both models, raises the
   concurrency ceiling rather than trimming latency. Measure on mixed arrivals.
4. **Delayed stop decision.** Contained, TTS-only, removes a per-step sync.
5. **Multi-chunk vocoder batching + exact-size graphs.** Together, and only once
   a profile actually wants a large left context.
6. Disaggregation, if a second GPU per replica is ever acceptable.

## One operational fix that gates all of this

`mstar serve` defaults its ZMQ IPC prefix to `/tmp/mstar_$USER/` — **per user,
not per server**. Two servers started by the same user bind the same
`worker_0.ipc` / `conductor.ipc` / `api_server.ipc` and steal each other's
messages: a request POSTed to B is executed by A's engine under A's config
while B waits forever. Nothing errors; both report healthy and only requests
hang, which reads exactly like a model bug.

This makes any A/B comparison silently wrong, and it blocks serving Qwen3-TTS
and Voxtral-RT on one machine. `scripts/inflection/launch_mstar_qwen3_tts.sh`
now sets `--socket-path-prefix` per port. **The default in `mstar/cli/main.py`
should be changed upstream to include the port**, so nobody has to know this.
