# M* Qwen3-TTS: drop-in parity with vllm-omni — program spec

**Status:** DRAFT — blocked on a disclosure decision (see [Gate 0](#gate-0-repository-visibility-blocking)).
**Branch:** `inflection/qwen3-tts` (the working "main" for this effort)
**Fork:** `rajath-cmd/mstar`, forked from `mstar-project/mstar`, at `ddb38a7`
**Reference implementation:** `~/workspace/vllm-omni` @ `raj/ws-incremental-word-timestamps` (PR #43)
**Reference checkpoint:** `/mnt/data/models/audio/tts/qwen3-tts/sft-cv-13l-14pv-ticr-1e-13l-fulldata_17946/checkpoint-final`

---

## Goal

Make an M*-served Qwen3-TTS a **drop-in replacement** for our vllm-omni image: same
WebSocket protocol, same HTTP API, same observability, same word-timestamp quality —
at equal or better latency and throughput.

"Drop-in" is the acceptance bar: point `inf2-inference`'s pipecat Qwen3-TTS service at
the M* endpoint, change nothing else, and the voice agent behaves identically.

---

## Why M* at all

M* is a Walk-Graph runtime: a model is a dataflow graph of components and a request is
a *Walk* over it. For Qwen3-TTS that maps cleanly onto what the model already is —
an autoregressive Talker feeding a streaming audio codec — and M* offers per-component
fast paths that our current two-stage vLLM setup does not expose: per-component TP × SP
meshes, sliding-window chunk streaming for the codec, component-level disaggregation
with pluggable transport, and CUDA-graph capture per walk.

That is the hypothesis. **This program does not assume it.** Phase 6 measures it, and
if M* does not beat vllm-omni on TTFA at the concurrencies we serve, the honest outcome
is a documented negative result, not a migration.

---

## Gate 0: repository visibility (BLOCKING)

`rajath-cmd/mstar` is **public** (`visibility=public`, verified 2026-09-15 via the
GitHub API). Everything pushed to it is world-readable.

Phase 3 of this program ports the temporal-alignment / TFA pointer-head work. Per the
decision recorded earlier in this workstream, that work is on an **internal tech report →
provisional patent → Interspeech/ICASSP** path, and the stated constraint is **no public
disclosure before the provisional is filed**. Porting it into a public fork is public
disclosure — it would start (or blow) the clock.

Also world-readable if pushed as-is: internal checkpoint paths and run names, the
serving topology, and the measured performance characteristics of our production stack.

**Nothing from Phases 1–7 lands on this remote until one of these is chosen:**

| Option | Action | Cost |
|---|---|---|
| **A (recommended)** | Make `rajath-cmd/mstar` private, or re-fork into the `InflectionAI` org as a private repo and re-point `origin` | Loses the GitHub fork relationship on a private re-fork; upstream syncs become a remote-tracking merge instead (already configured: `upstream` → `mstar-project/mstar`) |
| **B** | Keep public, and hard-split: Phases 1, 2, 4, 5, 6, 7 public; Phase 3 (alignment) stays in a private overlay repo mounted at build time | Two repos, a seam in CI, and the image is no longer buildable from one clone — directly conflicts with the "new user builds their own image" goal |
| **C** | Keep public, publish everything, accept disclosure | Forfeits the provisional. Requires an explicit, informed decision from whoever owns that call — not a default |

The branch and its protection rules are already in place; they disclose nothing beyond
the already-public upstream. All plan and design documents are held **locally** until
this is resolved.

---

## Current state, measured

Verified by reading the fork at `ddb38a7` on 2026-09-15.

### What M* already has

| Piece | Location | Notes |
|---|---|---|
| Qwen3-TTS model | `mstar/model/qwen3_tts/` | 2,135 LOC: `config.py`, `qwen3_tts_model.py` (827), `submodules.py` (987) |
| Walk graph | `qwen3_tts_model.py:277-343` | `talker_prefill` → `talker_decode` (Loop) → `codec_chunk` |
| Deployment config | `configs/qwen3tts.yaml` | Talker + Codec node groups, both rank 0, `flashinfer_backend: fa2` pinned |
| Checkpoint reader | `config.py` | Reads `config.json` + `generation_config.json` + `speech_tokenizer/config.json` — **the exact layout our checkpoint ships** |
| REST surface | `api_server/openai/router.py` | `/v1/models`, `/v1/chat/completions`, `/v1/audio/speech`, images, videos |
| Native surface | `api_server/entrypoint.py` | `/generate`, `/health` |
| Benchmark harness | `benchmark/` | `runner.py`, `request.py` (`OursOpenAI` client), `SeedTTSDataset` |
| CI | `.github/workflows/ci.yml` | jobs: `build` (ruff), `dynamo-smoke`, `rust-transport` |

### What M* does not have

| Gap | Evidence | Phase |
|---|---|---|
| **`qwen3_tts` is not in `ADAPTER_REGISTRY`** | `adapters.py:454-462` lists bagel, qwen3_omni, orpheus, cosmos3×3, wan22. `get_adapter("qwen3_tts")` → `None`. The base `speech_to_request` raises `NotImplementedError`. **`/v1/audio/speech` is unusable for Qwen3-TTS today.** | 1 |
| Impoverished `SpeechRequest` | `protocol.py:45` has 9 fields (input, model, voice, response_format, speed, stream, temperature, top_p, seed). vllm-omni's has 19 (`sample_rate` is WS-session-only, not REST). | 1 |
| No voices endpoints | No `/v1/audio/voices`, no upload/delete/list | 1 |
| **No WebSocket TTS endpoint** | Only route-level grep hit is `supports_realtime: bool = False` (`adapters.py:186`), an unused flag | 2 |
| No text chunker | No streaming/sentence/tag-aware segmentation | 2 |
| No word timestamps | No alignment head, no Viterbi, no `pointer_head.pt` loading | 3 |
| **No Prometheus / OTel / `/metrics`** | Sole repo-wide hit is `engine/resources/kv/transfer.py` | 4 |
| No serving Dockerfile | Only `mstar/integrations/dynamo/docker/*` | 7 |
| Not installed locally | No `.venv` in the checkout | 1 |

### The alignment seam (de-risked early, deliberately)

Phase 3 is the highest-risk phase, so its feasibility was checked **before** planning
rather than discovered mid-build.

`TalkerSubmodule._run_frame` (`submodules.py`) does:

```python
hidden = self.model(input_embeds, label="main")
last_hidden = hidden.index_select(0, last_token_indices)
...
return {"talker_input_embeds": codec_embed_sum,
        "codec_tokens": all_codes,
        "new_token": layer0_codes}
```

It returns a **named-tensor dict**, and `TalkerSubmodule.postprocess(request_id, ...)`
runs per request per frame. That pair is a direct analogue of vllm-omni's
`gpu_model_runner` → `registry.accumulate_decode` seam, and it is *cleaner*: M* hands us
a per-request hook with the frame's tensors already routed, where vllm-omni had to slice
`query_start_loc` out of a batched aux tensor.

**Open sub-risk.** Our trained head reads **layer 3**, not the final layer
(`pointer_head.pt`: `hidden_size=2048, proj_size=256, layer=3, head_type=mlp`).
`self.model(...)` returns only the final hidden state. vLLM exposed intermediates via its
Eagle3 surface (`aux_hidden_state_layers`); M*'s Talker backbone has no equivalent yet.
Phase 3 Task 1 is a spike to add one, and it is the gate for the whole phase. Prefill-side
text hidden states are needed too (the head pools per-word keys over text positions), so
the spike must cover `talker_prefill` as well as `talker_decode`.

---

## The parity contract

"100% parity" needs an operational definition, because one form of it is **provably
impossible**.

### What cannot be asserted, and why

Byte-identity and time-identity between the two servers are unavailable. Measured on the
vllm-omni server on 2026-09-04: two consecutive `chunk`-mode turns on the *same text*
produced **different audio SHA-256 and different word times**. The WS path exposes no
seed and its sampling is not greedy, so run-to-run variance swamps any cross-server
comparison. Any spec demanding bit-exact output would be untestable and would be quietly
abandoned the first time someone ran it.

So parity is defined at four levels, three of which are exactly assertable.

### L1 — Protocol parity (exact)

The wire is identical. Frame types, their ordering, field names, field types, which
fields are present vs absent, HTTP status codes, and error payload shapes.

Asserted by a **differential conformance suite** (Phase 1 Task 1) that runs the *same*
assertions against both servers, plus **golden traces** captured from vllm-omni and
replayed structurally against M*.

Specifically, for WS `/v1/audio/speech/stream`:
- Client→server: `session.config`, `input.text`, `input.done`, `cancel`, `voice.delete`, `voice.list`
- Server→client: `audio.start`, binary PCM frames, `audio.done`, `timestamps`, `session.done`, `cancelled`, `error`, `voice.registered`, `voice.deleted`, `voice.list`
- `timestamps` carries `{type, sentence_index, partial?, word_alignment:{words, word_start_time_seconds, word_end_time_seconds}}`; `partial` is **absent** (not `false`) on the final frame

For REST `/v1/audio/speech`: all 19 request fields accepted, the raw-bytes vs
JSON-envelope response switch on `timestamp_type`, and the batch (list `input`) path.

### L2 — Behavioral parity (exact)

Same input ⇒ same *structure* of output, independent of sampling:

- **Same word sequence.** The word list comes from prompt tokenisation, not sampling, so it is stable and comparable across servers. Tags (`<laughs/>`, `[sighs]`, `<pause=0.5s/>`) appear as literal entries in `words` — verified on vllm-omni — and must on M* too.
- **Timestamps are additive and monotone.** Every word exactly once, in voicing order, one turn timeline. No duplicates, no reordering.
- **Turn-relative timing.** `t=0` is the turn's first `audio.start`; inter-chunk silence is included in the offset. Verified on vllm-omni: a 3-chunk turn ran 0.00→4.88, 5.08→9.08, 9.28→12.88 s with a 0.20 s pause, total PCM 12.88 s == last word end.
- **Rate independence.** Word times are seconds regardless of `sample_rate` (verified at 8 kHz: 3.04 s of PCM, last word ends 3.04 s).
- **Identical degradation.** No `pointer_head.pt` ⇒ no `timestamps` frames, audio unaffected. Corrupt partial ⇒ frame skipped, end-of-chunk frame still complete. Cancel ⇒ words already sent are retained, `cancelled` carries `{sentence_index, drained}`.

### L3 — Quality parity (statistical, with a stated tolerance)

Word-timestamp accuracy on the blind30 gold set must be within tolerance of vllm-omni's
measured numbers, not merely "close":

- median AE, mean AE, `pct_within_80ms`, `match_rate`
- Acceptance: M* within the 95 % bootstrap CI of the vllm-omni arm, per metric
- Audio quality: WER parity on the same corpus via the existing ASR path

### L4 — Performance (statistical, directional)

This is the *reason* for the port, so it is a target, not just a floor:

- TTFA (p50/p90/p99), RTF, throughput (× realtime), success rate, at concurrency 1/4/8/16/32
- **Floor:** no regression vs vllm-omni on any metric at any concurrency
- **Target:** measurably better TTFA at concurrency ≥ 8
- Apples-to-apples: same client, same corpus, same concurrency ladder, same checkpoint, same GPU, interleaved arms to cancel drift

---

## Phase roadmap

Each phase is an independently reviewable, independently shippable unit with its own
plan document. Gates are hard: a failed gate stops the phase rather than being worked
around.

| Phase | Deliverable | Gate |
|---|---|---|
| **0** | Branch, protection, this spec, conformance-suite skeleton, golden traces from vllm-omni | Repo visibility resolved (Gate 0); M* installs and serves our checkpoint over `/generate` |
| **1** | `Qwen3TTSAdapter`; full `SpeechRequest`; voices endpoints; REST conformance green | REST differential suite passes L1+L2 for every non-timestamp field |
| **2** | WS `/v1/audio/speech/stream`: session protocol, chunker, inter-chunk pause, cancel, multi-turn, audio-stall guard | WS golden-trace replay passes L1+L2 |
| **3** | Temporal alignment: aux-layer capture, pointer head, Viterbi, streaming commit horizon, incremental emission | Spike (Task 1) proves layer-3 capture at prefill + decode; then L3 within CI on blind30 |
| **4** | Prometheus metrics (identical names), OTel spans, structured logs, Grafana dashboard | `/metrics` exposes the same series names vllm-omni does |
| **5** | Low-latency configuration sweep: TP/SP mesh, CUDA-graph mode, codec chunk policy, FA backend, disaggregation | A configuration that meets the L4 floor |
| **6** | Apples-to-apples benchmark harness; Plotly HTML + matplotlib PNG; parity report | L4 measured and published, pass or fail |
| **7** | Serving Dockerfile, image build docs, drop-in validation against the pipecat client | pipecat runs unmodified against the M* image |

**Sequencing note.** Phase 4 (metrics) is deliberately *after* Phase 3 rather than early:
several of the metric series that matter (`tts_alignment_*`, including the
`commit_revision` counter that guards the streaming horizon) only exist once alignment
does. Phases 5 and 6 are co-dependent — the sweep needs the harness — so Phase 6's
harness lands first inside Phase 5's plan and Phase 6 consumes it.

---

## Hard constraints

Copied verbatim into every phase plan; every task's requirements implicitly include these.

1. **Protocol is frozen.** The WS message shapes and the REST field names/semantics are fixed by the vllm-omni implementation. M* adapts to them; they do not adapt to M*. Any proposed deviation is a spec change, not an implementation detail.
2. **`uv` only.** Never bare `pip` / `python` (repo convention).
3. **No new top-level directories** in the consuming repos.
4. **Codec frame rate is 12.5 Hz exactly** (`24000 / 1920 = 80 ms/frame`). Not 12 Hz. A 5 % error is ~500 ms of drift over 10 s.
5. **Word times are turn-relative seconds**, inter-chunk pause included, reset on `session.done` and `cancel`.
6. **`timestamp_emission` defaults to `"incremental"`** (vllm-omni `ee9641f`); `"chunk"` is the opt-out.
7. **Every parity claim cites a measurement.** No "should be equivalent" in a report.
8. **Upstream syncs are merges from `upstream/main`**, never force-pushes; `inflection/qwen3-tts` keeps linear history.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Public fork (Gate 0) | Forfeits the provisional patent | Blocking gate; nothing pushed until resolved |
| Layer-3 aux capture not exposable in M*'s Talker | Phase 3 dead; no word timestamps ⇒ no drop-in | Spike is Phase 3 Task 1, before any other Phase 3 work |
| M* codec streaming ≠ vllm-omni's `codec_chunk_frames: 1` cadence | PCM frame timing differs; karaoke and splice math shift | Phase 2 measures PCM inter-arrival on both and pins the chunk policy |
| M* is not actually faster | The whole premise | Phase 6 publishes the negative result; migration decision is explicitly downstream of it |
| Two implementations drift after launch | Parity rots silently | The conformance suite runs against **both** servers in CI, so drift fails a build rather than a customer |
| Sampling nondeterminism misread as a parity bug | Wasted debugging, false alarms | L1/L2 are structural; L3/L4 are statistical with CIs. Stated up front. |

---

## Non-goals

- Changing the WS or REST protocol in any way
- Replacing vllm-omni before Phase 6 publishes numbers
- Upstreaming any of this to `mstar-project/mstar` (separate decision, gated on Gate 0)
- Supporting M* models other than Qwen3-TTS on this branch
