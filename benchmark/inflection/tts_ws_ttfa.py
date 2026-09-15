"""TTFA over the WebSocket — the number a voice agent actually feels.

The REST harness (``tts_ab.py``) deliberately refuses to report TTFA: neither
server's REST streaming delivers it (M* flushes at completion, vllm-omni
re-sends cumulatively). The WebSocket does — it is the path pipecat uses, and
both servers now speak it.

TTFA here is time from ``input.done`` to the first BINARY frame, i.e. the first
PCM the client could play. The ``audio.start`` control frame is explicitly not
counted: it carries no audio, and timing it would flatter both servers by the
whole synthesis time.

    python -m benchmark.inflection.tts_ws_ttfa \
        --mstar http://127.0.0.1:8100 --omni http://127.0.0.1:8901 \
        --concurrencies 1,4,8,16,32 --reps 5 --out out/ttfa
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import time

import websockets

CORPUS = [
    "Hello there, how can I help you today?",
    "The quick brown fox jumps over the lazy dog while the morning light spills across the valley.",
    "I was thinking we could go over the quarterly numbers before the meeting starts.",
    "Somewhere further down the road a second traveller is already awake and walking, "
    "counting the miles that still separate them from the town they left behind.",
    "Sure thing. Let me pull that up for you right now.",
]


WS_PATH = "/v1/audio/speech/stream"


def ws_url(base: str) -> str:
    """Normalise a server address to the streaming endpoint.

    Accepts an http(s) base, a ws(s) base, or either already carrying the
    endpoint path. Appending unconditionally is the obvious implementation and
    the wrong one: pass the full ws:// URL -- which is what anyone who has read
    the protocol docs will reach for -- and every handshake comes back 404,
    reported as `ALL FAILED ['InvalidStatus']` with no hint that the URL was
    doubled.
    """
    url = base.strip().rstrip("/")
    url = url.replace("https://", "wss://").replace("http://", "ws://")
    if not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    if url.endswith(WS_PATH):
        return url
    return url + WS_PATH


async def one_turn(url: str, text: str, voice: str, timeout: float) -> dict:
    cfg = {"type": "session.config", "voice": voice, "language": "Auto", "response_format": "wav"}
    try:
        async with websockets.connect(url, max_size=None, open_timeout=60) as ws:
            await ws.send(json.dumps(cfg))
            await ws.send(json.dumps({"type": "input.text", "text": text + " "}))
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "input.done"}))
            ttfa = None
            pcm = 0
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                if isinstance(raw, bytes):
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                    pcm += len(raw)
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "error":
                    return {"ok": False, "error": msg.get("message", "")[:80]}
                if msg.get("type") in ("session.done", "cancelled"):
                    break
            total = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 — a failed turn is a datapoint
        return {"ok": False, "error": type(e).__name__}
    if ttfa is None or pcm == 0:
        return {"ok": False, "error": "no audio"}
    audio_s = pcm / 2 / 24000
    return {"ok": True, "ttfa_s": ttfa, "total_s": total, "audio_s": audio_s,
            "rtf": total / audio_s if audio_s else None}


async def run_level(base: str, conc: int, voice: str, timeout: float, reps: int) -> dict:
    url = ws_url(base)
    results: list[dict] = []
    t0 = time.perf_counter()
    for rep in range(reps):
        texts = [CORPUS[(rep * conc + i) % len(CORPUS)] for i in range(conc)]
        results.extend(await asyncio.gather(*[one_turn(url, t, voice, timeout) for t in texts]))
    wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    if not ok:
        errs = {r.get("error") for r in results}
        return {"concurrency": conc, "n_ok": 0, "n_total": len(results),
                "success_rate": 0.0, "errors": sorted(e for e in errs if e)}

    def agg(key: str) -> dict:
        vals = sorted(r[key] for r in ok if r.get(key) is not None)
        return {"mean": statistics.fmean(vals), "p50": vals[len(vals) // 2],
                "p90": vals[min(int(len(vals) * 0.9), len(vals) - 1)], "max": vals[-1]}

    return {
        "concurrency": conc, "n_ok": len(ok), "n_total": len(results),
        "success_rate": 100.0 * len(ok) / len(results),
        "ttfa_ms": {k: v * 1000 for k, v in agg("ttfa_s").items()},
        "total_s": agg("total_s"), "rtf": agg("rtf"),
        "audio_s_mean": statistics.fmean(r["audio_s"] for r in ok),
        "xrt_aggregate": sum(r["audio_s"] for r in ok) / wall if wall > 0 else None,
    }


async def main_async(args) -> dict:
    arms = {k: v for k, v in (("mstar", args.mstar), ("omni", args.omni)) if v}
    out: dict = {"arms": list(arms), "voice": args.voice, "surface": "websocket", "levels": {}}
    for name, base in arms.items():
        print(f"warming {name} ...", flush=True)
        await run_level(base, 1, args.voice, args.timeout, args.warmup)
    for conc in [int(c) for c in args.concurrencies.split(",")]:
        out["levels"][str(conc)] = {}
        for name, base in arms.items():  # interleaved per level
            print(f"  c={conc:<3} {name} ...", end="", flush=True)
            r = await run_level(base, conc, args.voice, args.timeout, args.reps)
            out["levels"][str(conc)][name] = r
            if r["n_ok"]:
                print(f" TTFA p50={r['ttfa_ms']['p50']:.0f}ms p90={r['ttfa_ms']['p90']:.0f}ms "
                      f"rtf={r['rtf']['p50']:.3f} xrt={r['xrt_aggregate']:.1f}x "
                      f"ok={r['success_rate']:.0f}%", flush=True)
            else:
                print(f" ALL FAILED {r.get('errors')}", flush=True)
    return out


def render(out: dict, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {outdir / 'results.json'}")
    arms, levels = out["arms"], sorted(int(c) for c in out["levels"])

    def series(arm, path):
        vals = []
        for c in levels:
            node = out["levels"][str(c)].get(arm, {})
            for k in path:
                node = node.get(k, {}) if isinstance(node, dict) else None
                if node is None:
                    break
            vals.append(node if isinstance(node, (int, float)) else None)
        return vals

    panels = [("TTFA p50 (ms)", ("ttfa_ms", "p50"), "lower is better"),
              ("TTFA p90 (ms)", ("ttfa_ms", "p90"), "lower is better"),
              ("RTF p50", ("rtf", "p50"), "lower is better; <1 beats realtime"),
              ("Throughput (x realtime)", ("xrt_aggregate",), "higher is better")]
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} — {s}" for t, _, s in panels])
        for i, (_t, path, _s) in enumerate(panels):
            row, col = i // 2 + 1, i % 2 + 1
            for arm in arms:
                fig.add_trace(go.Scatter(x=levels, y=series(arm, path), name=arm,
                                         mode="lines+markers", legendgroup=arm, showlegend=(i == 0)),
                              row=row, col=col)
            fig.update_xaxes(title_text="concurrency", type="log", row=row, col=col)
        fig.update_layout(title="Qwen3-TTS WebSocket: M* vs vllm-omni (TTFA = first PCM after input.done)",
                          height=800, template="plotly_white")
        fig.write_html(str(outdir / "ttfa.html"), include_plotlyjs="cdn")
        print(f"wrote {outdir / 'ttfa.html'}")
    except ImportError:
        print("plotly not installed — skipping HTML")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (title, path, sub) in zip(axes.flat, panels, strict=True):
            for arm in arms:
                ys = series(arm, path)
                ax.plot([c for c, y in zip(levels, ys, strict=True) if y is not None],
                        [y for y in ys if y is not None], marker="o", label=arm)
            ax.set_title(f"{title}\n{sub}", fontsize=10)
            ax.set_xlabel("concurrency")
            ax.set_xscale("log", base=2)
            ax.set_xticks(levels)
            ax.set_xticklabels([str(c) for c in levels])
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Qwen3-TTS WebSocket: M* vs vllm-omni (TTFA = first PCM after input.done)")
        fig.tight_layout()
        fig.savefig(outdir / "ttfa.png", dpi=140)
        print(f"wrote {outdir / 'ttfa.png'}")
    except ImportError:
        print("matplotlib not installed — skipping PNG")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mstar")
    ap.add_argument("--omni")
    ap.add_argument("--concurrencies", default="1,4,8,16,32")
    ap.add_argument("--voice", default="alexandra")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="out/ttfa")
    args = ap.parse_args()
    if not (args.mstar or args.omni):
        raise SystemExit("give at least one of --mstar / --omni")
    render(asyncio.run(main_async(args)), pathlib.Path(args.out))


if __name__ == "__main__":
    main()
