"""Apples-to-apples TTS benchmark: M* vs vllm-omni on one checkpoint.

Same client, same corpus, same concurrency ladder, same checkpoint. Arms are
INTERLEAVED per concurrency level rather than run back to back, so thermal
drift and co-tenant noise hit both arms equally instead of landing entirely on
whichever ran second.

    python -m benchmark.inflection.tts_ab \
        --mstar http://127.0.0.1:8100 --omni http://127.0.0.1:8901 \
        --concurrencies 1,4,8,16,32 --out out/tts_ab

Writes results.json plus, when plotly/matplotlib are installed, an interactive
HTML and a PNG.

Metric definitions (identical for both arms, so the comparison is meaningful):
  TTFB      wall time from request send to the FIRST audio byte
  total     wall time to the last byte
  audio_s   decoded PCM duration
  RTF       total / audio_s        (lower is better; <1 is faster than realtime)
  xrt       audio_s / total        (throughput as a multiple of realtime)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import time

import httpx

CORPUS = [
    "Hello there, how can I help you today?",
    "The quick brown fox jumps over the lazy dog while the morning light spills across the valley.",
    "I was thinking we could go over the quarterly numbers before the meeting starts.",
    "Somewhere further down the road a second traveller is already awake and walking, "
    "counting the miles that still separate them from the town they left behind.",
    "Sure thing. Let me pull that up for you right now.",
]


async def one_request(client: httpx.AsyncClient, base: str, text: str, voice: str) -> dict:
    """One non-streaming speech request, timed. Returns a metrics dict."""
    body = {"input": text, "voice": voice, "response_format": "wav"}
    t0 = time.perf_counter()
    ttfb = None
    total_bytes = 0
    try:
        async with client.stream("POST", f"{base}/v1/audio/speech", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                return {"ok": False, "error": f"HTTP {r.status_code}"}
            async for chunk in r.aiter_bytes():
                if not chunk:
                    continue
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                total_bytes += len(chunk)
    except Exception as e:  # noqa: BLE001 — a failed request is a datapoint
        return {"ok": False, "error": type(e).__name__}
    total = time.perf_counter() - t0
    # WAV container: 44-byte header, int16 mono @ 24 kHz.
    pcm = max(total_bytes - 44, 0)
    audio_s = pcm / 2 / 24000
    return {
        "ok": True,
        "ttfb_s": ttfb if ttfb is not None else total,
        "total_s": total,
        "audio_s": audio_s,
        "rtf": (total / audio_s) if audio_s > 0 else None,
        "xrt": (audio_s / total) if total > 0 else None,
    }


async def run_level(base: str, conc: int, voice: str, timeout: float) -> dict:
    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        texts = [CORPUS[i % len(CORPUS)] for i in range(conc)]
        results = await asyncio.gather(*[one_request(client, base, t, voice) for t in texts])
    ok = [r for r in results if r["ok"]]
    if not ok:
        return {"concurrency": conc, "n_ok": 0, "n_total": len(results), "success_rate": 0.0}

    def agg(key: str) -> dict:
        vals = sorted(r[key] for r in ok if r.get(key) is not None)
        if not vals:
            return {}
        return {
            "mean": statistics.fmean(vals),
            "p50": vals[len(vals) // 2],
            "p90": vals[min(int(len(vals) * 0.9), len(vals) - 1)],
            "max": vals[-1],
        }

    return {
        "concurrency": conc,
        "n_ok": len(ok),
        "n_total": len(results),
        "success_rate": 100.0 * len(ok) / len(results),
        "ttfb_ms": {k: v * 1000 for k, v in agg("ttfb_s").items()},
        "total_s": agg("total_s"),
        "rtf": agg("rtf"),
        "audio_s_mean": statistics.fmean(r["audio_s"] for r in ok),
        # Aggregate throughput: total audio produced / wall time of the slowest
        # request in the level. Per-request xrt would overstate a batch.
        "xrt_aggregate": sum(r["audio_s"] for r in ok) / max(r["total_s"] for r in ok),
    }


async def main_async(args) -> dict:
    arms = {k: v for k, v in (("mstar", args.mstar), ("omni", args.omni)) if v}
    if not arms:
        raise SystemExit("give at least one of --mstar / --omni")
    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    # Warm each arm so first-request JIT/graph effects do not land in level 1.
    for name, base in arms.items():
        print(f"warming {name} ...", flush=True)
        await run_level(base, args.warmup, args.voice, args.timeout)

    out: dict = {"arms": list(arms), "voice": args.voice, "levels": {}}
    for conc in concurrencies:
        out["levels"][str(conc)] = {}
        # Interleaved: both arms at this level before moving on.
        for name, base in arms.items():
            print(f"  c={conc:<3} {name} ...", end="", flush=True)
            res = await run_level(base, conc, args.voice, args.timeout)
            out["levels"][str(conc)][name] = res
            if res["n_ok"]:
                print(f" ttfb_p50={res['ttfb_ms']['p50']:.0f}ms rtf={res['rtf']['p50']:.3f} "
                      f"xrt={res['xrt_aggregate']:.1f}x ok={res['success_rate']:.0f}%", flush=True)
            else:
                print(" ALL FAILED", flush=True)
    return out


def render(out: dict, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {outdir / 'results.json'}")

    arms = out["arms"]
    levels = sorted(int(c) for c in out["levels"])

    def series(arm: str, path: tuple[str, ...]) -> list[float | None]:
        vals = []
        for c in levels:
            node = out["levels"][str(c)].get(arm, {})
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else None
                if node is None:
                    break
            vals.append(node if isinstance(node, (int, float)) else None)
        return vals

    panels = [
        ("TTFB p50 (ms)", ("ttfb_ms", "p50"), "lower is better"),
        ("TTFB p90 (ms)", ("ttfb_ms", "p90"), "lower is better"),
        ("RTF p50", ("rtf", "p50"), "lower is better; <1 beats realtime"),
        ("Throughput (x realtime)", ("xrt_aggregate",), "higher is better"),
    ]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} — {s}" for t, _, s in panels])
        for i, (_title, path, _sub) in enumerate(panels):
            row, col = i // 2 + 1, i % 2 + 1
            for arm in arms:
                fig.add_trace(
                    go.Scatter(x=levels, y=series(arm, path), name=arm, mode="lines+markers",
                               legendgroup=arm, showlegend=(i == 0)),
                    row=row, col=col,
                )
            fig.update_xaxes(title_text="concurrency", type="log", row=row, col=col)
        fig.update_layout(title="Qwen3-TTS: M* vs vllm-omni (same checkpoint, interleaved arms)",
                          height=800, template="plotly_white")
        fig.write_html(str(outdir / "tts_ab.html"))
        print(f"wrote {outdir / 'tts_ab.html'}")
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
        fig.suptitle("Qwen3-TTS: M* vs vllm-omni (same checkpoint, interleaved arms)")
        fig.tight_layout()
        fig.savefig(outdir / "tts_ab.png", dpi=140)
        print(f"wrote {outdir / 'tts_ab.png'}")
    except ImportError:
        print("matplotlib not installed — skipping PNG")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mstar", default=None, help="M* base URL, e.g. http://127.0.0.1:8100")
    ap.add_argument("--omni", default=None, help="vllm-omni base URL, e.g. http://127.0.0.1:8901")
    ap.add_argument("--concurrencies", default="1,4,8,16,32")
    ap.add_argument("--voice", default="alexandra")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="out/tts_ab")
    args = ap.parse_args()
    render(asyncio.run(main_async(args)), pathlib.Path(args.out))


if __name__ == "__main__":
    main()
