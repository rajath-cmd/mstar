"""Latency / RTF benchmark for the Voxtral-Realtime transcription endpoint.

    python -m benchmark.inflection.voxtral_bench \\
        --url http://127.0.0.1:8200 --audio-dir /path/to/wavs \\
        --concurrencies 1,2,4,8 --out out/voxtral

Metric definitions, so the numbers mean something:
  latency   wall time from request send to the complete response
  audio_s   duration of the submitted audio
  RTF       latency / audio_s  -- below 1.0 is faster than realtime
  xrt       audio_s / latency  -- throughput as a multiple of realtime

Note what a concurrency sweep measures against the bring-up server: it serves
one request at a time behind a lock, so RTF is EXPECTED to rise roughly
linearly with concurrency. That is the number to beat once the Walk Graph
engine path lands, not a defect to hide.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import pathlib
import statistics
import time

import soundfile as sf


async def _one(session, url: str, path: str, audio_s: float) -> dict:
    import aiohttp

    t0 = time.perf_counter()
    with open(path, "rb") as fh:
        data = aiohttp.FormData()
        data.add_field("file", fh, filename=os.path.basename(path))
        data.add_field("response_format", "json")
        try:
            async with session.post(f"{url}/v1/audio/transcriptions", data=data) as r:
                body = await r.json()
                ok = r.status == 200
        except Exception as exc:  # noqa: BLE001 — a failed clip is a datapoint
            return {"ok": False, "error": type(exc).__name__, "audio_s": audio_s}
    latency = time.perf_counter() - t0
    return {
        "ok": ok,
        "latency": latency,
        "audio_s": audio_s,
        "rtf": latency / audio_s if audio_s else 0.0,
        "chars": len(body.get("text", "")) if ok else 0,
    }


async def _level(url: str, clips: list[tuple[str, float]], conc: int, reps: int) -> dict:
    import aiohttp

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1800)
    ) as session:
        jobs = [clips[i % len(clips)] for i in range(conc * reps)]
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(conc)

        async def run(item):
            async with sem:
                return await _one(session, url, item[0], item[1])

        results = await asyncio.gather(*[run(j) for j in jobs])
        wall = time.perf_counter() - t0

    good = [r for r in results if r.get("ok")]
    if not good:
        return {"concurrency": conc, "ok_rate": 0.0}
    lat = sorted(r["latency"] for r in good)
    total_audio = sum(r["audio_s"] for r in good)
    return {
        "concurrency": conc,
        "n": len(results),
        "ok_rate": len(good) / len(results),
        "latency_p50_ms": round(statistics.median(lat) * 1000, 1),
        "latency_p90_ms": round(lat[int(0.9 * (len(lat) - 1))] * 1000, 1),
        "rtf_p50": round(statistics.median(r["rtf"] for r in good), 4),
        # Raw per-request values so any percentile can be recomputed without
        # re-running the sweep.
        "samples": {
            "latency_ms": [r["latency"] * 1000 for r in good],
            "rtf": [r["rtf"] for r in good],
            "audio_s": [r["audio_s"] for r in good],
        },
        # Aggregate throughput uses the LEVEL's wall clock, not the sum of
        # per-request times: summing would count overlapped work twice and
        # report a throughput the server never achieved.
        "xrt_aggregate": round(total_audio / wall, 2),
        "wall_s": round(wall, 2),
    }


def render(report: dict, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(report, indent=1))
    levels = report["levels"]
    xs = [lv["concurrency"] for lv in levels]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Latency p50/p90 (ms)", "RTF p50"))
        fig.add_trace(go.Scatter(x=xs, y=[lv["latency_p50_ms"] for lv in levels], name="p50"), 1, 1)
        fig.add_trace(go.Scatter(x=xs, y=[lv["latency_p90_ms"] for lv in levels], name="p90"), 1, 1)
        fig.add_trace(go.Scatter(x=xs, y=[lv["rtf_p50"] for lv in levels], name="RTF"), 1, 2)
        fig.add_hline(y=1.0, line_dash="dash", row=1, col=2)
        fig.update_layout(title="Voxtral-Realtime on M*", template="plotly_white", height=420)
        fig.write_html(str(outdir / "voxtral.html"), include_plotlyjs="cdn")
    except ImportError:
        print("plotly not installed — skipping HTML")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(xs, [lv["latency_p50_ms"] for lv in levels], "o-", label="p50")
        ax[0].plot(xs, [lv["latency_p90_ms"] for lv in levels], "s--", label="p90")
        ax[0].set_xlabel("concurrency")
        ax[0].set_ylabel("ms")
        ax[0].legend()
        ax[0].set_title("Transcription latency")
        ax[1].plot(xs, [lv["rtf_p50"] for lv in levels], "o-", color="tab:red")
        ax[1].axhline(1.0, ls="--", c="k", lw=0.8)
        ax[1].set_xlabel("concurrency")
        ax[1].set_ylabel("RTF")
        ax[1].set_title("RTF p50 (<1 = realtime)")
        fig.tight_layout()
        fig.savefig(outdir / "voxtral.png", dpi=140)
    except ImportError:
        print("matplotlib not installed — skipping PNG")
    print(f"wrote {outdir}/results.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="out/voxtral")
    args = ap.parse_args()

    clips = []
    for p in sorted(glob.glob(os.path.join(args.audio_dir, "*.wav"))):
        clips.append((p, sf.info(p).duration))
    if not clips:
        raise SystemExit(f"no .wav files in {args.audio_dir}")
    print(f"{len(clips)} clips, {sum(c[1] for c in clips):.1f}s of audio")

    levels = []
    for conc in [int(c) for c in args.concurrencies.split(",")]:
        print(f"  c={conc:<4}", end=" ", flush=True)
        lv = asyncio.run(_level(args.url, clips, conc, args.reps))
        levels.append(lv)
        if lv.get("ok_rate"):
            print(f"p50={lv['latency_p50_ms']}ms rtf={lv['rtf_p50']} "
                  f"xrt={lv['xrt_aggregate']}x ok={lv['ok_rate']*100:.0f}%")
        else:
            print("ALL FAILED")
    render({"url": args.url, "n_clips": len(clips), "levels": levels}, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
