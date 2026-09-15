"""Is the streamed audio COMPLETE, or did the codec cadence truncate it?

A config that drops the start of the stream still sounds fluent — it just begins
mid-sentence — so it wins on TTFA while being wrong. Duration alone is a weak
signal because TTS length varies run to run, so this records the duration AND
transcribes, letting a caller diff the transcript against the script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import wave

import websockets

SCRIPT = ("The quick brown fox jumps over the lazy dog while the morning light "
          "spills across the quiet valley below.")
# First few words of the script; their absence is the truncation signature.
OPENING = ["the", "quick", "brown", "fox"]


async def capture(base_url: str, wav_path: pathlib.Path) -> float:
    url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream"
    pcm = b""
    async with websockets.connect(url, max_size=None, open_timeout=60) as ws:
        await ws.send(json.dumps({"type": "session.config", "voice": "alexandra",
                                  "language": "Auto", "response_format": "wav"}))
        await ws.send(json.dumps({"type": "input.text", "text": SCRIPT + " "}))
        await ws.send(json.dumps({"type": "input.done"}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180)
            if isinstance(raw, bytes):
                pcm += raw
                continue
            if json.loads(raw).get("type") in ("session.done", "cancelled", "error"):
                break
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return len(pcm) / 2 / 24000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    duration = asyncio.run(capture(args.url, wav))

    transcript = ""
    try:
        import subprocess
        r = subprocess.run(
            ["python3", str(pathlib.Path.home() / ".claude/skills/moss-listen/scripts/moss_listen.py"),
             str(wav), "Transcribe verbatim. Nothing else."],
            capture_output=True, text=True, timeout=180, check=False,
        )
        transcript = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    except Exception as e:  # noqa: BLE001 — transcription is best-effort
        transcript = f"<unavailable: {type(e).__name__}>"

    low = transcript.lower()
    complete = all(w in low for w in OPENING) if transcript and not transcript.startswith("<") else None
    out.write_text(json.dumps({
        "duration_s": round(duration, 2),
        "transcript": transcript,
        "opening_words_present": complete,
        "wav": str(wav),
    }, indent=2))
    print(f"    audio {duration:.2f}s | opening present: {complete} | {transcript[:70]}")


if __name__ == "__main__":
    main()
