"""WebSocket /v1/audio/speech/stream contract, asserted against either server.

This is the surface pipecat actually talks to, so it is the one that decides
whether an M* image is a drop-in replacement. As with the REST contract,
NOTHING here compares audio bytes or word times across servers — sampling is
not greedy and no seed is exposed, so two runs of the SAME server already
differ. What is compared is frame types, ordering, field presence, and the
invariants a client relies on.
"""

from __future__ import annotations

import json

import websockets


async def run_turn(
    base_ws: str, cfg_extra: dict, texts: list[str], *, cancel_after_frames: int | None = None,
    timeout: float = 180.0,
) -> dict:
    """Drive one turn; return the observed frame stream.

    ``order`` is the interleaving a client sees ("pcm" for a binary frame), which
    is what ordering assertions are made against.
    """
    cfg = {"type": "session.config", "voice": "alexandra", "language": "Auto",
           "response_format": "wav", **cfg_extra}
    order: list[str] = []
    frames: list[dict] = []
    pcm_bytes = 0
    pcm_count = 0
    import asyncio

    async with websockets.connect(base_ws, max_size=None, open_timeout=60) as ws:
        await ws.send(json.dumps(cfg))
        for t in texts:
            await ws.send(json.dumps({"type": "input.text", "text": t}))
        await ws.send(json.dumps({"type": "input.done"}))
        cancelled = False
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            if isinstance(raw, bytes):
                order.append("pcm")
                pcm_bytes += len(raw)
                pcm_count += 1
                if cancel_after_frames is not None and not cancelled and pcm_count >= cancel_after_frames:
                    await ws.send(json.dumps({"type": "cancel"}))
                    cancelled = True
                continue
            msg = json.loads(raw)
            order.append(msg.get("type", "?"))
            frames.append(msg)
            if msg.get("type") in ("session.done", "cancelled"):
                break
            if msg.get("type") == "error":
                break
    return {"order": order, "frames": frames, "pcm_bytes": pcm_bytes, "pcm_count": pcm_count}


def of(result: dict, kind: str) -> list[dict]:
    return [f for f in result["frames"] if f.get("type") == kind]


# --- assertions --------------------------------------------------------------

SHORT = "Hello there, how can I help you today? "
MULTI = ("First sentence here for the demonstration of chunk boundaries. "
         "Second sentence follows along afterwards with more words. "
         "Third and final sentence closes the whole turn out completely. ")


async def assert_basic_turn_shape(base_ws: str) -> None:
    """audio.start -> PCM... -> audio.done -> session.done, in that order."""
    r = await run_turn(base_ws, {}, [SHORT])
    order = r["order"]
    assert "audio.start" in order, f"no audio.start; got {order[:6]}"
    assert "audio.done" in order, f"no audio.done; got {order[:6]}"
    assert order[-1] == "session.done", f"turn did not end with session.done: {order[-3:]}"
    assert order.index("audio.start") < order.index("pcm") < order.index("audio.done"), (
        f"PCM did not land between audio.start and audio.done: {order[:8]}"
    )
    assert r["pcm_bytes"] > 1000, "suspiciously little audio"


async def assert_audio_start_fields(base_ws: str) -> None:
    r = await run_turn(base_ws, {}, [SHORT])
    start = of(r, "audio.start")[0]
    for key in ("sentence_index", "sentence_text", "format"):
        assert key in start, f"audio.start missing {key!r}; got {sorted(start)}"
    assert start["sentence_index"] == 0


async def assert_audio_done_fields(base_ws: str) -> None:
    r = await run_turn(base_ws, {}, [SHORT])
    done = of(r, "audio.done")[0]
    for key in ("sentence_index", "sample_rate", "chunk_count"):
        assert key in done, f"audio.done missing {key!r}; got {sorted(done)}"
    assert done["sample_rate"] in (8000, 24000)
    assert done["chunk_count"] >= 1


async def assert_session_done_counts_chunks(base_ws: str) -> None:
    r = await run_turn(base_ws, {}, [MULTI])
    sd = of(r, "session.done")[0]
    assert "total_sentences" in sd, f"session.done missing total_sentences: {sorted(sd)}"
    assert sd["total_sentences"] == len(of(r, "audio.done")), (
        f"total_sentences={sd['total_sentences']} but {len(of(r,'audio.done'))} audio.done frames"
    )


async def assert_sentence_index_increments(base_ws: str) -> None:
    """Multi-chunk turns number chunks 0,1,2,... with no gaps or repeats."""
    r = await run_turn(base_ws, {"timestamp_type": "word"}, [MULTI])
    idxs = [f["sentence_index"] for f in of(r, "audio.start")]
    assert idxs == list(range(len(idxs))), f"sentence_index not sequential: {idxs}"


async def assert_cancel_acks_and_stops(base_ws: str) -> None:
    """Barge-in returns a `cancelled` ack and ends the turn."""
    r = await run_turn(base_ws, {}, [MULTI], cancel_after_frames=2)
    acks = of(r, "cancelled")
    assert acks, f"no cancelled ack; got {r['order'][-5:]}"
    ack = acks[0]
    for key in ("sentence_index", "drained"):
        assert key in ack, f"cancelled missing {key!r}; got {sorted(ack)}"
    assert r["order"][-1] == "cancelled", f"frames continued after cancel: {r['order'][-4:]}"


async def assert_unknown_message_errors(base_ws: str) -> None:
    import asyncio

    async with websockets.connect(base_ws, max_size=None, open_timeout=60) as ws:
        await ws.send(json.dumps({"type": "session.config", "voice": "alexandra"}))
        await ws.send(json.dumps({"type": "nonsense"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=60)
        msg = json.loads(raw)
        assert msg.get("type") == "error", f"expected an error frame, got {msg}"
        assert "message" in msg


async def assert_voice_list_is_a_preconfig_oneshot(base_ws: str, voice: str) -> None:
    """``voice.list`` is answered BEFORE session.config, and ends the session.

    Not mid-session: vllm-omni handles it (and ``voice.delete``) only in its
    config-receive step, and answers "Unknown message type: voice.list" if it
    arrives after session.config — despite its own module docstring listing it
    among the general client messages. The contract is the implementation, so a
    client must send it as the FIRST message on a throwaway connection.
    """
    import asyncio

    async with websockets.connect(base_ws, max_size=None, open_timeout=60) as ws:
        await ws.send(json.dumps({"type": "voice.list"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
        assert msg.get("type") == "voice.list", f"got {msg.get('type')}: {msg}"
        names = [v.lower() for v in msg.get("voices", [])]
        assert voice.lower() in names, f"{voice!r} absent from {names}"
        assert "uploaded_voices" in msg, f"missing uploaded_voices; got {sorted(msg)}"
