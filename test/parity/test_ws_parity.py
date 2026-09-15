"""Differential WebSocket parity: one assertion set, both servers.

This is the surface pipecat uses, so it is the one that decides drop-in
replacement.
"""

import pytest

from test.parity import contract_ws

pytestmark = pytest.mark.asyncio


def _ws(url: str) -> str:
    return url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/audio/speech/stream"


async def test_basic_turn_shape(server):
    await contract_ws.assert_basic_turn_shape(_ws(server[1]))


async def test_audio_start_fields(server):
    await contract_ws.assert_audio_start_fields(_ws(server[1]))


async def test_audio_done_fields(server):
    await contract_ws.assert_audio_done_fields(_ws(server[1]))


async def test_session_done_counts_chunks(server):
    await contract_ws.assert_session_done_counts_chunks(_ws(server[1]))


async def test_sentence_index_increments(server):
    await contract_ws.assert_sentence_index_increments(_ws(server[1]))


async def test_cancel_acks_and_stops(server):
    await contract_ws.assert_cancel_acks_and_stops(_ws(server[1]))


async def test_unknown_message_errors(server):
    await contract_ws.assert_unknown_message_errors(_ws(server[1]))


async def test_voice_list_is_a_preconfig_oneshot(server, voice):
    await contract_ws.assert_voice_list_is_a_preconfig_oneshot(_ws(server[1]), voice)
