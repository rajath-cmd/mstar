"""Differential REST parity: one set of assertions, both servers.

Parametrised by the ``server`` fixture, so each test reports twice —
``[mstar]`` and ``[omni]`` — and a divergence is visible as one red bar next to
one green one rather than as a narrative.
"""

from test.parity import contract_rest


def test_accepts_every_field(server, voice):
    contract_rest.assert_accepts_every_field(server[1], voice)


def test_raw_audio_without_timestamps(server, voice):
    contract_rest.assert_raw_audio_without_timestamps(server[1], voice)


def test_envelope_with_timestamps(server, voice):
    contract_rest.assert_envelope_with_timestamps(server[1], voice)


def test_batch(server, voice):
    contract_rest.assert_batch_returns_one_result_per_input(server[1], voice)


def test_rejects_unknown_timestamp_type(server, voice):
    contract_rest.assert_rejects_unknown_timestamp_type(server[1], voice)


def test_lists_voices(server, voice):
    contract_rest.assert_lists_voices(server[1], voice)
