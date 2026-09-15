"""Fixtures for the differential parity suite.

Every assertion runs against BOTH servers. A server whose URL is unset is
skipped, so the suite is useful with one server up (conformance against the
contract) and decisive with two (M* vs vllm-omni, same assertions).

    MSTAR_TTS_URL=http://127.0.0.1:8100 \
    OMNI_TTS_URL=http://127.0.0.1:8901 \
    .venv/bin/python -m pytest test/parity -v
"""

import os

import pytest

MSTAR_URL = os.environ.get("MSTAR_TTS_URL")
OMNI_URL = os.environ.get("OMNI_TTS_URL")
# Per-checkpoint. Our 14-voice SFT line ships alexandra/...; the stock
# CustomVoice model ships "Vivian". Never hardcode either into an assertion.
VOICE = os.environ.get("QWEN3_TTS_VOICE", "alexandra")


@pytest.fixture(scope="session")
def voice() -> str:
    return VOICE


@pytest.fixture(params=["mstar", "omni"])
def server(request) -> tuple[str, str]:
    url = {"mstar": MSTAR_URL, "omni": OMNI_URL}[request.param]
    if not url:
        pytest.skip(f"set {'MSTAR_TTS_URL' if request.param == 'mstar' else 'OMNI_TTS_URL'}")
    return request.param, url
