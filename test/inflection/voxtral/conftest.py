"""Fixtures for the Voxtral-Realtime suites.

Two tiers, deliberately separated:

* CPU tier -- config parsing, cadence arithmetic, the mel front end. Runs in
  CI with no GPU and no checkpoint beyond the tokenizer.
* GPU tier -- loads the real checkpoint and asserts token-level parity against
  recorded reference output. Skipped when the checkpoint is absent.
"""

import os
from pathlib import Path

import pytest

CHECKPOINT = os.environ.get(
    "VOXTRAL_MODEL_PATH",
    "/mnt/data/models/audio/stt/voxtral-rt/pretrained/Voxtral-Mini-4B-Realtime-2602",
)


def _has_checkpoint() -> bool:
    d = Path(CHECKPOINT)
    return all((d / f).is_file() for f in ("config.json", "model.safetensors", "tekken.json"))


@pytest.fixture(scope="session")
def checkpoint() -> str:
    if not _has_checkpoint():
        pytest.skip(f"no Voxtral checkpoint at {CHECKPOINT}")
    return CHECKPOINT


@pytest.fixture(scope="session")
def config(checkpoint):
    from mstar.model.voxtral_rt.config import VoxtralRealtimeConfig

    return VoxtralRealtimeConfig.from_pretrained(checkpoint)


@pytest.fixture(scope="session")
def model(checkpoint):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Voxtral parity needs a GPU")
    from mstar.model.voxtral_rt.model import VoxtralRealtime

    return VoxtralRealtime.from_pretrained(checkpoint, device="cuda", dtype=torch.bfloat16)
