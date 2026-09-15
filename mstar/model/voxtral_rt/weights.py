"""One checkpoint -> two module trees.

Voxtral has two implementations in this repo on purpose: the standalone one in
``model.py`` (the verified oracle, token-identical to the HF reference) and the
engine-backed one behind the Walk Graph. They must load the SAME weights the
same way, or the parity test between them is testing two different models.

So the checkpoint key mapping lives here, once, and both paths call it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import safe_open
from torch import nn

logger = logging.getLogger(__name__)

# The checkpoint nests the decoder one level deeper than either module tree
# does, and may or may not carry a leading "model." depending on how it was
# exported. Longest prefix first: "model.language_model.model." must win over
# "model.", or half the decoder silently fails to map.
_RENAMES: tuple[tuple[str, str], ...] = (
    ("model.language_model.model.", "language_model."),
    ("language_model.model.", "language_model."),
    ("model.audio_tower.", "audio_tower."),
    ("model.multi_modal_projector.", "multi_modal_projector."),
    ("model.time_embedding.", "time_embedding."),
)


def canonical_name(key: str) -> str:
    for src, dst in _RENAMES:
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def checkpoint_path(model_dir: str | Path) -> Path:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Voxtral-Realtime on M* loads the HF-format "
            "weights; a checkpoint shipping only consolidated.safetensors is "
            "not supported yet."
        )
    return path


def iter_weights(model_dir: str | Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(canonical_name, tensor)`` for every weight in the checkpoint."""
    with safe_open(str(checkpoint_path(model_dir)), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 — safetensors handle, not a dict
            yield canonical_name(key), f.get_tensor(key)


def load_into(
    targets: dict[str, nn.Module],
    model_dir: str | Path,
    *,
    device: str = "cuda",
    strict: bool = True,
) -> set[str]:
    """Copy checkpoint weights into modules keyed by canonical prefix.

    ``targets`` maps a canonical prefix (``"audio_tower."``) to the module that
    owns everything under it. Returns the set of prefixes that received at
    least one weight.

    With ``strict``, a target module left with an unfilled parameter raises.
    That matters more here than in most loaders: a partially loaded Voxtral
    does not crash, it produces a fluent and completely wrong transcript.
    """
    owned: dict[str, dict[str, torch.Tensor]] = {p: {} for p in targets}
    for prefix, module in targets.items():
        for name, param in module.named_parameters():
            owned[prefix][name] = param
        for name, buf in module.named_buffers():
            owned[prefix][name] = buf

    filled: dict[str, set[str]] = {p: set() for p in targets}
    touched: set[str] = set()
    for name, tensor in iter_weights(model_dir):
        for prefix in targets:
            if not name.startswith(prefix):
                continue
            local = name[len(prefix):]
            dest = owned[prefix].get(local)
            if dest is None:
                logger.debug("voxtral: ignoring unmapped weight %s", name)
                break
            if dest.shape != tensor.shape:
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint "
                    f"{tuple(tensor.shape)} vs model {tuple(dest.shape)}"
                )
            dest.data.copy_(tensor.to(device=device, dtype=dest.dtype))
            filled[prefix].add(local)
            touched.add(prefix)
            break

    if strict:
        for prefix, params in owned.items():
            missing = sorted(set(params) - filled[prefix] - _NON_CHECKPOINT)
            if missing:
                raise ValueError(
                    f"{len(missing)} weights under {prefix!r} were not found in "
                    f"the checkpoint, e.g. {missing[:5]}. Loading a "
                    "Voxtral-Realtime model with missing weights produces a "
                    "fluent, wrong transcript rather than an error."
                )
    return touched


# Non-persistent buffers: rebuilt at load, never stored in a checkpoint.
_NON_CHECKPOINT = {"inv_freq", "time_embedding.inv_freq"}
