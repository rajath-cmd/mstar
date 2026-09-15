"""Load ``pointer_head.pt`` sidecar (trained alignment head).

The head ships as a torch dict with five keys: ``state_dict``,
``hidden_size``, ``proj_size``, ``layer``, ``head_type``. Reading the
dims from the file means a new head (different layer, head_type, etc.)
drops in via path swap — no code change.

Missing sidecar returns ``None`` quietly so the server boots against
checkpoints without an alignment head; word-timestamp requests then
return no words via ``pop_word_alignment``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.model.qwen3_tts.temporal_alignment.pointer_head import (
    AlignmentPointerHead,
)

logger = logging.getLogger(__name__)


def _set_head_loaded_gauge(value: int) -> None:
    """Best-effort: set the head-loaded Prometheus gauge to 0 or 1."""
    try:
        from mstar.model.qwen3_tts.temporal_alignment._metrics import TTS_ALIGNMENT_HEAD_LOADED

        TTS_ALIGNMENT_HEAD_LOADED.set(value)
    except Exception:
        # Worker process may not have prometheus_client; never let metrics
        # break the load path.
        pass


def find_pointer_head(model_dir: str | Path) -> Path | None:
    p = Path(model_dir) / "pointer_head.pt"
    if p.is_file():
        return p
    logger.info(
        "no pointer_head.pt found at %s; word-timestamp requests will return no words "
        "(this checkpoint does not ship an alignment head)",
        p,
    )
    _set_head_loaded_gauge(0)
    return None


def load_pointer_head_from_pt(
    pt_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[AlignmentPointerHead, int]:
    """Load the trained head; return ``(head, layer_index)``.

    ``layer_index`` is the index into vLLM's ``aux_hidden_state_layers``
    tuple (= HF's ``outputs.hidden_states[N]`` indexing).
    """
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        raise FileNotFoundError(f"pointer_head.pt not found: {pt_path}")

    d = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    required = {"state_dict", "hidden_size", "proj_size", "layer", "head_type"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"pointer_head.pt at {pt_path} is missing required keys: {sorted(missing)}")

    head = AlignmentPointerHead(
        hidden_size=int(d["hidden_size"]),
        proj_size=int(d["proj_size"]),
        head_type=str(d["head_type"]),
    )
    head.load_state_dict(d["state_dict"])
    head.eval()
    if dtype is not None:
        head = head.to(dtype=dtype)
    head = head.to(device=torch.device(device))
    layer = int(d["layer"])

    logger.info(
        "loaded AlignmentPointerHead from %s (hidden_size=%d, proj_size=%d, layer=%d, head_type=%s)",
        pt_path,
        head.hidden_size,
        head.proj_size,
        layer,
        head.head_type,
    )
    _set_head_loaded_gauge(1)
    return head, layer
