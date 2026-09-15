"""Word-level alignment timestamps for Qwen3-TTS streaming inference.

Ports the trained pointer head, monotonic Viterbi decoder, and word
segmentation from the training-side ``temporal_alignment`` package.
See :mod:`registry` for the per-request state holder + :mod:`load_head`
for the sidecar ``pointer_head.pt`` loader (no training-side analog).
"""

from mstar.model.qwen3_tts.temporal_alignment.frame_rate import (
    CODEC_FRAME_RATE_HZ,
    frame_to_sec,
    sec_to_frame,
)
from mstar.model.qwen3_tts.temporal_alignment.pointer_head import (
    AlignmentPointerHead,
    pool_word_keys,
)
from mstar.model.qwen3_tts.temporal_alignment.readout import (
    word_timestamps,
)
from mstar.model.qwen3_tts.temporal_alignment.viterbi import (
    monotonic_viterbi,
    word_timestamps_viterbi,
    word_timestamps_viterbi_full_coverage,
)
from mstar.model.qwen3_tts.temporal_alignment.word_segmentation import (
    WordSpan,
    segment_words,
    spoken_words,
)

__all__ = [
    "CODEC_FRAME_RATE_HZ",
    "AlignmentPointerHead",
    "WordSpan",
    "frame_to_sec",
    "monotonic_viterbi",
    "pool_word_keys",
    "sec_to_frame",
    "segment_words",
    "spoken_words",
    "word_timestamps",
    "word_timestamps_viterbi",
    "word_timestamps_viterbi_full_coverage",
]
