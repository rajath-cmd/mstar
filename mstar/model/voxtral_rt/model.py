"""Voxtral-Realtime: streaming ASR as a synchronous decoder.

The architecture is unusual and worth stating plainly, because every design
decision downstream follows from it.

Audio embeddings are **added** to text embeddings, not spliced in beside them.
Position i of the sequence carries BOTH the i-th 80 ms audio frame and the
i-th text token, summed. So the sequence length is the audio length -- 12.5
positions per second, exactly -- and the model emits either a word piece or
``[STREAMING_PAD]`` at each one. Transcription is therefore not "generate until
EOS": it is a fixed budget of ``audio_seconds * 12.5`` decode steps, and the
transcript is whatever is not a pad.

That is what makes the model realtime-capable: it never needs to see the end of
the audio to emit the beginning of the transcript.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import safe_open
from torch import nn

from mstar.model.voxtral_rt.components.audio_tower import AudioTower
from mstar.model.voxtral_rt.components.projector import Projector
from mstar.model.voxtral_rt.components.text_decoder import TextDecoder, TimeEmbedding
from mstar.model.voxtral_rt.config import VoxtralRealtimeConfig

logger = logging.getLogger(__name__)


@dataclass
class Transcription:
    text: str
    token_ids: list[int]
    audio_seconds: float
    n_audio_tokens: int
    generation_seconds: float

    @property
    def rtf(self) -> float:
        return self.generation_seconds / self.audio_seconds if self.audio_seconds else 0.0


class VoxtralRealtime(nn.Module):
    """Audio tower + projector + time-conditioned Mistral decoder."""

    def __init__(self, cfg: VoxtralRealtimeConfig):
        super().__init__()
        self.config = cfg
        self.audio_tower = AudioTower(cfg.audio)
        self.multi_modal_projector = Projector(cfg)
        self.language_model = TextDecoder(cfg.text)
        self.time_embedding = TimeEmbedding(cfg.text.hidden_size)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        # Where the Tekken tokenizer lives; set by from_pretrained.
        self._model_dir = ""

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    # The checkpoint stores the decoder one level deeper than this module tree
    # does (`language_model.model.layers` vs `language_model.layers`), and
    # optionally behind a `model.` prefix depending on how it was exported.
    # Longest prefix first: `model.language_model.model.` must win over
    # `model.`, or half the decoder silently fails to map.
    _RENAMES = (
        ("model.language_model.model.", "language_model."),
        ("language_model.model.", "language_model."),
        ("model.audio_tower.", "audio_tower."),
        ("model.multi_modal_projector.", "multi_modal_projector."),
        ("model.time_embedding.", "time_embedding."),
    )

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, *, device: str = "cuda", dtype=torch.bfloat16
    ) -> VoxtralRealtime:
        model_dir = Path(model_dir)
        cfg = VoxtralRealtimeConfig.from_pretrained(model_dir)
        with torch.device("meta"):
            model = cls(cfg)
        model = model.to(dtype)
        model.to_empty(device=device)

        # Prefer model.safetensors: it carries HF-style names matching this
        # module tree. consolidated.safetensors is the Mistral-native layout of
        # the same weights and would need a second, different rename table.
        path = model_dir / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Voxtral-Realtime on M* loads the HF-format "
                "weights; a checkpoint shipping only consolidated.safetensors "
                "is not supported yet."
            )

        own = dict(model.named_parameters())
        own.update(dict(model.named_buffers()))
        loaded: set[str] = set()
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safetensors handle, not a dict
                name = key
                for src, dst in cls._RENAMES:
                    if name.startswith(src):
                        name = dst + name[len(src) :]
                        break
                if name not in own:
                    if name != "lm_head.weight":
                        logger.debug("voxtral: ignoring unmapped weight %s", key)
                    continue
                tensor = f.get_tensor(key)
                if tensor.shape != own[name].shape:
                    raise ValueError(
                        f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
                        f"vs model {tuple(own[name].shape)}"
                    )
                own[name].data.copy_(tensor.to(device=device, dtype=own[name].dtype))
                loaded.add(name)

        # Embeddings are tied; the checkpoint stores them once.
        if cfg.text.tie_word_embeddings and "lm_head.weight" not in loaded:
            model.lm_head.weight = model.language_model.embed_tokens.weight
            loaded.add("lm_head.weight")

        missing = sorted(set(own) - loaded - {"time_embedding.inv_freq"})
        if missing:
            raise ValueError(
                f"{len(missing)} weights were not found in the checkpoint, e.g. "
                f"{missing[:5]}. Loading a Voxtral-Realtime model with missing "
                "weights produces a fluent, wrong transcript rather than an error."
            )

        model.eval()
        model._model_dir = str(model_dir)
        # to_empty() above allocated buffer storage without initialising it,
        # and non-persistent buffers are by definition absent from the
        # checkpoint, so they must be rebuilt explicitly here.
        model.time_embedding.reset_buffer(device=device, dtype=dtype)
        model.prepare_time_conditioning(cfg.default_num_delay_tokens)
        return model

    def prepare_time_conditioning(self, num_delay_tokens: int) -> None:
        """Precompute each layer's constant time-conditioning scale.

        ``t_cond`` depends only on ``num_delay_tokens``, which is fixed for a
        deployment, so every layer's ``ada_rms_norm(t_cond)`` is a constant
        vector. Computing it once here is not an approximation: the decode path
        produces identical values with two fewer matmuls per layer per token.
        """
        p = next(self.parameters())
        t = torch.full((1,), float(num_delay_tokens), device=p.device, dtype=p.dtype)
        with torch.no_grad():
            t_cond = self.time_embedding(t)[None, ...]
            self.language_model.t_scales = [
                layer.ada_rms_norm(t_cond) for layer in self.language_model.layers
            ]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """log-mel ``[n_mels, T]`` -> one text-space embedding per 80 ms.

        The tower halves the mel rate, then ``downsample_factor`` output frames
        are concatenated channel-wise into one projector input. 8 mel frames in,
        one embedding out.
        """
        p = next(self.parameters())
        mel = input_features.to(device=p.device, dtype=p.dtype)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        hidden = self.audio_tower(mel)
        b, t, h = hidden.shape
        k = self.config.downsample_factor
        usable = (t // k) * k
        hidden = hidden[:, :usable].reshape(b, usable // k, h * k)
        return self.multi_modal_projector(hidden)

    @torch.no_grad()
    def transcribe(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        *,
        max_tokens: int | None = None,
    ) -> Transcription:
        """Run the full synchronous decode and return the transcript.

        The loop is bounded by AUDIO, not by an EOS token: it runs until the
        audio embeddings are exhausted. A model that never stops talking is not
        a failure mode here -- there is nothing left to condition on.
        """
        t0 = time.perf_counter()
        p = next(self.parameters())
        audio_embeds = self.encode_audio(input_features)  # [1, n_tok, hidden]
        n_audio = audio_embeds.shape[1]
        budget = n_audio if max_tokens is None else min(n_audio, max_tokens)

        ids = input_ids.to(p.device).view(1, -1)
        n_prompt = ids.shape[1]
        if n_prompt > budget:
            raise ValueError(
                f"prompt ({n_prompt} tokens) is longer than the audio "
                f"({budget} frames); the utterance is too short to transcribe."
            )

        caches: list[dict] = [{} for _ in range(self.config.text.num_hidden_layers)]

        # Prefill: every prompt position gets its own audio frame added.
        embeds = self.language_model.embed_tokens(ids) + audio_embeds[:, :n_prompt]
        hidden = self.language_model(embeds, 0, caches)
        next_id = self.lm_head(hidden[:, -1:]).argmax(-1)  # [1, 1]

        out: list[int] = [int(next_id.item())]
        for step in range(n_prompt, budget - 1):
            embeds = (
                self.language_model.embed_tokens(next_id)
                + audio_embeds[:, step : step + 1]
            )
            hidden = self.language_model(embeds, step, caches)
            next_id = self.lm_head(hidden[:, -1:]).argmax(-1)
            out.append(int(next_id.item()))

        from mstar.model.voxtral_rt.audio_frontend import decode

        return Transcription(
            text=decode(self._model_dir, out).strip(),
            token_ids=out,
            audio_seconds=n_audio / self.config.frame_rate_hz,
            n_audio_tokens=n_audio,
            generation_seconds=time.perf_counter() - t0,
        )

