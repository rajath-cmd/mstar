"""Graph nodes for Voxtral-Realtime.

Two nodes, matching the model's actual shape:

``audio_tower``  one-shot at prefill. Turns the whole log-mel into one text-space
                 embedding per 80 ms. Resource-free -- it holds no KV cache,
                 because a dense causal pass over the full utterance is
                 numerically identical to an incremental one and needs no state
                 between calls.

``decoder``      the AR loop. Its one unusual feature is that every step ADDS the
                 step's audio embedding to the token embedding, so the audio
                 embeddings are persisted in request state at prefill and indexed
                 by step. That is the same pattern the Qwen3-TTS talker already
                 uses for ``trailing_text_hidden``.

The decode budget is the audio length, not an EOS token: a request stops when
its audio embeddings run out. ``check_stop`` enforces exactly that, so no
sampling parameter can make a transcript longer or shorter than its audio.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.voxtral_rt.config import (
    SAMPLER,
    TEXT_ATTN,
    TEXT_KV,
    TEXT_POS,
    VoxtralRealtimeConfig,
)


class AudioTowerSubmodule(NodeSubmodule):
    """Log-mel -> one text-space embedding per 80 ms of audio.

    Runs once per request. Emits ``[n_audio_tokens, text_hidden]``, which the
    decoder both prefills against and consumes one row at a time.
    """

    def __init__(self, tower: nn.Module, projector: nn.Module,
                 config: VoxtralRealtimeConfig):
        super().__init__()
        self.audio_tower = tower
        self.multi_modal_projector = projector
        self.config = config

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList, **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, fwd_info, kwargs
        return NodeInputs(
            tensor_inputs={"input_features": inputs["input_features"][0]}
        )

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        input_features: torch.Tensor, **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        p = next(self.audio_tower.parameters())
        mel = input_features.to(device=p.device, dtype=p.dtype)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        hidden = self.audio_tower(mel)
        b, t, h = hidden.shape
        k = self.config.downsample_factor
        usable = (t // k) * k
        # downsample_factor tower frames are concatenated channel-wise into one
        # projector input: 8 mel frames in, one text-space embedding out.
        hidden = hidden[:, :usable].reshape(b, usable // k, h * k)
        return {"audio_embeds": [self.multi_modal_projector(hidden).squeeze(0)]}


class TextDecoderSubmodule(ARNodeSubmodule):
    """Autoregressive decoder, conditioned on one audio frame per position."""

    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, decoder: nn.Module, lm_head: nn.Module,
                 config: VoxtralRealtimeConfig):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    # -- inputs ---------------------------------------------------------

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList, **kwargs: Any,
    ) -> ARNodeInputs:
        del kwargs
        device = self.get_device()
        state = self.request_state(fwd_info.request_id)

        if graph_walk == "prefill":
            audio = inputs["audio_embeds"][0].to(device)
            token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
            n_prompt = int(token_ids.shape[0])
            if n_prompt >= audio.shape[0]:
                raise ValueError(
                    f"prompt ({n_prompt} tokens) is not shorter than the audio "
                    f"({audio.shape[0]} frames); the utterance is too short to "
                    "transcribe."
                )
            # Persisted for the whole request: the decode loop indexes one row
            # per step. Held on device because it is read every 80 ms.
            state.add_all(
                audio_embeds=audio,
                n_audio_tokens=int(audio.shape[0]),
                generation_step=n_prompt,
            )
            embeds = self.decoder.embed_tokens(token_ids) + audio[:n_prompt]
        else:
            step = int(state["generation_step"])
            audio = state["audio_embeds"]
            token_ids = inputs["text_inputs"][0].to(device).reshape(-1)
            row = audio[min(step, audio.shape[0] - 1)].unsqueeze(0)
            embeds = self.decoder.embed_tokens(token_ids) + row
            state.add("generation_step", step + 1)

        return ARNodeInputs(
            input_embeds=embeds,
            input_seq_len=int(embeds.shape[0]),
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """Pack the batch and record where each request's last token sits."""
        del engine_inputs
        seq_lens = [item.input_seq_len for item in inputs]
        packed = (
            inputs[0].input_embeds if len(inputs) == 1
            else torch.cat([item.input_embeds for item in inputs], dim=0)
        )
        last = torch.tensor(seq_lens, device=self.get_device()).cumsum(0) - 1
        return {"input_embeds": packed, "last_token_indices": last}

    # -- engine plan ----------------------------------------------------

    def declare_step(
        self, graph_walk: str, request_ids: list[str],
        inputs: list[ARNodeInputs], slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None, **kwargs: Any,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                TEXT_KV: KVStep(),
                TEXT_ATTN: AttentionStep(causal=True),
                TEXT_POS: PositionStep(),
                # Transcription is greedy; there are no seen-token buffers to
                # maintain, and a repetition penalty would fight the model on
                # legitimately repeated words.
                SAMPLER: SamplerStep(apply_penalty=False),
            },
        )

    # -- forward --------------------------------------------------------

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor, last_token_indices: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, kwargs
        return {
            "new_token": self._decode_batch(
                engine_inputs, input_embeds, last_token_indices
            )
        }

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor, last_token_indices: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        """Batched form of ``forward``: one packed pass, split per request.

        The whole point of the engine port. ``preprocess`` already packed every
        request's tokens into one tensor and recorded where each request's last
        token sits, so the batch is a single decoder call and a single sampler
        call regardless of how many requests are in flight.
        """
        del graph_walk, kwargs
        tokens = self._decode_batch(engine_inputs, input_embeds, last_token_indices)
        # Slice, do not index: tokens[i] is a 0-dim scalar, and the edge
        # fan-out reads tensor_info.dims[0] to size each shard. A scalar makes
        # that an IndexError deep in the distributed layer, with nothing in the
        # message connecting it back to this line.
        return {
            request_id: {"new_token": [tokens[i:i + 1]]}
            for i, request_id in enumerate(engine_inputs.request_ids)
        }

    def _decode_batch(
        self, engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor, last_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.decoder(input_embeds)
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.lm_head(last_hidden)
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        return sampler.sample(engine_inputs.request_ids, logits)

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        """Capture the decode step. Prefill is variable-length and runs once."""
        del tp_world_size
        dtype = next(self.decoder.parameters()).dtype
        return [BatchedCudaGraphConfig(
            capture_graph_walk="decode",
            single_request_inputs=ARNodeInputs(
                input_embeds=torch.zeros(
                    1, self.config.text.hidden_size, dtype=dtype, device=device,
                ),
                input_seq_len=1,
            ),
            capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
        )]

    # -- batching and stopping ------------------------------------------

    def can_use_cuda_graphs(self, batch: Any, model_inputs: list[NodeInputs]) -> bool:
        """Replay the captured decode graph.

        ``VOXTRAL_DISABLE_CUDA_GRAPHS=1`` forces the eager path. Keep it: a
        wrong transcript that appears only under graph replay is otherwise very
        hard to separate from a modelling bug, and flipping this is the fastest
        way to tell the two apart.
        """
        if os.environ.get("VOXTRAL_DISABLE_CUDA_GRAPHS"):
            return False
        if getattr(batch, "graph_walk", "") != "decode":
            return False
        return super().can_use_cuda_graphs(batch, model_inputs)

    def can_batch(self, batch: Any, model_inputs: list[NodeInputs]) -> bool:
        return (
            getattr(batch, "graph_walk", "") in {"prefill", "decode"}
            and bool(model_inputs)
            and len(model_inputs) <= self.MAX_BATCH_SIZE
        )

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], **kwargs: Any,
    ) -> None:
        """Rebind the sampled token as the next step's input.

        Metadata only -- no tensor is copied. The decode loop declares a
        recurrent ``text_inputs`` edge back into this node, but the forward
        produces ``new_token``; without this alias the next step's
        ``text_inputs`` list is empty and prepare_inputs fails on an index that
        says nothing about a missing edge.
        """
        del request_info, kwargs
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]
    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop when the audio runs out -- never on a sampled token.

        Voxtral emits one token per 80 ms of audio and pads the silence, so
        there is no EOS to wait for: the transcript is exactly as long as the
        utterance. Bounding on the audio is not a safety cap, it IS the
        termination condition, and it means no sampling parameter can change
        how much of the audio gets transcribed.
        """
        del request_info, outputs
        state = self.request_state(request_id)
        step = int(state.get("generation_step", 0))
        budget = int(state.get("n_audio_tokens", 0))
        return {"decode_loop"} if step >= budget else set()
