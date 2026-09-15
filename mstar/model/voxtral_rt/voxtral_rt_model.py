"""VoxtralRealtimeModel: vanilla Voxtral-Realtime on M*'s Walk Graph.

Two nodes, one partition:

    audio_tower (one-shot)  log-mel -> one text-space embedding per 80 ms
    decoder     (ar)        paged self-attention; one audio frame added per step

Graph walks:
    prefill  audio_tower -> decoder over the prompt positions; the engine
             samples the first transcript token from the logits
    decode   decoder loop; each step feeds back the sampled token and adds the
             next audio embedding

Why there is no cross-attention here, unlike Whisper: Voxtral does not attend
to the audio, it is SUMMED into the token embedding position by position. The
sequence length is the audio length -- 12.5 positions per second -- and the
transcript is whatever the model emits at those positions instead of
``[STREAMING_PAD]``. A request therefore stops when its audio runs out, not on
an EOS token, which is why ``check_stop`` bounds on the audio and no sampling
parameter can change how much of an utterance gets transcribed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.voxtral_rt.config import (
    SAMPLER,
    TEXT_ATTN,
    TEXT_KV,
    TEXT_POS,
    VoxtralRealtimeConfig,
)

logger = logging.getLogger(__name__)


class VoxtralRealtimeModel(Model):
    """Vanilla ``Voxtral-Mini-4B-Realtime-2602`` (no auxiliary heads)."""

    def __init__(self, model_path_hf: str, cache_dir: str | None = None, **kwargs):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf
        self.local_dir = self._resolve_model_path(model_path_hf)
        self.config = VoxtralRealtimeConfig.from_pretrained(self.local_dir)
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        for required in ("config.json", "model.safetensors", "tekken.json"):
            if not (Path(self.local_dir) / required).is_file():
                raise FileNotFoundError(
                    f"missing {required} in {self.local_dir}. tekken.json is not "
                    "optional: it is the reference tokenizer AND the source of "
                    "the audio padding that aligns mel frames to text positions."
                )
        del kwargs

    @staticmethod
    def _resolve_model_path(model_path_hf: str) -> str:
        """Prefer a local directory; fall back to an HF snapshot.

        A local path is the normal case here -- the checkpoint is large and
        already on disk -- so check that before reaching for the network.
        """
        import os

        for candidate in (model_path_hf, os.environ.get("VOXTRAL_MODEL_PATH")):
            if candidate and Path(candidate).is_dir():
                return str(candidate)
        from huggingface_hub import snapshot_download

        return str(snapshot_download(repo_id=model_path_hf))

    # -- resources ------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Paged KV + attention/position/sampler for the decoder only.

        The audio tower declares nothing: it runs one dense causal pass per
        request and keeps no state between calls, so there is no cache for the
        engine to plan.
        """
        text = self.config.text
        kv = KVConfig(
            num_layers=text.num_hidden_layers,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            # A request is exactly audio_seconds * 12.5 positions long, so the
            # sliding window (8192 = ~11 minutes of audio) is a far tighter and
            # more honest bound than max_position_embeddings (131072).
            max_seq_len=text.sliding_window,
            num_qo_heads=text.num_attention_heads,
        )
        return [
            KVSpec(resource_key=TEXT_KV, nodes={"decoder"}, config=kv),
            AttentionSpec(
                resource_key=TEXT_ATTN, nodes={"decoder"},
                config=AttentionConfig(kv_cache=TEXT_KV),
            ),
            PositionSpec(
                resource_key=TEXT_POS, nodes={"decoder"},
                config=PositionConfig(kv_cache=TEXT_KV),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"decoder"},
                vocab_size=text.vocab_size,
                # Transcription is greedy, and a repetition penalty would fight
                # the model on legitimately repeated words.
                enable_repetion_penalty=False,
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                temperature=float(model_kwargs.get("temperature", 0.0)),
                top_p=float(model_kwargs.get("top_p", 1.0)),
                # There is no EOS to ignore: the decode budget is the audio.
                ignore_eos=True,
            )
        }

    # -- walk graph -----------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs) -> int:
        """Upper bound only; the real stop is the audio running out.

        Sized to the decoder's sliding window, which is ~11 minutes of audio at
        12.5 Hz -- far beyond any single utterance this server is asked for.
        """
        cap = self.config.text.sliding_window
        return int(min(model_kwargs.get("max_output_tokens", cap), cap))

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name="audio_tower",
                input_names=["input_features"],
                outputs=[GraphEdge(next_node="decoder", name="audio_embeds")],
            ),
            GraphNode(
                name="decoder",
                input_names=["audio_embeds", "text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT, name="new_token",
                        output_modality="text", persist=True,
                    ),
                ],
            ),
        ])
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="decoder",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT, name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(next_node="decoder", name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return dict(prefill=prefill, decode=decode)

    # -- forward pass args ----------------------------------------------

    def get_initial_forward_pass_args(
        self, partition_name: str, input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        del partition_name, model_kwargs
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )
        inputs = []
        for node, name in (("audio_tower", "input_features"), ("decoder", "text_inputs")):
            edge = GraphEdge(next_node=node, name=name)
            edge.tensor_info = input_signals.get(name, [])
            inputs.append(edge)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([e.tensor_info for e in inputs], start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self, partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections=None,
    ) -> ForwardPassArgs:
        """prefill -> decode loop -> done."""
        del partition_name, incoming_connections
        metadata = partition_metadata
        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[],
                request_done=True,
            )
        edge = GraphEdge(next_node="decoder", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[edge],
            unpersist_tensors=list(edge.tensor_info),
            step_metadata={"is_prefill": False},
        )

    # -- prompt / output ------------------------------------------------

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> NameToTensorList:
        """Waveform -> (prompt token ids, log-mel).

        ``prompt`` is unused: the prompt carries no text, only positions. The
        text side is ``<s>`` followed by ``[STREAMING_PAD]``.
        """
        del prompt, input_modalities, output_modalities
        from mstar.model.voxtral_rt.audio_frontend import prepare

        audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(audio_inputs) != 1:
            raise ValueError(
                f"Voxtral-Realtime expects exactly one audio input per request; "
                f"got {len(audio_inputs)}."
            )
        wav = audio_inputs[0].detach().cpu().float().numpy()
        out = prepare(
            wav, int(kwargs.get("sampling_rate", self.config.sampling_rate)),
            self.local_dir, self.config,
        )
        return {
            "input_features": [out.input_features],
            "text_inputs": [out.input_ids],
        }

    def load_audio(self, filepath: str, device: str):
        """Decode an upload to mono float32 at 16 kHz, without torchcodec.

        The base implementation uses ``torchcodec.AudioDecoder``, which dlopens
        FFmpeg's shared libraries and fails hard where they are absent -- the
        error is a 40-line libtorchcodec traceback that says nothing about
        audio. Voxtral only ever wants 16 kHz mono, which ``soundfile`` (already
        a dependency) reads directly, so the heavier decoder buys nothing here.
        """
        import numpy as np
        import soundfile as sf

        from mstar.model.voxtral_rt.audio_frontend import resample

        del device
        audio, sr = sf.read(filepath, dtype="float32", always_2d=False)
        if audio.ndim > 1:  # mixdown; the tower is single channel
            audio = audio.mean(axis=-1)
        audio = resample(np.asarray(audio), int(sr), self.config.sampling_rate)
        return TensorAndMetadata(
            data=torch.from_numpy(audio),
            metadata={"sample_rate": self.config.sampling_rate, "num_channels": 1},
        )

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        del kwargs
        if modality != "text":
            raise ValueError(f"Unsupported modality for Voxtral-Realtime: {modality!r}")
        from mstar.model.voxtral_rt.audio_frontend import decode

        ids = output.reshape(-1).tolist()
        return decode(self.local_dir, ids).encode("utf-8")

    # -- submodules -----------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        dtype = autocast_dtype or torch.bfloat16
        if node_name == "audio_tower":
            sub = self._build_audio_tower(device, dtype)
        elif node_name == "decoder":
            sub = self._build_decoder(device, dtype)
        else:
            sub = None
        if sub is not None:
            logger.info("loaded Voxtral-Realtime submodule for %s", node_name)
        self._submodule_cache[node_name] = sub
        return sub

    def _build_audio_tower(self, device: str, dtype: torch.dtype) -> NodeSubmodule:
        from mstar.model.voxtral_rt.components.audio_tower import AudioTower
        from mstar.model.voxtral_rt.components.projector import Projector
        from mstar.model.voxtral_rt.submodules import AudioTowerSubmodule
        from mstar.model.voxtral_rt.weights import load_into

        with torch.device("meta"):
            tower = AudioTower(self.config.audio)
            projector = Projector(self.config)
        for mod in (tower, projector):
            mod.to(dtype)
            mod.to_empty(device=device)
        load_into(
            {"audio_tower.": tower, "multi_modal_projector.": projector},
            self.local_dir, device=device,
        )
        tower.eval()
        projector.eval()
        return AudioTowerSubmodule(tower, projector, self.config)

    def _build_decoder(self, device: str, dtype: torch.dtype) -> NodeSubmodule:
        from torch import nn

        from mstar.model.voxtral_rt.components.engine_decoder import EngineTextDecoder
        from mstar.model.voxtral_rt.components.text_decoder import TimeEmbedding
        from mstar.model.voxtral_rt.submodules import TextDecoderSubmodule
        from mstar.model.voxtral_rt.weights import load_into

        text = self.config.text
        with torch.device("meta"):
            decoder = EngineTextDecoder(text)
            time_embedding = TimeEmbedding(text.hidden_size)
        decoder = decoder.to(dtype)
        decoder.to_empty(device=device)
        load_into({"language_model.": decoder}, self.local_dir, device=device)

        # Embeddings are tied; the checkpoint stores them once.
        lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        lm_head.to(dtype)
        lm_head.to_empty(device=device)
        lm_head.weight = decoder.embed_tokens.weight

        # to_empty() allocated buffer storage WITHOUT initialising it, and
        # inv_freq is non-persistent so no checkpoint key restores it. Skipping
        # this does not crash: the delay conditioning is computed from noise and
        # the model transcribes the right words at the wrong times.
        time_embedding.to_empty(device=device)
        time_embedding.reset_buffer(device=device, dtype=dtype)
        t = torch.full(
            (1,), float(self.config.default_num_delay_tokens),
            device=device, dtype=dtype,
        )
        with torch.no_grad():
            t_cond = time_embedding(t)[None, ...]
            decoder.t_scales = [
                layer.ada_rms_norm(t_cond) for layer in decoder.layers
            ]
        decoder.eval()
        return TextDecoderSubmodule(decoder, lm_head, self.config)
