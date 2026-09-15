#!/usr/bin/env python
"""Re-record the HF reference output that test/inflection/voxtral asserts against.

Run this ONLY when the checkpoint changes. It must run in an environment whose
`transformers` ships `voxtral_realtime` (5.16+), which is deliberately NOT the
environment M* pins -- the whole point of porting the front end was to avoid
upgrading transformers under the Qwen3-TTS stack. In practice that means the
vllm-realtime venv:

    ~/workspace/vllm-realtime/.venv/bin/python \\
        scripts/inflection/voxtral_record_reference.py \\
        --model /path/to/Voxtral-Mini-4B-Realtime-2602 \\
        --audio-dir test/inflection/voxtral/reference/audio \\
        --out test/inflection/voxtral/reference/hf_reference.json

Recording reference output from the implementation under test would be
circular, which is exactly why this script does not import mstar.
"""

import argparse
import glob
import json
import os

import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model = (
        VoxtralRealtimeForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )

    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(args.audio_dir, "*.wav"))):
        audio, _sr = sf.read(path)
        inputs = proc(audio, return_tensors="pt").to(model.device, dtype=model.dtype)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8192)
        n_prompt = inputs["input_ids"].shape[1]
        name = os.path.basename(path)
        out[name] = {
            "text": proc.batch_decode(gen, skip_special_tokens=True)[0],
            "new_token_ids": gen[0, n_prompt:].tolist(),
        }
        print(f"{name}: {len(out[name]['new_token_ids'])} tokens  {out[name]['text'][:60]!r}")

    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
