#!/usr/bin/env bash
# Validate the mounted checkpoint, then serve. Every failure here is one a
# operator can act on, rather than a traceback from inside worker startup.
set -uo pipefail

MODEL="${MSTAR_MODEL_PATH:-/checkpoint}"
CONFIG="${MSTAR_TTS_CONFIG:-/opt/mstar/src/configs/inflection_qwen3tts_lowlatency.yaml}"
PORT="${MSTAR_TTS_PORT:-8100}"
GPUS="${MSTAR_TTS_GPUS:-0}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$MODEL" ]] || die "checkpoint dir not found: $MODEL (mount it and set MSTAR_MODEL_PATH)"
for f in config.json generation_config.json model.safetensors speech_tokenizer/config.json; do
    [[ -e "$MODEL/$f" ]] || die "missing $f in $MODEL — not a Qwen3-TTS checkpoint?"
done
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

if [[ -f "$MODEL/pointer_head.pt" ]]; then
    echo "note: pointer_head.pt present (word timestamps land in Phase 3; M* does not read it yet)"
else
    echo "note: no pointer_head.pt — word-timestamp requests return no words"
fi

echo "model : $MODEL"
echo "config: $CONFIG"
echo "port  : $PORT   gpus: $GPUS"
exec mstar serve qwen3_tts --config "$CONFIG" --gpus "$GPUS" \
    --host 0.0.0.0 --port "$PORT" --log-level "${MSTAR_TTS_LOG_LEVEL:-INFO}" "$@"
