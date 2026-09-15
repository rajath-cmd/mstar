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
    echo "note: pointer_head.pt present — word timestamps available (timestamp_type=word)"
else
    echo "note: no pointer_head.pt — word-timestamp requests return no words"
fi

# uvicorn resolves its WebSocket implementation at bind time and, finding none,
# degrades to answering every upgrade with 404 instead of refusing to start. A
# TTS server that serves REST but silently 404s /v1/audio/speech/stream is worse
# than one that fails to boot, so make it fail to boot.
python - <<'PREFLIGHT' || die "WebSocket support missing; rebuild the image"
import sys
try:
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol as P
except Exception as exc:
    sys.exit(f"cannot resolve a uvicorn WebSocket protocol: {exc}")
if P is None or "Unsupported" in P.__name__:
    sys.exit("uvicorn has no WebSocket protocol; install websockets or wsproto")
print(f"preflight: websocket protocol = {P.__name__}")
try:
    import prometheus_client  # noqa: F401
    print("preflight: prometheus_client present, /metrics live")
except ImportError:
    print("preflight: WARNING prometheus_client absent, /metrics will be empty")
PREFLIGHT

echo "model : $MODEL"
echo "config: $CONFIG"
echo "port  : $PORT   gpus: $GPUS"
exec mstar serve qwen3_tts --config "$CONFIG" --gpus "$GPUS" \
    --host 0.0.0.0 --port "$PORT" --log-level "${MSTAR_TTS_LOG_LEVEL:-INFO}" "$@"
