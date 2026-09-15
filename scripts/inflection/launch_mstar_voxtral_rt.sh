#!/usr/bin/env bash
# Serve a local Voxtral-Realtime checkpoint on M*'s Walk Graph engine.
#
#   MSTAR_MODEL_PATH=/path/to/Voxtral-Mini-4B-Realtime-2602 \
#     scripts/inflection/launch_mstar_voxtral_rt.sh [PORT]
#
# Env:
#   MSTAR_MODEL_PATH  checkpoint dir (config.json, model.safetensors, tekken.json)
#   GPUS              --gpus value (default: 0)
#   LOG               log path
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$PWD"

CONFIG="${CONFIG:-$REPO/configs/inflection_voxtral_rt.yaml}"
PORT="${1:-${PORT:-8300}}"
GPUS="${GPUS:-0}"
LOG="${LOG:-$REPO/logs/voxtral-$PORT.log}"
mkdir -p "$(dirname "$LOG")"

if [[ -z "${MSTAR_MODEL_PATH:-}" ]]; then
    echo "ERROR: set MSTAR_MODEL_PATH to a Voxtral-Realtime checkpoint directory" >&2
    exit 1
fi
for f in config.json model.safetensors tekken.json; do
    [[ -e "$MSTAR_MODEL_PATH/$f" ]] || { echo "ERROR: missing $f in $MSTAR_MODEL_PATH" >&2; exit 1; }
done

export PATH="$REPO/.venv/bin:$PATH"
export HF_TOKEN_PATH=/dev/null
export MSTAR_MODEL_PATH
export VOXTRAL_MODEL_PATH="$MSTAR_MODEL_PATH"

# Isolate this server's ZMQ IPC namespace BY PORT. `mstar serve` defaults it to
# /tmp/mstar_$USER/ -- per USER, not per SERVER -- so two servers started by the
# same user bind the same sockets and steal each other's requests. Serving
# Voxtral-RT alongside Qwen3-TTS on one box is exactly that case.
SOCKET_PREFIX="${MSTAR_SOCKET_PREFIX:-/tmp/mstar_${USER:-u}_${PORT}/}"
UPLOAD_DIR="${MSTAR_UPLOAD_DIR:-/tmp/mstar_uploads_${USER:-u}_${PORT}/}"
mkdir -p "$SOCKET_PREFIX" "$UPLOAD_DIR"

echo "model : $MSTAR_MODEL_PATH"
echo "port  : $PORT   gpus: $GPUS"
echo "config: $CONFIG"
echo "ipc   : $SOCKET_PREFIX"
echo "log   : $LOG"

exec "$REPO/.venv/bin/mstar" serve voxtral_rt \
    --config "$CONFIG" \
    --gpus "$GPUS" --host 0.0.0.0 --port "$PORT" \
    --socket-path-prefix "$SOCKET_PREFIX" \
    --upload-dir "$UPLOAD_DIR" \
    --log-level INFO >"$LOG" 2>&1
