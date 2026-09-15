#!/usr/bin/env bash
# Serve a local Qwen3-TTS checkpoint on M* for parity/benchmark work.
#
#   scripts/inflection/launch_mstar_qwen3_tts.sh [PORT]
#
# Env:
#   MSTAR_MODEL_PATH  checkpoint dir (must hold config.json, generation_config.json,
#                     model.safetensors, speech_tokenizer/config.json)
#   GPUS              --gpus value (default: 0)
#   LOG               log path
#
# The venv's bin MUST lead PATH. FlashInfer JIT-compiles attention kernels and
# shells out to `ninja`; it is installed in .venv/bin, and without it on PATH
# every capture fails with `FileNotFoundError: ninja` — the server then either
# falls back or hangs before binding, with the real cause buried mid-log.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$PWD"

# CONFIG selects the deployment profile: the default is tuned for batch
# throughput, inflection_qwen3tts_lowlatency.yaml for voice agents (see the
# codec_chunk_frames note in that file).
CONFIG="${CONFIG:-$REPO/configs/inflection_qwen3tts.yaml}"
PORT="${1:-${PORT:-8100}}"
GPUS="${GPUS:-0}"
LOG="${LOG:-$REPO/logs/mstar-$PORT.log}"
mkdir -p "$(dirname "$LOG")"

if [[ -z "${MSTAR_MODEL_PATH:-}" ]]; then
    echo "ERROR: set MSTAR_MODEL_PATH to a Qwen3-TTS checkpoint directory" >&2
    exit 1
fi
for f in config.json generation_config.json model.safetensors speech_tokenizer/config.json; do
    [[ -e "$MSTAR_MODEL_PATH/$f" ]] || { echo "ERROR: missing $f in $MSTAR_MODEL_PATH" >&2; exit 1; }
done
if [[ ! -f "$MSTAR_MODEL_PATH/pointer_head.pt" ]]; then
    echo "NOTE: no pointer_head.pt — word-timestamp requests will return no words" >&2
fi

export PATH="$REPO/.venv/bin:$PATH"
export HF_TOKEN_PATH=/dev/null
export MSTAR_MODEL_PATH

echo "model : $MSTAR_MODEL_PATH"
echo "port  : $PORT   gpus: $GPUS"
echo "config: $CONFIG"
echo "log   : $LOG"

# Isolate this server's ZMQ IPC namespace BY PORT.
#
# `mstar serve` defaults the prefix to /tmp/mstar_$USER/ -- per USER, not per
# SERVER. Two servers started by the same user therefore bind the same
# /tmp/mstar_$USER/worker_0.ipc, conductor.ipc and api_server.ipc, and they
# steal each other's messages: a request POSTed to server B gets executed by
# server A's engine, under A's config, while B waits forever.
#
# Nothing errors. Both servers report healthy, /v1/models answers, and only
# requests hang -- which reads exactly like a model bug. Anything that runs two
# M* servers on one box (an A/B test, or serving Qwen3-TTS and Voxtral-RT side
# by side) MUST set this.
SOCKET_PREFIX="${MSTAR_SOCKET_PREFIX:-/tmp/mstar_${USER:-u}_${PORT}/}"
UPLOAD_DIR="${MSTAR_UPLOAD_DIR:-/tmp/mstar_uploads_${USER:-u}_${PORT}/}"
mkdir -p "$SOCKET_PREFIX" "$UPLOAD_DIR"
echo "ipc   : $SOCKET_PREFIX"

exec "$REPO/.venv/bin/mstar" serve qwen3_tts \
    --config "$CONFIG" \
    --gpus "$GPUS" --host 0.0.0.0 --port "$PORT" \
    --socket-path-prefix "$SOCKET_PREFIX" \
    --upload-dir "$UPLOAD_DIR" \
    --log-level INFO >"$LOG" 2>&1
