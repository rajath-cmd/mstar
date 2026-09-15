#!/usr/bin/env bash
# Validate the mounted checkpoint, then serve. Every failure here is one an
# operator can act on, rather than a traceback from inside model loading.
set -uo pipefail

MODEL="${VOXTRAL_MODEL_PATH:-/checkpoint}"
PORT="${VOXTRAL_PORT:-8200}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$MODEL" ]] || die "checkpoint dir not found: $MODEL (mount it and set VOXTRAL_MODEL_PATH)"
for f in config.json model.safetensors tekken.json; do
    [[ -e "$MODEL/$f" ]] || die "missing $f in $MODEL — not a Voxtral-Realtime checkpoint?"
done

# tekken.json is load-bearing twice over: it is the tokenizer AND the source of
# the audio padding that aligns mel frames to text positions. A checkpoint
# without it cannot be served at all, so say so here rather than at first request.
echo "model : $MODEL"
echo "port  : $PORT"
exec python -m mstar.model.voxtral_rt.serving.app \
    --model "$MODEL" --host 0.0.0.0 --port "$PORT" "$@"
