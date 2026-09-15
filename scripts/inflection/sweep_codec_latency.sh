#!/usr/bin/env bash
# Phase 5: sweep codec streaming cadence for the best low-latency config.
#
# chunk_frames is the TTFA floor (the Codec waits for that many 12.5 Hz frames
# before emitting anything) and left_context_frames is the vocoder warm-up
# overlap. Lower chunk = faster first audio, more decoder invocations, less
# throughput. Lower left_context risks audible seams at chunk boundaries.
#
# Each arm is checked for CORRECTNESS as well as speed: LeftContextChunkPolicy
# requires chunk > left_context, and a violating config silently drops the start
# of the stream rather than failing (see that class). The sweep compares audio
# duration against a reference and transcribes, so a truncating config is caught
# instead of winning on TTFA.
#
#   MSTAR_MODEL_PATH=<ckpt> scripts/inflection/sweep_codec_latency.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$PWD"
PORT="${PORT:-8100}"
GPUS="${GPUS:-0}"
OUT="${OUT:-$REPO/out/codec_sweep}"
ARMS="${ARMS:-300:25 32:8 16:4 8:4 8:2 4:2 2:1}"
mkdir -p "$OUT"
[[ -n "${MSTAR_MODEL_PATH:-}" ]] || { echo "set MSTAR_MODEL_PATH" >&2; exit 1; }

kill_server() {
  local p
  p=$(ps -eo pid,args --no-headers | grep "mstar serve qwen3_tts" | grep -v "grep\|snapshot-bash" | awk '{print $1}' | head -1)
  [[ -n "$p" ]] && kill -9 -"$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null
  sleep 5
}

for arm in $ARMS; do
  chunk="${arm%%:*}"; ctx="${arm##*:}"
  echo "=============================================="
  echo "arm: chunk_frames=$chunk left_context_frames=$ctx  (TTFA floor ~$(python3 -c "print(f'{$chunk*0.08:.2f}')")s)"
  cfg="$OUT/cfg_${chunk}_${ctx}.yaml"
  sed -e "s/^  codec_chunk_frames:.*/  codec_chunk_frames: $chunk/" \
      -e "s/^  codec_left_context_frames:.*/  codec_left_context_frames: $ctx/" \
      "$REPO/configs/inflection_qwen3tts_lowlatency.yaml" > "$cfg"
  kill_server
  rm -f "$REPO/logs/mstar-$PORT.log"
  CONFIG="$cfg" GPUS="$GPUS" setsid nohup "$REPO/scripts/inflection/launch_mstar_qwen3_tts.sh" "$PORT" >/dev/null 2>&1 &
  ok=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! ps -eo args --no-headers | grep -q "[m]star serve qwen3_tts"; then break; fi
    sleep 10
  done
  if [[ "$ok" != "1" ]]; then
    echo "  SERVER FAILED TO START — likely a rejected chunk policy:"
    grep -oE "ValueError: LeftContextChunkPolicy.{0,120}" "$REPO/logs/mstar-$PORT.log" | head -1
    echo "{\"chunk\":$chunk,\"ctx\":$ctx,\"failed\":true}" > "$OUT/arm_${chunk}_${ctx}.json"
    continue
  fi
  "$REPO/.venv/bin/python" -m benchmark.inflection.tts_ws_ttfa \
      --mstar "http://127.0.0.1:$PORT" --concurrencies 1,8,32 --reps 3 \
      --out "$OUT/arm_${chunk}_${ctx}" 2>&1 | grep -E "c=|wrote results" | head -5
  "$REPO/.venv/bin/python" "$REPO/scripts/inflection/check_audio_complete.py" \
      --url "http://127.0.0.1:$PORT" --out "$OUT/arm_${chunk}_${ctx}/audio_check.json" || true
done
kill_server
echo "=============================================="
"$REPO/.venv/bin/python" "$REPO/scripts/inflection/summarize_codec_sweep.py" "$OUT"
