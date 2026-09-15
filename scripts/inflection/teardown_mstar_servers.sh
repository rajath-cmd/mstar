#!/usr/bin/env bash
# Stop M* dev servers launched from THIS checkout and reclaim their GPUs.
#
# Why this exists: killing a server's parent process leaves its engine workers
# reparented to init, still holding tens of GB of VRAM AND the conductor's IPC
# resources. The next server on that GPU then starts, reports healthy, accepts
# requests -- and never executes them, with the GPU at 0%. Diagnosing that as a
# code bug costs hours; it is the single most expensive trap in this workflow.
#
# So: kill the whole process TREE, then sweep any surviving CUDA process that
# belongs to this checkout, then verify the memory actually came back.
#
#   scripts/inflection/teardown_mstar_servers.sh            # all dev servers
#   scripts/inflection/teardown_mstar_servers.sh 8103 8108  # specific ports
#
# Containers are left alone -- stop those with `docker rm -f`.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

kill_tree() {
    local pid=$1
    local kids
    kids=$(pgrep -P "$pid" 2>/dev/null)
    for k in $kids; do kill_tree "$k"; done
    kill -9 "$pid" 2>/dev/null && echo "  killed $pid"
}

targets=()
if [[ $# -gt 0 ]]; then
    for port in "$@"; do
        while read -r p; do [[ -n "$p" ]] && targets+=("$p"); done < <(
            pgrep -f "mstar serve .*--port $port" 2>/dev/null)
    done
else
    while read -r p; do [[ -n "$p" ]] && targets+=("$p"); done < <(
        pgrep -f "$REPO/.venv/bin/mstar serve" 2>/dev/null)
    while read -r p; do [[ -n "$p" ]] && targets+=("$p"); done < <(
        pgrep -f "mstar.model.voxtral_rt.serving.app" 2>/dev/null)
fi

echo "stopping ${#targets[@]} server process(es)"
for p in "${targets[@]}"; do kill_tree "$p"; done
sleep 8

# Sweep orphans: CUDA processes from this checkout that belong to NO live
# server.
#
# Do NOT use "ppid == 1" as the orphan test. Servers are launched with setsid
# so they survive the shell that started them, which means a perfectly healthy
# server's top process ALSO has ppid 1 -- that test kills the very servers you
# meant to keep, and the failure looks like the next A/B arm mysteriously
# refusing connections.
#
# The correct test: build the descendant set of every live `mstar serve`, and
# sweep only what is outside it.
echo "sweeping orphaned CUDA processes from $REPO"
live_tree=" "
collect() {
    local pid=$1
    live_tree+="$pid "
    for k in $(pgrep -P "$pid" 2>/dev/null); do collect "$k"; done
}
for root in $(pgrep -f "$REPO/.venv/bin/mstar serve" 2>/dev/null) \
            $(pgrep -f "mstar.model.voxtral_rt.serving.app" 2>/dev/null); do
    collect "$root"
done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    cmd=$(ps -o cmd= -p "$p" 2>/dev/null) || continue
    [[ "$cmd" == *"$REPO"* ]] || continue
    [[ "$live_tree" == *" $p "* ]] && continue
    kill -9 "$p" 2>/dev/null && echo "  swept orphan $p"
done
sleep 6

echo "GPU memory after teardown:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu/'
