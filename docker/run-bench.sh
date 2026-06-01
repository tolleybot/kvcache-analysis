#!/usr/bin/env bash
# Run the benchmark sweep inside a container against an already-running server.
#
# Uses host networking so it reaches a server published on localhost:PORT (the
# default run-server.sh mode). Results are written to bench/results/ on the host
# via a bind mount.
#
# Environment knobs:
#   IMAGE   container image (default mloss-vllm-kvcache:latest)
#   HOST    server host as seen from the container (default localhost)
#   PORT    server port (default 8000)
#   LABEL, MODEL, CONCURRENCY, MAX_MODEL_LEN  passed to scripts/run_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-mloss-vllm-kvcache:latest}"

docker run --rm --network host \
  -v "$(pwd)/bench/results:/app/bench/results" \
  -v "$(pwd)/data:/app/data" \
  -e HOST="${HOST:-localhost}" -e PORT="${PORT:-8000}" \
  -e LABEL -e MODEL -e CONCURRENCY -e MAX_MODEL_LEN \
  "$IMAGE" -lc "bash scripts/run_sweep.sh"
