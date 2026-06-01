#!/usr/bin/env bash
# Start the Mooncake Store master server, which coordinates object metadata and
# location for the distributed KV pool. Both vLLM instances point their
# master_server_address at this process. Start it before the instances.
#
# On a host virtualenv the CUDA runtime comes only from pip wheels, so the
# master binary cannot find libcudart unless we add the wheel's lib dir to the
# loader path. In a container with a system CUDA install this is a no-op.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDART_DIR="$(dirname "$(find .venv -name 'libcudart.so.1*' 2>/dev/null | head -1)" 2>/dev/null || true)"
if [[ -n "$CUDART_DIR" ]]; then
  export LD_LIBRARY_PATH="${CUDART_DIR}:${LD_LIBRARY_PATH:-}"
fi

MASTER_BIN="${MASTER_BIN:-.venv/bin/mooncake_master}"
MASTER_PORT="${MASTER_PORT:-50051}"
METRICS_PORT="${MASTER_METRICS_PORT:-9003}"

echo "Starting mooncake_master on :${MASTER_PORT} (metrics :${METRICS_PORT})"
exec "$MASTER_BIN" \
  --port "$MASTER_PORT" \
  --metrics_port "$METRICS_PORT" \
  --enable_metric_reporting=true
