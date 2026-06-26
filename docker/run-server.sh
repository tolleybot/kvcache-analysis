#!/usr/bin/env bash
# Launch the baseline vLLM server inside the container with GPU access.
#
# Environment knobs (all optional):
#   IMAGE          container image (default kvcache:latest)
#   PORT           host port to publish (default 8000)
#   RDMA=1         enable RDMA device passthrough for cluster runs
#   DETACH=1       run detached instead of foreground
#   MODEL, MAX_MODEL_LEN, GPU_MEM_UTIL, NUM_GPU_BLOCKS, PREFIX_CACHING
#                  passed through to scripts/serve_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-kvcache:latest}"
PORT="${PORT:-8000}"

RUN_FLAGS=(--rm --gpus all --shm-size=8g)
[[ "${DETACH:-0}" == "1" ]] && RUN_FLAGS+=(-d) || RUN_FLAGS+=(-it)

# Cluster RDMA: expose InfiniBand devices and allow pinned memory. Requires the
# host to have OFED/rdma-core and the IB kernel modules loaded. With host
# networking the published port is ignored. See docs/runbook.md.
if [[ "${RDMA:-0}" == "1" ]]; then
  RUN_FLAGS+=(--network host --cap-add=IPC_LOCK --ulimit memlock=-1 --device=/dev/infiniband)
else
  RUN_FLAGS+=(-p "${PORT}:8000")
fi

docker run "${RUN_FLAGS[@]}" \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e MODEL -e MAX_MODEL_LEN -e GPU_MEM_UTIL -e NUM_GPU_BLOCKS -e PREFIX_CACHING \
  "$IMAGE" -lc "bash scripts/serve_baseline.sh"
