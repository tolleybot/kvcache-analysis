#!/usr/bin/env bash
# Launch a native vLLM server for baseline measurement.
#
# Two baseline modes, selected by environment variable:
#   (default)            ample KV cache, prefix caching on  -> the "fits" ceiling
#   NUM_GPU_BLOCKS=<n>   cap KV blocks to force eviction     -> per-instance limit
#
# Prefix caching is on by default in vLLM v1; we set it explicitly for clarity
# and pin the hash algorithm to sha256, which is the content-addressing the
# distributed layers also rely on (see docs/survey.md).
set -euo pipefail

cd "$(dirname "$0")/.."

# FlashInfer's JIT sampler cannot resolve the Blackwell (sm_120) arch on this
# CUDA build and aborts engine init. We use greedy decoding for benchmarking, so
# the native PyTorch sampler is equivalent and avoids the failing JIT path.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

ARGS=(
  "$MODEL"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --enable-prefix-caching
  --prefix-caching-hash-algo sha256
)

# Optional eviction-pressure mode.
if [[ -n "${NUM_GPU_BLOCKS:-}" ]]; then
  ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS")
fi

# Optional: disable prefix caching entirely for a no-cache reference point.
if [[ "${PREFIX_CACHING:-on}" == "off" ]]; then
  ARGS+=(--no-enable-prefix-caching)
fi

# VLLM_BIN lets the same script run on the host (.venv) and in the container
# (system vllm). Defaults to the local virtualenv.
VLLM_BIN="${VLLM_BIN:-.venv/bin/vllm}"

echo "Serving: $VLLM_BIN serve ${ARGS[*]}"
exec "$VLLM_BIN" serve "${ARGS[@]}"
