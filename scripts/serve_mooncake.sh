#!/usr/bin/env bash
# Launch one vLLM instance backed by the Mooncake Store distributed KV cache.
#
# Two such instances pointed at the same master server share one KV pool, which
# is what gives cross-instance prefix reuse. Each instance uses kv_role=kv_both
# by default so it both writes and reads the pool (either instance can serve any
# request). This is the Stage 3 option (b): full Mooncake Store via KVConnector.
#
# The Mooncake Store JSON config is generated from environment variables so the
# same script serves a local TCP test and a cluster RDMA run. Start the master
# first with scripts/serve_master.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
VLLM_BIN="${VLLM_BIN:-.venv/bin/vllm}"

# CRITICAL for cross-instance reuse: vLLM seeds its block-hash chain from
# NONE_HASH, which is os.urandom(32) per process unless PYTHONHASHSEED is set
# (vllm/v1/core/kv_cache_utils.py). Two instances with different seeds compute
# different block hashes and can never match in the shared store. All instances
# sharing a Mooncake pool MUST use the same value.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Two Store instances on one host must not share the transfer-engine handshake
# port. Override BOOTSTRAP_PORT per instance (e.g. 8998 and 8999).
export VLLM_MOONCAKE_BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

# On a host virtualenv the CUDA runtime is only in pip wheels; make libcudart
# discoverable so the Mooncake transfer-engine extension loads.
CUDART_DIR="$(dirname "$(find .venv -name 'libcudart.so.1*' 2>/dev/null | head -1)" 2>/dev/null || true)"
if [[ -n "$CUDART_DIR" ]]; then
  export LD_LIBRARY_PATH="${CUDART_DIR}:${LD_LIBRARY_PATH:-}"
fi

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
KV_ROLE="${KV_ROLE:-kv_both}"

# Mooncake Store settings (see vLLM mooncake/store/worker.py for the schema).
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"        # tcp | rdma | efa
MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-}"               # RDMA NIC name(s); empty for tcp
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1:50051}"
METADATA_SERVER="${METADATA_SERVER:-P2PHANDSHAKE}"   # P2PHANDSHAKE needs no extra service
SEGMENT_SIZE="${SEGMENT_SIZE:-1073741824}"           # 1 GiB store segment (embedded mode)
BUFFER_SIZE="${BUFFER_SIZE:-1073741824}"             # 1 GiB local transfer buffer

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/mooncake_config.json}"
cat > "$MOONCAKE_CONFIG_PATH" <<JSON
{
  "metadata_server": "${METADATA_SERVER}",
  "master_server_address": "${MASTER_ADDR}",
  "protocol": "${MOONCAKE_PROTOCOL}",
  "device_name": "${MOONCAKE_DEVICE}",
  "mode": "embedded",
  "global_segment_size": ${SEGMENT_SIZE},
  "local_buffer_size": ${BUFFER_SIZE},
  "enable_offload": false
}
JSON
echo "Wrote Mooncake config to $MOONCAKE_CONFIG_PATH:"
cat "$MOONCAKE_CONFIG_PATH"

KV_TRANSFER_CONFIG="{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_connector_extra_config\":{\"load_async\":true}}"

EXTRA_ARGS=()
# enforce-eager skips CUDA graph capture, which saves GPU memory and startup
# time. Useful when co-locating two instances on one GPU for a local test.
[[ "${ENFORCE_EAGER:-0}" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)

echo "Serving $MODEL on :$PORT (kv_role=$KV_ROLE, protocol=$MOONCAKE_PROTOCOL)"
exec "$VLLM_BIN" serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  "${EXTRA_ARGS[@]}"
