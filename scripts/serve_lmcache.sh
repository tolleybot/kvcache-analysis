#!/usr/bin/env bash
# Launch one vLLM instance backed by LMCache, with Mooncake Store as the L2 remote
# tier. This is survey option (c): LMCache as the cache-management layer on top of
# Mooncake. It is benchmarked head-to-head against serve_mooncake.sh (option b),
# with the same Mooncake + RDMA transport underneath, so the comparison isolates
# the LMCache layer rather than the transport. Start the Mooncake master first with
# scripts/serve_master.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
VLLM_BIN="${VLLM_BIN:-vllm}"

# Same cross-instance determinism requirement as the bare Mooncake path.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
KV_ROLE="${KV_ROLE:-kv_both}"

# Mooncake Store settings, used as LMCache's L2 remote tier.
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-mlx5_0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1:50051}"
METADATA_SERVER="${METADATA_SERVER:-P2PHANDSHAKE}"
SEGMENT_SIZE="${SEGMENT_SIZE:-8589934592}"
BUFFER_SIZE="${BUFFER_SIZE:-2147483648}"

# LMCache tier config. This matches the LMCache Mooncake docs' RDMA embedded
# example: local_cpu False, local_buffer_size 0, save_chunk_meta False and
# prefer_local_alloc true select the zero-copy RDMA path, which avoids the separate
# L1 buffer registration that otherwise fails with Mooncake error=-600 and segfaults.
LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-false}"
LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-2}"

# LMCache reads its config from a YAML file pointed to by LMCACHE_CONFIG_FILE.
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_PATH:-/tmp/lmcache_config.yaml}"
cat > "$LMCACHE_CONFIG_FILE" <<YAML
local_cpu: ${LMCACHE_LOCAL_CPU}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
remote_url: "mooncakestore://${MASTER_ADDR}/"
numa_mode: "auto"
pre_caching_hash_algorithm: sha256_cbor_64bit
extra_config:
  use_exists_sync: true
  save_chunk_meta: false
  local_hostname: "${VLLM_HOST_IP:-localhost}"
  metadata_server: "${METADATA_SERVER}"
  protocol: "${MOONCAKE_PROTOCOL}"
  device_name: "${MOONCAKE_DEVICE}"
  global_segment_size: ${SEGMENT_SIZE}
  master_server_address: "${MASTER_ADDR}"
  # The LMCache docs example uses 0 here, but Mooncake 0.3.11 rejects 0
  # ("Invalid local_buffer_size: 0, must be between 1024 and 1099511627776") and
  # the client then never initializes, which LMCache does not surface; the first
  # transfer segfaults. Keep this a valid size.
  local_buffer_size: ${BUFFER_SIZE:-1073741824}
  mooncake_prefer_local_alloc: true
YAML
echo "Wrote LMCache config to $LMCACHE_CONFIG_FILE:"; cat "$LMCACHE_CONFIG_FILE"

KV_TRANSFER_CONFIG="{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"${KV_ROLE}\"}"

EXTRA_ARGS=()
[[ "${ENFORCE_EAGER:-0}" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)

echo "Serving $MODEL on :$PORT (LMCacheConnectorV1, L2=mooncake protocol=$MOONCAKE_PROTOCOL)"
exec "$VLLM_BIN" serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  "${EXTRA_ARGS[@]}"
