#!/usr/bin/env bash
# Stop a local vLLM server cleanly.
#
# vLLM v1 runs the engine in a separate "VLLM::EngineCore" worker process that
# does not match "vllm serve", so killing only the launcher leaves the worker
# holding GPU memory. Kill both, then wait for the GPU to release.
set -uo pipefail

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "entrypoints.openai" 2>/dev/null || true
pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true
# Mooncake Store master (Stage 3). Harmless if not running.
pkill -9 -f "mooncake_master" 2>/dev/null || true

for _ in 1 2 3 4 5 6 7 8 9 10; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "gpu used: ${used} MiB"
  [ "${used:-9999}" -lt 1000 ] && { echo "GPU released"; exit 0; }
  sleep 1
done
echo "WARNING: GPU memory still held; check 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'"
exit 1
