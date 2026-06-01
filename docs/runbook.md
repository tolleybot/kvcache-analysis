# Runbook: Running Anywhere

How to build and run this project reproducibly on any machine, and the
environment facts learned along the way. This file is the portable source of
truth: Claude's per-machine memory does not travel with a clone, so anything an
operator (human or Claude) needs on a fresh machine lives here, not in private
memory.

## Quick start (Docker, recommended)

```bash
# 1. Build the image (see the tag matrix below if not on a recent driver)
docker build -f docker/Dockerfile -t mloss-vllm-kvcache:latest .

# 2. Start the baseline server (foreground)
bash docker/run-server.sh                      # serves Qwen2.5-3B on :8000

# 3. In another shell, run the benchmark sweep
bash docker/run-bench.sh                        # writes bench/results/*.json
```

Constrained-cache (eviction-pressure) baseline:

```bash
MAX_MODEL_LEN=2048 NUM_GPU_BLOCKS=256 bash docker/run-server.sh
LABEL=baseline_b2 MAX_MODEL_LEN=2048 bash docker/run-bench.sh
```

## Quick start (host virtualenv, no Docker)

```bash
python3 -m venv .venv
.venv/bin/pip install vllm                      # let vLLM pick its own CUDA build
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest                      # 17 tests should pass
bash scripts/gen_traces.sh
bash scripts/serve_baseline.sh                  # one shell
bash scripts/run_sweep.sh                       # another shell
bash scripts/stop_server.sh                     # when done
```

## Hardware portability: the driver -> CUDA -> tag matrix

There is no single image that runs everywhere. A CUDA runtime needs a host
driver new enough for it, so the image's CUDA build must match the node. Pick the
base tag per hardware family with `--build-arg VLLM_TAG=...` (or override
`FROM`). Confirm the host driver first with `nvidia-smi` (top-right "CUDA
Version" is the maximum the driver supports).

| Hardware | Arch | Notes |
| --- | --- | --- |
| RTX 5070 (local dev) | Blackwell sm_120 | Needs CUDA 13 (driver 595 here). Requires the FlashInfer sampler workaround below. |
| V100 (cluster) | Volta sm_70 | Broadly supported; older drivers may need a CUDA 12.x image tag. |
| H200 (cluster) | Hopper sm_90 | Well supported; match the tag to the node driver. |

If a node's driver predates the default image's CUDA, build with an older
`VLLM_TAG` whose CUDA build the driver supports, or build from a matching
`nvidia/cuda:<ver>` base and `pip install vllm`.

## Known environment gotchas

These were learned on the Blackwell dev box; the fixes are baked into the scripts
and image so they apply everywhere harmlessly.

- **FlashInfer sampler on Blackwell.** FlashInfer's JIT sampler cannot resolve
  sm_120 and aborts engine init (`FlashInfer requires GPUs with sm75 or higher`).
  Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` (set in `serve_baseline.sh` and the
  Dockerfile). Equivalent for greedy benchmarking; harmless on other GPUs.
- **CUDA stack consistency (host installs).** Do not force a CUDA wheel index
  (`--extra-index-url .../cu128`). It can pin a torch CUDA build that mismatches
  vLLM's compiled extension (`libcudart.so.NN not found`). Plain `pip install
  vllm` resolves a consistent stack.
- **Stopping the server.** vLLM v1 runs a separate `VLLM::EngineCore` worker that
  `pkill -f "vllm serve"` misses, leaving GPU memory held. Use
  `scripts/stop_server.sh`, which kills the worker too and waits for VRAM to
  release. (In Docker, stopping the container handles this.)
- **Prefix-cache metrics (vLLM 0.22.0).** `vllm:prefix_cache_queries_total` and
  `vllm:prefix_cache_hits_total`. Connector-backed caches report under
  `vllm:external_prefix_cache_*`, which is where cross-instance hits surface in
  the Stage 3 prototype.

## RDMA on the cluster (Stage 3 performance tier)

For representative numbers, run on an RDMA fabric and pass the devices into the
container:

```bash
RDMA=1 bash docker/run-server.sh
```

This adds `--network host --cap-add=IPC_LOCK --ulimit memlock=-1
--device=/dev/infiniband`. It requires the host to have OFED/rdma-core and the IB
kernel modules loaded, and the container's user-space RDMA libraries to match.
The confirmation questions and diagnostic commands for the fabric are in
`environment-checklist.md`. Without RDMA the Transfer Engine falls back to TCP,
which works but is not representative.

## Stage 3: Mooncake Store (cross-instance KV reuse)

Two vLLM instances sharing one Mooncake Store pool. Full design, recipe, and the
measured local proof are in `stage3.md`. The two requirements that gate it:

- **Mooncake wheel must match the host CUDA**, exactly like vLLM. CUDA 13 box:
  `pip install mooncake-transfer-engine-cuda13`. CUDA 12: the base
  `mooncake-transfer-engine`. Otherwise the import fails with `libcudart.so.NN`.
- **`PYTHONHASHSEED` must be identical on every instance** (set in
  `serve_mooncake.sh`, default 0). vLLM seeds block hashes per process otherwise,
  so instances never match in the store and cross-instance hits stay at zero.

Local proof (two co-located instances over TCP):

```bash
.venv/bin/pip install mooncake-transfer-engine-cuda13
bash scripts/serve_master.sh &
MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8000 BOOTSTRAP_PORT=8998 GPU_MEM_UTIL=0.38 \
  ENFORCE_EAGER=1 MOONCAKE_CONFIG_PATH=/tmp/mc_a.json bash scripts/serve_mooncake.sh &
MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8001 BOOTSTRAP_PORT=8999 GPU_MEM_UTIL=0.38 \
  ENFORCE_EAGER=1 MOONCAKE_CONFIG_PATH=/tmp/mc_b.json bash scripts/serve_mooncake.sh &
.venv/bin/python -m bench.run_xinstance --trace data/trace_xinst.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --port-a 8000 --port-b 8001 --settle-s 5
bash scripts/stop_server.sh
```

Multi-GPU node (one instance per GPU) via Docker Compose:

```bash
docker build -f docker/Dockerfile --build-arg INSTALL_MOONCAKE=1 -t mloss-vllm-kvcache:mooncake .
MODEL=Qwen/Qwen2.5-3B-Instruct docker compose -f docker/compose.mooncake.yml up
```

For the RDMA performance tier, set `MOONCAKE_PROTOCOL=rdma` and `MOONCAKE_DEVICE`
to the RNIC, and run with host networking and IB device passthrough.

## Where things are

- `docs/` evaluation rubric, candidate survey, baseline results, this runbook
- `bench/` trace generator and harness (with unit tests)
- `scripts/` host-side serve, generate, sweep, stop
- `docker/` Dockerfile and containerized run wrappers
- `bench/results/` per-run JSON output
