# Runbook: Running Anywhere

How to build and run this project reproducibly on any machine, and the
environment facts learned along the way. This file is the portable source of
truth: Claude's per-machine memory does not travel with a clone, so anything an
operator (human or Claude) needs on a fresh machine lives here, not in private
memory.

## Quick start (Docker, recommended)

```bash
# 1. Build the image (see the tag matrix below if not on a recent driver)
docker build -f docker/Dockerfile -t kvcache:latest .

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
| 8x A100-SXM4-80GB (`latpoc51`, benchmark tier in use) | Ampere sm_80 | Driver 570 (max CUDA 12.8). The stock `v0.22.0` image (CUDA 13 build) runs as-is through CUDA forward compatibility, verified by a kernel launch, so no tag override is needed. The base `mooncake-transfer-engine` wheel works. No FlashInfer workaround needed. |
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

## Transport for the multi-GPU tier (Stage 3 performance)

Scope is a single node, one vLLM instance per GPU sharing one Store pool. The local
proof used TCP, which works but is not representative (it measured near 16 MB/s).
The Store accepts only two protocols, `tcp` and `rdma`. NVLink is **not** an option
here: although the Transfer Engine config lists an `nvlink_intra` protocol, the
Mooncake Store client rejects it at init (`unsupported_protocol
protocol=nvlink_intra`), so the GPUs' NVLink cannot be used through the Store. For
representative numbers, use RDMA:

- **RDMA** over a GPU-affined local NIC. Set `MOONCAKE_PROTOCOL=rdma` and
  `MOONCAKE_DEVICE` to the RNIC (e.g. `mlx5_4` for a GPU4/5 pair), and pass the
  devices into the container:

  ```bash
  RDMA=1 bash docker/run-server.sh
  ```

  This adds `--network host --cap-add=IPC_LOCK --ulimit memlock=-1
  --device=/dev/infiniband`. It needs the host's OFED/rdma-core and IB kernel
  modules, the container's user-space RDMA libraries to match, and
  `nvidia_peermem` loaded if you want GPUDirect (NIC DMA straight to and from GPU
  memory) rather than a CPU bounce.

Multi-machine reuse over the cross-node InfiniBand fabric is deferred as future
work; the cross-node confirmation questions are in `environment-checklist.md`.

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

Multi-GPU node, one instance per GPU, via Docker Compose. This is exactly how the
multi-GPU cross-instance result in `report.md` and `stage3.md` was produced (8x
A100, instances on GPU 0 and GPU 1, TCP transport).

```bash
# 1. Build the Mooncake-enabled image. The stock CUDA 13 image runs on this
#    CUDA 12.8 driver via forward compatibility, so no tag override is needed, and
#    the base mooncake-transfer-engine wheel works alongside it.
docker build -f docker/Dockerfile --build-arg INSTALL_MOONCAKE=1 \
  -t kvcache:mooncake .

# 2. Bring up the master plus one instance per GPU (instance-a on GPU 0 -> :8000,
#    instance-b on GPU 1 -> :8001). Detached, so the same shell drives the test.
IMAGE=kvcache:mooncake MODEL=Qwen/Qwen2.5-3B-Instruct \
  docker compose -f docker/compose.mooncake.yml up -d

# 3. Wait for both instances to report healthy.
until curl -sf localhost:8000/health && curl -sf localhost:8001/health; do sleep 5; done

# 4. Run the cross-instance test in a client container on the host network. No
#    host virtualenv is needed: the image carries bench/ and its deps. The tools
#    are python3, not python. Results persist to bench/results/ on the host.
docker run --rm --network host -v "$(pwd)/bench/results:/app/bench/results" \
  kvcache:mooncake -lc '
    python3 -m bench.trace --num-sessions 8 --turns-per-session 2 \
      --shared-system-fraction 0.5 --system-words 300 --out /tmp/trace_xinst.jsonl
    python3 -m bench.run_xinstance --trace /tmp/trace_xinst.jsonl \
      --model Qwen/Qwen2.5-3B-Instruct --port-a 8000 --port-b 8001 --settle-s 5 \
      --out /app/bench/results/xinstance_a100_tcp.json'

# 5. Tear down and free the GPUs.
docker compose -f docker/compose.mooncake.yml down
```

`run_xinstance.py` populates the pool from instance A, then serves the same
prompts from instance B and reports B's external (cross-instance) hit rate, plus
A's own (cold) latency as the recompute control. The recorded run reached 96.7%
external hits over TCP on two A100s.

Two transport notes from running this, both pointing at RDMA (see `report.md`
Section 6.3):

- **Pool size.** The default 1 GiB segment is too small for long prefixes; a few
  thousand tokens of KV per request can exceed it and force store eviction, which
  shows up as a low hit rate. Raise it with `SEGMENT_SIZE` and `BUFFER_SIZE`
  (bytes), for example `SEGMENT_SIZE=8589934592 BUFFER_SIZE=2147483648 docker
  compose -f docker/compose.mooncake.yml up -d`.
- **TCP ceiling.** TCP transport measured near 16 MB/s here, so a cross-instance
  load can cost seconds and is slower than recomputing on a fast GPU. Large
  prefixes (around 3,300 tokens, roughly 120 MB of KV) also exhaust ephemeral TCP
  ports and the transfers fail. TCP is fine to prove the mechanism on small
  prefixes; representative performance needs `MOONCAKE_PROTOCOL=rdma`.

For the RDMA performance tier, set `MOONCAKE_PROTOCOL=rdma` and `MOONCAKE_DEVICE`
to the RNIC, and run with host networking and IB device passthrough.

## Where things are

- `docs/` evaluation rubric, candidate survey, baseline results, this runbook
- `bench/` trace generator and harness (with unit tests)
- `scripts/` host-side serve, generate, sweep, stop
- `docker/` Dockerfile and containerized run wrappers
- `bench/results/` per-run JSON output
