# Stage 3: Mooncake Store Prototype (Cross-Instance KV Reuse)

Status: the integration works and cross-instance reuse is proven, first on the
local development box (two co-located instances on one GPU) and then on the
benchmark node (two instances, one per GPU, on an 8x A100 box). Both runs reached
a **96.7% external (cross-instance) cache hit rate** on instance B for prefixes
computed by instance A. This is the distributed value that single-instance prefix
caching (Stage 2 baseline) cannot provide. The exact Docker recipe for the
multi-GPU run is in `runbook.md`, and the result is recorded in `report.md`.

This is option (b) from the survey: full Mooncake Store via the vLLM
`KVConnector`, chosen as the first prototype for its directness and clean
attribution. It was validated over TCP, first co-located on one GPU and then
across two separate GPUs, as a mechanical proof. Representative performance
numbers still require switching the multi-GPU run off TCP to NVLink or RDMA; the
scope is single-node multi-GPU and multi-machine reuse is deferred.

## Architecture

```
            +-------------------+
            |  mooncake_master  |  metadata + object location (:50051)
            +---------+---------+
                      |
        +-------------+-------------+
        |                           |
+-------v--------+         +--------v-------+
| vLLM instance A|         | vLLM instance B|
| MooncakeStore  | <-----> | MooncakeStore  |   shared KV pool
| kv_role=kv_both|  Store  | kv_role=kv_both|   (embedded segments)
+----------------+         +----------------+
   :8000                       :8001
```

Both instances run `kv_role=kv_both`, so each both writes computed KV to the pool
and reads it back. Either instance can serve any request and still find a prefix
another instance computed. The master coordinates object metadata; the actual KV
bytes move instance to instance via the Transfer Engine (TCP here, RDMA on the
cluster).

## The exact recipe (vLLM 0.22.0)

Connector name (verified in the installed source factory): `MooncakeStoreConnector`.

Per-instance vLLM flag:

```
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
```

Store config, written to the path in `MOONCAKE_CONFIG_PATH`:

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "protocol": "tcp",
  "device_name": "",
  "mode": "embedded",
  "global_segment_size": 536870912,
  "local_buffer_size": 536870912,
  "enable_offload": false
}
```

Master server: `mooncake_master --port 50051 --metrics_port 9003`. `P2PHANDSHAKE`
needs no separate metadata service, which keeps the setup self-contained.

All of this is wrapped by `scripts/serve_master.sh` and `scripts/serve_mooncake.sh`.

## Two findings that gate cross-instance reuse

Both were discovered during the local bring-up and are the kind of thing that
would otherwise burn expensive cluster time.

1. **`PYTHONHASHSEED` must be identical across all instances.** vLLM seeds its
   block-hash chain from `NONE_HASH`, which is `os.urandom(32)` per process
   unless `PYTHONHASHSEED` is set (`vllm/v1/core/kv_cache_utils.py:109`). With
   different seeds, two instances compute different block hashes and can never
   match in the shared store. Before the fix the external hit rate was **0.0%**
   despite the store holding the keys; after setting `PYTHONHASHSEED=0` on both
   instances it was **96.7%**. This is now exported in `serve_mooncake.sh`.

2. **Mooncake's CUDA build must match the host, like vLLM's.** The base
   `mooncake-transfer-engine` wheel is CUDA 12 and fails to import on this CUDA
   13 box (`libcudart.so.12 not found`). The `mooncake-transfer-engine-cuda13`
   wheel matches. On a host virtualenv the master binary also needs the wheel's
   CUDA lib dir on `LD_LIBRARY_PATH` (handled in `serve_master.sh`).

## How to reproduce the local proof

Requires Mooncake installed (the CUDA-matched wheel) and a model small enough to
co-locate two instances on one GPU.

```bash
# Mooncake for CUDA 13 (match the variant to the host; see runbook)
.venv/bin/pip install mooncake-transfer-engine-cuda13

# Master, then two instances (0.5B so both fit one 12 GB GPU), then the test
bash scripts/serve_master.sh &
MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8000 BOOTSTRAP_PORT=8998 GPU_MEM_UTIL=0.38 \
  ENFORCE_EAGER=1 MOONCAKE_CONFIG_PATH=/tmp/mc_a.json bash scripts/serve_mooncake.sh &
MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8001 BOOTSTRAP_PORT=8999 GPU_MEM_UTIL=0.38 \
  ENFORCE_EAGER=1 MOONCAKE_CONFIG_PATH=/tmp/mc_b.json bash scripts/serve_mooncake.sh &

# Wait for :8000 and :8001 /health, then:
.venv/bin/python -m bench.trace --num-sessions 8 --turns-per-session 2 \
  --shared-system-fraction 0.5 --system-words 300 --out data/trace_xinst.jsonl
.venv/bin/python -m bench.run_xinstance --trace data/trace_xinst.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --port-a 8000 --port-b 8001 --settle-s 5

bash scripts/stop_server.sh   # stops instances and master
```

`run_xinstance.py` populates the store from A, then serves the same prompts from
B and reports B's external (cross-instance) hit rate. A run with no connector
shows zero external queries, which is the control.

## Measured result (local, TCP, single GPU)

| Metric | Value |
| --- | --- |
| B external (cross-instance) hit rate | 96.7% |
| B external queries (blocks) | 2,912 |
| B local hit rate | 58.4% (within-instance, phase 2) |
| Model | Qwen2.5-0.5B-Instruct |
| Transport | TCP (not representative; NVLink or RDMA is the multi-GPU tier) |
| KV bytes written by A to store | ~28 MB over 11 puts |

The hit rate is the correctness signal, not a performance claim. Absolute
latencies are meaningless on TCP with a co-located pair; the point proven is that
the wiring is correct and a prefix crosses the instance boundary.

## What is left for the multi-GPU tier

Scope is a single node, one vLLM instance per GPU sharing one Store pool, on the
8x A100 benchmark node (`environment-checklist.md`). The cross-instance run itself
is done: instances on GPU 0 and GPU 1 over TCP reached the 96.7% external hit rate
above, with the exact Docker recipe in `runbook.md`. What remains turns that into
a report-grade performance result:

- Switch the pool off TCP. NVLink (the GPUs are fully NV12-connected) is the
  natural fast path between instances on one node; the alternative is `rdma`
  loopback over a GPU-affined local NIC, which wants `nvidia_peermem` loaded for
  GPUDirect. The cross-node fabric is not involved at this scope.
- Measure representative TTFT and throughput with the cross-instance hit rate, and
  compare against the Stage 2 baseline to quantify the gain.
- Exercise the reliability gates from the rubric: behavior when the master or a
  peer instance goes away (degrade to recompute).

Multi-machine reuse over the cross-node InfiniBand fabric is deferred as future
work.
