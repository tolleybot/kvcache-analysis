# Cross-Node RDMA Test: Mooncake Store Across Two Machines

This document describes how to take the single-node Mooncake Store result from
`stage3.md` (one host, one Store pool, two vLLM instances on two GPUs of the same
box) and extend it to a real cross-machine measurement, with the two vLLM
instances on separate hosts sharing one Store pool over an InfiniBand fabric.

The single-node tier proved that a prefix written by instance A is reused by
instance B through the Store. The cross-node tier is the result that actually
matters for the distributed-pool thesis: a prefix paid for on one machine is
reused on another machine, over RDMA, at a representative cost.

Terminology used below:

- **Node A** is the host running the Store master and vLLM instance A.
- **Node B** is the host running vLLM instance B and joining Node A's Store
  pool over the IB fabric.
- Substitute the IPs and RNIC names of your two nodes wherever this document uses
  the placeholders `10.0.0.1` (Node A), `10.0.0.2` (Node B), and `mlx5_0` (RNIC).

## Prerequisites

The pair of hosts must satisfy all of:

1. The same GPU family (mixed A100 and H200 is fine for a smoke test, but
   performance results should pin one family). At least one GPU per host that is
   PCIe-affined (PXB or NODE) to an active RDMA NIC on the same node; check with
   `nvidia-smi topo -m`.
2. The same InfiniBand fabric, with both hosts on the same subnet manager. Each
   host should see the other's GIDs/LIDs through `ibstat` and `ibv_devinfo`, and
   `ibping` or `ib_write_bw` must succeed both directions.
3. The same Linux kernel major and the same OFED/`rdma-core` major. Mixing
   `rdma-core` and Mellanox OFED across the pair is a common cause of silent
   QP-state failures; match the stack.
4. The `nvidia_peermem` module loaded on both hosts. Without it, RDMA still works
   but bounces through CPU memory, so any "GPUDirect" claim is invalid. Confirm
   with `lsmod | grep nvidia_peermem`. Load with `sudo modprobe nvidia_peermem`.
5. CUDA driver and toolkit versions compatible with the project image. The
   project image's CUDA matrix is in `runbook.md`; the tag override mechanism
   there applies per-host if the driver is older on one machine.
6. The benchmark model's HuggingFace cache populated on both hosts (or mounted
   from a shared filesystem), so neither instance blocks on a first-boot
   download.

## Pre-flight: prove the fabric before involving vLLM

Run these from Node A unless noted. Failing any of these means the cross-node
result will be uninterpretable; do not skip them.

```bash
# 1. IB link is up on the RNIC you intend to use (State: Active, Rate: 200 Gb/s
#    or whatever the fabric provides). Repeat on Node B.
ibstat mlx5_0
ibv_devinfo -d mlx5_0

# 2. Two-way reachability over IB. Run the server on Node B first, then this
#    client from Node A. Bandwidth should land near the line rate of the link.
#    On Node B:
ib_write_bw -d mlx5_0 -D 5
#    On Node A:
ib_write_bw -d mlx5_0 -D 5 10.0.0.2

# 3. TCP reachability between the two nodes on the ports the Store will use
#    (50051 for the master, plus 8000 and 8001 for the vLLM HTTP endpoints).
#    On Node A:
nc -l 50051 &
#    On Node B:
nc -vz 10.0.0.1 50051

# 4. NVIDIA peer memory loaded (both nodes).
lsmod | grep nvidia_peermem
```

If any of these fail, fix the fabric first. The Mooncake Store does not surface
QP-level failures clearly; you will see a low hit rate and have to bisect blind.

## Topology

The simplest split that matches the existing Compose file is:

- **Node A** runs the Store master plus vLLM instance A on its GPU 0.
- **Node B** runs vLLM instance B on its GPU 0, pointing at the master on Node A.

This is exactly the single-node topology with instance B relocated to a separate
host. No new container roles are introduced. The cross-instance benchmark
(`bench.run_xinstance`) runs from either host (or a third) and talks to the two
HTTP endpoints over IB or TCP, whichever is faster between the chosen hosts. The
RDMA path that matters is the Store traffic between the two instances, not the
benchmark's HTTP calls.

## Image and config: what must match across both nodes

These are the items that, when out of sync between the two hosts, produce a
zero-percent cross-instance hit rate with no error. Verify each before bringing
the topology up.

1. **Same image, same tag.** Build `kvcache:mooncake` on one host and either
   `docker save | docker load` it on the other, or rebuild from the same commit
   on both. Different vLLM or Mooncake versions across the pair will not
   interoperate at the block-hash level.
2. **Identical `PYTHONHASHSEED`.** vLLM seeds its block hashes per process; if
   the seed differs, the two instances compute different hashes for the same KV
   block and never match in the Store. `scripts/serve_mooncake.sh` fixes this
   for both instances, but if you customize the launcher, keep the seed pinned.
3. **Identical model and tokenizer.** Same `MODEL` env var, same revision. A
   tokenizer revision mismatch shifts token IDs and therefore block hashes.
4. **Identical `MAX_MODEL_LEN` and prefix-caching settings.** Differences here
   change the block layout.
5. **Mooncake protocol = `rdma` on both instances.** Mixing transports across
   instances in one pool is not supported.

## Compose: cross-node overlay

The existing `docker/compose.mooncake.yml` plus `docker/compose.rdma.yml`
overlay assumes both instances run on one host. For the cross-node test, split
the topology across two Compose invocations, one per host, each bringing up
only its part of the stack and using host networking so the IB devices and the
master's port are reachable across machines.

Sketch (do not commit verbatim; tune to the chosen RNIC, IPs, and pool size):

```yaml
# docker/compose.node-a.yml: master plus instance A on Node A.
# Reuses the service definitions from compose.mooncake.yml and
# compose.rdma.yml; this file only sets cross-node-specific values.
services:
  master:
    network_mode: host
  instance-a:
    network_mode: host
    environment:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_DEVICE: mlx5_0
      MOONCAKE_REQUESTER_LOCAL_HOSTNAME: 10.0.0.1
      MASTER_ADDR: 10.0.0.1:50051
      SEGMENT_SIZE: "8589934592"
      BUFFER_SIZE: "2147483648"
```

```yaml
# docker/compose.node-b.yml: instance B on Node B.
services:
  instance-b:
    network_mode: host
    environment:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_DEVICE: mlx5_0
      MOONCAKE_REQUESTER_LOCAL_HOSTNAME: 10.0.0.2
      MASTER_ADDR: 10.0.0.1:50051
      SEGMENT_SIZE: "8589934592"
      BUFFER_SIZE: "2147483648"
```

Key points the overlay encodes:

- `network_mode: host`. The IB device passthrough plus the cross-node master
  contact both want host networking. Docker's default bridge will not carry IB,
  and per-port publishing breaks the master/instance handshake across hosts.
- `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` set to the IB-routable IP, not the
  container hostname or `localhost`. The Store advertises this value to peers,
  so it must be reachable from the other host.
- `MASTER_ADDR` set to Node A's IP on both hosts. Both instances must contact
  the same master.
- `SEGMENT_SIZE` and `BUFFER_SIZE` raised to a useful value (8 GiB segment, 2
  GiB local buffer is a reasonable starting point; tune to the largest expected
  KV per prompt). The 1 GiB default is too small for long prefixes and shows up
  as a low hit rate.

## Run order

```bash
# On Node A: bring up the master plus instance A.
IMAGE=kvcache:mooncake MODEL=Qwen/Qwen2.5-3B-Instruct \
  docker compose \
    -f docker/compose.mooncake.yml \
    -f docker/compose.rdma.yml \
    -f docker/compose.node-a.yml \
    up -d master instance-a

# Wait for the master and instance A to be healthy.
until curl -sf 10.0.0.1:8000/health; do sleep 5; done

# On Node B: bring up instance B pointed at the master on Node A.
IMAGE=kvcache:mooncake MODEL=Qwen/Qwen2.5-3B-Instruct \
  docker compose \
    -f docker/compose.mooncake.yml \
    -f docker/compose.rdma.yml \
    -f docker/compose.node-b.yml \
    up -d instance-b

# Wait for instance B to be healthy.
until curl -sf 10.0.0.2:8001/health; do sleep 5; done
```

If either instance fails health, capture its container logs before retrying.
The most common first-run failures are: wrong RNIC name in `MOONCAKE_DEVICE`,
wrong advertised hostname in `MOONCAKE_REQUESTER_LOCAL_HOSTNAME`, or `IPC_LOCK`
and `memlock` ulimits not propagating through host networking. The RDMA
overlay in `compose.rdma.yml` covers the last item.

## Measure

Run the cross-instance harness from either node (or a third), pointing port A
at Node A and port B at Node B. The benchmark itself does not need IB
connectivity; it speaks HTTP.

```bash
docker run --rm --network host \
  -v "$(pwd)/bench/results:/app/bench/results" \
  kvcache:mooncake -lc '
    python3 -m bench.trace \
      --num-sessions 8 --turns-per-session 2 \
      --shared-system-fraction 0.5 --system-words 300 \
      --out /tmp/trace_xinst.jsonl
    python3 -m bench.run_xinstance \
      --trace /tmp/trace_xinst.jsonl \
      --model Qwen/Qwen2.5-3B-Instruct \
      --port-a 8000 --host-a 10.0.0.1 \
      --port-b 8001 --host-b 10.0.0.2 \
      --settle-s 5 \
      --out /app/bench/results/xinstance_crossnode_rdma.json'
```

Adjust `bench.run_xinstance` invocation to whatever flags the current harness
takes for remote hosts; the single-node version assumes `localhost` for both
ports.

## What good looks like

A pass on the cross-node tier needs three results, in order of importance:

1. **Cross-instance hit rate is similar to the single-node tier.** The
   single-node TCP run reached 96.7% external hits (`stage3.md`); the cross-node
   RDMA run should land in the same neighborhood. A large drop indicates a
   block-hash mismatch (revisit the "must match" list above), not a transport
   issue.
2. **TTFT for instance B's reused prefix beats recomputing it locally**, at the
   prefix lengths that justify the pool (a few thousand tokens and up). Below
   that, recompute wins and the pool is not the right tool.
3. **Throughput at representative concurrency** matches or beats the Stage 2
   baseline (`baseline.md`) at the same hit rate. This is the result that lands
   in the report.

Record the RNIC, the measured `ib_write_bw` between the two hosts, the pool
segment size, the model and `MAX_MODEL_LEN`, and the trace parameters alongside
each run in `bench/results/`. Without those, the number is not reproducible.

## Tear down

```bash
# On each node, in either order.
docker compose -f docker/compose.mooncake.yml \
               -f docker/compose.rdma.yml \
               -f docker/compose.node-{a,b}.yml down
```

The Store pool is in-memory on each instance, so down means the pool is gone.
If a future run wants the pool persisted, that is a separate Store
configuration step and is not in scope here.

## Failure modes seen so far on the single-node tier, still relevant

These reproduce across nodes and should be checked first when a result looks
wrong, before suspecting the fabric:

- **Zero cross-instance hits with no error.** Almost always one of:
  `PYTHONHASHSEED` mismatch, different model revision, or different
  `MAX_MODEL_LEN` between instances. See `stage3.md`.
- **Hit rate degrades as the prompt grows.** Pool segment too small; raise
  `SEGMENT_SIZE` and `BUFFER_SIZE`. See the TCP pool-size note in `runbook.md`.
- **Transfers fail on large prefixes over TCP, around 3,300 tokens.** A known
  TCP limitation, fixed by switching to RDMA. Mentioned for completeness; on
  this tier the protocol is already RDMA.

## What this tier does not cover

- **Failure handling.** Master loss, peer loss, and IB link flap behaviors are
  separate gates from the rubric and should be measured after the happy-path
  result lands.
- **More than two instances.** The Compose split above scales to N hosts by
  repeating the Node B pattern, but the benchmark harness presently compares two
  endpoints. A larger pool needs a harness change.
- **Mixed GPU families.** Pin one family for the headline result.
