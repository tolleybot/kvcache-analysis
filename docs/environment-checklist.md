# Environment Confirmation Checklist

A Stage 0 gate. Before any Stage 3 benchmark on shared hardware counts as
representative, we confirm the network and GPU path can move a KV block from one
GPU's memory across the network into another GPU's memory without the CPU copying
it on the way. That path, GPUDirect RDMA over a real fabric, is what makes
Mooncake's numbers meaningful. Without it the bytes detour through CPU RAM and the
TCP stack, and the result is a lower bound that must be labelled as such.

## Scope

The first target was multi-GPU cross-instance reuse on a single node, one vLLM
instance per GPU sharing one Store pool; that is done (the single-node RDMA run in
`report.md` Section 6.3).

**UPDATE 2026-06-29: the cross-node (multi-machine) phase is now active (GR-1331).**
A second A100 node, `latpoc52` (192.168.147.152), was allocated on the same
InfiniBand fabric as `latpoc51` (192.168.147.151). The cross-node fabric questions
below now apply directly rather than to a deferred phase. The fabric gate has
passed: `ib_write_bw` between the two nodes over `mlx5_0` measured **169 Gb/sec
average** (about 85% of the 200 Gb line rate), so cross-node RDMA is healthy. The
remaining work is to run one vLLM instance per node sharing one Store pool with KV
crossing the fabric, and to measure cross-node reuse latency.

## Hardware strategy

Two tiers, used in order.

1. **Local development tier.** Used now, for integration correctness and a
   mechanical proof that cross-instance hits work. Single node, TCP transport,
   small or quantized models. Numbers from this tier are not representative and
   are always labelled.
2. **Benchmark tier.** A multi-GPU node with NVLink between GPUs. Used once the
   prototype works locally, to produce report-grade numbers. The node now in use
   is recorded below. For the deferred multi-machine phase this is a fabric-local
   multi-node allocation, and only then does the cross-node section of this
   checklist apply.

### Local development tier, recorded

Captured on the workstation in use:

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070, 12 GB |
| Compute capability | 12.0 (Blackwell) |
| Driver | 595.58.03 |
| CUDA (nvcc) | 12.6 |
| Topology | single GPU, single NUMA node |
| RDMA NIC | none |
| GPUDirect module | none |

Implications:

- TCP-only and single-node. Good for correctness and the integration shape, not
  for performance claims.
- 12 GB VRAM constrains model choice. Use a small or quantized model so KV
  pressure and eviction can still be exercised.
- CUDA 12.6 maps to the default `mooncake-transfer-engine` package, the CUDA < 13
  build, not the `-cuda13` or `-non-cuda` variants.

### Benchmark tier, recorded

Captured on host `latpoc51`, the multi-GPU node now in use:

| Property | Value |
| --- | --- |
| GPU | 8x NVIDIA A100-SXM4-80GB |
| Compute capability | 8.0 (Ampere) |
| GPU interconnect | NVLink, fully connected (NV12 between every GPU pair) |
| Driver | 570.86.15 |
| CUDA (driver max) | 12.8 |
| CPU / RAM | 2x AMD EPYC 7542 (128 threads), 2 TB RAM, 4 NUMA nodes |
| RDMA NIC | ConnectX-6, four IB ports Active at 200 Gb (mlx5_0/2/4/10), MLNX_OFED 25.01 |
| GPU-to-NIC affinity | PXB-local NIC pair per GPU pair (GPU0/1 to mlx5_0/1, GPU2/3 to mlx5_2/3, GPU4/5 to mlx5_4/5, GPU6/7 to mlx5_10/11) |
| GPUDirect module | `nvidia_peermem` not loaded (only `nvidia_fs`); see note below |
| Other | Kubernetes node; Docker 28.4 with NVIDIA Container Toolkit 1.17.7; perftest present |

Implications:

- This single node satisfies the multi-GPU scope on its own. Run one vLLM
  instance per A100 sharing one Store pool, with a real model (80 GB per GPU
  allows far more than the 3B baseline).
- `nvidia-smi` reports max CUDA 12.8, but the stock vLLM `v0.22.0` image ships a
  CUDA 13 (`cu130`) torch build and runs here through CUDA forward compatibility
  on the A100, verified by a kernel launch. So the default image needs no tag
  override, and the base `mooncake-transfer-engine` wheel (not `-cuda13`) works
  alongside it.
- `nvidia_peermem` is not loaded. It matters for the single-node performance path,
  because the Mooncake Store accepts only `tcp` and `rdma` (it rejects
  `nvlink_intra` at init), so beating TCP means RDMA, and RDMA wants GPUDirect (NIC
  DMA straight to and from GPU memory) rather than a CPU bounce. Load it with
  `modprobe nvidia_peermem` (needs privilege) before measuring RDMA on-node.
- NVLink P2P is validated on this box at about 270 GB/s GPU-to-GPU
  (`scripts/check_nvlink_p2p.py`). The `nvlink_intra` rejection noted above is a
  Mooncake Store build limitation, not a hardware one; the NVLink path itself is
  healthy.
- The dedicated InfiniBand fabric here is what a future multi-machine phase would
  use; it is not required for the current single-node multi-GPU work.

## Questions for the deferred multi-machine phase, by owner

These confirm a cross-node RDMA fabric and apply only to the deferred
multi-machine phase, not to the current single-node multi-GPU work.

### HPC, fabric, and platform team

1. Is there a dedicated RDMA fabric, or is inter-node traffic Ethernet and TCP?
   If RDMA, is it InfiniBand or RoCEv2?
   Good answer: InfiniBand or RoCEv2. TCP-only means proof-of-concept only.
2. What NICs are in the GPU nodes, and at what line rate, for example
   ConnectX-6 or 7 at 200 or 400 Gb? Sets the transfer bandwidth ceiling and
   decides whether GPUDirect is supported at all.
3. Is GPUDirect RDMA supported and enabled? Is the `nvidia-peermem` or dma-buf
   path available so the NIC can DMA directly to and from GPU memory? This is the
   zero-copy GPU-to-GPU-across-network path that carries the performance win.
4. What is the GPU-to-NIC topology per node? Is there a NIC with a direct
   PCIe and NUMA path to each GPU? Poor affinity quietly halves bandwidth.
5. Which RDMA software stack and version is installed, MLNX_OFED, DOCA, or inbox
   rdma-core, and what CUDA driver version? Mooncake links against RDMA verbs
   libraries and expects CUDA 12.1 or newer.

### ML platform, scheduler, and k8s team

6. Can I get at least two GPU nodes on the same fabric or partition at the same
   time, and are they guaranteed to share the same leaf or switch? Cross-instance
   hits need two nodes that can RDMA to each other concurrently.
7. Is the RDMA device exposed inside containers or pods, via SR-IOV, an RDMA
   device plugin, or host networking? A perfect fabric can still be invisible from
   inside a container. This is the most common silent blocker.
8. Can I run containers that load kernel modules or need elevated privileges, or
   is that locked down? Some GPUDirect setups need specific modules or
   capabilities.
9. How are multi-node jobs allocated, Slurm or k8s, and how do I request
   fabric-local placement?

### Sanity checking

10. Are the standard `perftest` tools, `ib_write_bw` and `ib_send_bw`, available?
    Measure raw fabric bandwidth and latency before touching Mooncake, so a
    disappointing result can be attributed to the fabric or to the software.

## Commands to run on a cluster node

Run these on a benchmark node, not the local workstation, to answer most of the
hardware questions without waiting on anyone. They are read-only.

```bash
# RDMA device present and link up? InfiniBand vs Ethernet/RoCE shows here
ibstat
ibv_devices
ibv_devinfo            # link rate and port state per device

# NIC hardware
lspci | grep -i mellanox

# GPU, NIC, and NUMA affinity matrix; look for PIX or PXB between a GPU and a NIC
nvidia-smi topo -m

# GPUDirect RDMA kernel module loaded?
lsmod | grep -iE "peermem|nv_peer"

# Does the transport layer see RDMA devices?
ucx_info -d | grep -i rdma

# Raw fabric bandwidth and latency, server on one node, client on the other
ib_write_bw        # run with no args on node A, then 'ib_write_bw <node-A-ip>' on node B
```

## Reading the answers

- **Green light.** InfiniBand or RoCEv2, ConnectX-6 or 7, `nvidia-peermem`
  loaded, RDMA visible inside the container, and two fabric-local nodes
  obtainable. Stage 3 produces credible numbers.
- **Yellow.** RDMA exists on the metal but is not exposed to containers, or only
  one node is available at a time. Fixable, but it is a conversation with the
  platform team.
- **Red.** TCP or Ethernet only. Correctness and cross-instance hits can still be
  proven mechanically, but the performance numbers carry an asterisk and the
  report frames them as a lower bound.
