# Environment Confirmation Checklist

A Stage 0 gate. Before any Stage 3 benchmark on shared hardware counts as
representative, we confirm the network and GPU path can move a KV block from one
GPU's memory across the network into another GPU's memory without the CPU copying
it on the way. That path, GPUDirect RDMA over a real fabric, is what makes
Mooncake's numbers meaningful. Without it the bytes detour through CPU RAM and the
TCP stack, and the result is a lower bound that must be labelled as such.

## Hardware strategy

Two tiers, used in order.

1. **Local development tier.** Used now, for integration correctness and a
   mechanical proof that cross-instance hits work. Single node, TCP transport,
   small or quantized models. Numbers from this tier are not representative and
   are always labelled.
2. **Benchmark tier.** GResearch multi-GPU clusters (V100, H200, and similar).
   Used only once the prototype works locally. Produces report-grade numbers,
   and only after this checklist passes on it.

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

## Questions for the benchmark cluster, by owner

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
