# Distributed KV Caching for vLLM: Status Summary

**TL;DR:** We investigated distributed KV caching options for LLM serving,
recommended Mooncake Store, and proved cross-instance cache sharing works across
two GPUs in a single machine (two A100s, one vLLM instance each). The key learning
is that the mechanism is correct but the transport matters enormously. Our
single-node setup was using TCP, which is the wrong choice and makes caching a net
loss. We confirmed production uses RDMA, not TCP, and we tested the NVLink path and
found the Mooncake Store does not accept it (only TCP and RDMA). So the next
decision is whether to redo the single-node run on RDMA, or to reconsider whether a
networked pool is the right tool for a single node at all.

## What we set out to do

Evaluate distributed KV caching solutions for vLLM serving, pick one, and prove it
with measured numbers. The motivating workload is agentic and multi-turn serving,
where long shared prefixes (system prompts, prior turns, tool output) get
recomputed wastefully. The primary deliverable is a written investigation and
recommendation.

## What we did

1. **Surveyed the field** and framed it cleanly. There is a transport layer (moves
   KV bytes) and a cache-management layer (decides what to store and reuse). The
   real choice is the management layer. Recommendation: Mooncake Store via vLLM's
   connector first, then LMCache for the operability story. This comparison is
   sourced from public documentation and is clearly marked as not our own
   benchmark.
2. **Measured the baseline** (native vLLM). Cache hit rate tracks prefix sharing,
   and when the per-instance cache is too small the hit rate collapses and time to
   first token roughly doubles. That gap is the reason to want a shared pool.
3. **Stood up the prototype on the 8x A100 box.** Two vLLM instances, one per GPU,
   sharing one Mooncake Store pool, all on a single machine.

## Key findings

- **Cross-instance reuse works.** Instance B reused about 98% of the KV that
  instance A computed, with zero transfer failures. The mechanism is correct.
- **Over TCP it is a net loss.** Side by side, on the same hardware:

  | | Recompute (cold) | Reuse from pool (TCP) |
  | --- | --- | --- |
  | Time to first token | about 39 ms | about 1,843 ms |

  Pulling cached KV over TCP was roughly 47 times slower than recomputing it,
  because the TCP transport ran at only about 16 MB/s. Large prefixes failed
  outright through TCP port exhaustion.
- **Why.** We were moving data between two GPUs over TCP loopback on a machine
  where those GPUs are directly connected by NVLink (hundreds of GB/s) that sat
  idle. That is the wrong transport for a single box.

## What production actually uses (verified)

We checked this against Mooncake's own config and docs and the vLLM Mooncake Store
blog, rather than guessing:

- Production uses **GPUDirect RDMA**, even in the single-node baseline (a 12-GPU
  GB200 run), and scales across nodes over RDMA with multi-NIC pooling. TCP is
  documented only as the universal, no-special-hardware fallback, never the
  performance transport.
- We tried switching our single-node run to NVLink (`nvlink_intra`). The Mooncake
  Store **rejects it at startup** (`unsupported_protocol`): the Store path supports
  only `tcp` and `rdma`. So the realistic single-node fix is **RDMA**, not NVLink.
  (A host shared-memory transport exists but only as a non-default build absent
  from the standard package, so it is not a quick win either.)

## The open question (the decision for the meeting)

Two things follow. First, our TCP result is unrepresentative by construction, and
the fix is to switch the transport to RDMA (we confirmed the Store does not accept
NVLink). On this box that means InfiniBand device passthrough into the containers
and, for true zero-copy GPU transfers, loading `nvidia_peermem` for GPUDirect.
Second, and more strategically: distributed KV pooling earns its keep across nodes
over RDMA, so on a single node the better-fit production pattern may be the engine's
native cache plus CPU or NVMe offload rather than a networked Store. The decision is
whether to invest in single-node RDMA numbers or to revisit whether single-node is
the right frame for a distributed pool at all.

## Where the work lives

Everything is committed and pushed. The consolidated report is `docs/report.md`.
The prototype, the benchmark harness, and a reproducible Docker recipe are in the
repository. Honest negatives, such as the TCP net loss, are written up rather than
hidden.
