# Distributed KV Caching for vLLM: Status Summary

**TL;DR:** We investigated distributed KV caching options for LLM serving,
recommended Mooncake Store, and proved cross-instance cache sharing works across
two GPUs in a single machine (two A100s, one vLLM instance each), with the second
instance reusing about 98% of the cache the first computed. The key learning is
that transport is everything. Over TCP the reuse was about 47 times slower than
recomputing, a net loss; over RDMA with GPUDirect the pooled fetch is faster than
recompute, so the cache pays off. Both the mechanism and the win are now
demonstrated. What remains is scaling it (bigger models, real concurrency) and a
strategic call on whether a networked pool fits a single node or is really a
multi-node tool.

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
- **Transport decides everything.** Same trace, same hardware, time to first token:

  | | Recompute (cold) | Reuse from pool (TCP) | Reuse from pool (RDMA) |
  | --- | --- | --- | --- |
  | TTFT (p50) | about 27-39 ms | about 1,843 ms | about 19.5 ms |

  Over TCP the cache was ~47x slower than recomputing (the transport ran at only
  ~16 MB/s, and large prefixes failed outright). Over RDMA with GPUDirect the same
  fetch dropped to about 2.9 ms (from ~3.3 s), so reuse became faster than
  recompute. The cache is a net loss over TCP and a net win over RDMA.

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

The transport question is now settled: RDMA works and the cache pays off on one
node (we ran it; NVLink is not an option because the Store rejects it). Two things
remain. First, scale the win, the margin is small on a 3B model with a short
prefix, and it grows with model size, context length, real concurrency, and cache
pressure; that is the report-grade number still to produce. Second, and more
strategically: distributed KV pooling earns its keep most across nodes over RDMA,
so on a single node the better-fit production pattern may be the engine's native
cache plus CPU or NVMe offload rather than a networked Store. The decision is how
far to push single-node RDMA numbers versus framing the pool as a multi-node tool.

## Where the work lives

Everything is committed and pushed. The consolidated report is `docs/report.md`.
The prototype, the benchmark harness, and a reproducible Docker recipe are in the
repository. Honest negatives, such as the TCP net loss, are written up rather than
hidden.
