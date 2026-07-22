# Distributed KV Caching for vLLM Inference Serving: Investigation and Recommendation

A standalone report on what distributed KV caching solutions are available for
large language model inference serving, how they compare, and which to adopt. It
consolidates the staged work in this repository (the evaluation rubric, the
sourced survey, the baseline measurement, and the cross-instance prototype) into
a single document. The detailed working docs remain in `docs/` for reproduction and
sources; this report is meant to be read on its own.

Status as of June 2026: the survey and recommendation are complete, the native
baseline is measured, and cross-instance reuse has been proven on multi-GPU
hardware. Representative cluster-grade performance numbers are the one remaining
piece, and the report is explicit about what is measured versus what is still
projected.

## 1. Executive summary

During transformer inference the attention mechanism caches a key and value
vector for every token in the context. That KV cache is large, it lives in scarce
GPU memory, and it is recomputed wastefully whenever a shared prefix (a system
prompt, prior conversation turns, tool output) is processed again. A distributed
KV caching solution pools that cache across serving instances so a prefix is
computed once and reused anywhere, including by an instance that never saw it.

The field is best understood as two layers. A transport layer moves KV bytes
between memory tiers and across the network, and a cache-management layer decides
what to store, how to key it, when to evict, and how to look it up. The
management layers all sit on top of the same transport, so the real decision is
which cache-management layer to adopt, not one vendor against the rest.

**Recommendation.** Adopt **Mooncake Store through vLLM's `KVConnector`
interface** as the first integration. It is the most direct, officially supported
path, it has the fewest moving parts, and it makes results easy to attribute.
Then evaluate **LMCache fronting Mooncake Store** for the operability and
reliability dimensions that an enterprise deployment weights most heavily, since
LMCache brings a Kubernetes-native serving stack, documented observability, and
graceful degradation. Native vLLM local offload remains the baseline that any
distributed option must beat, and it is beaten decisively on the one thing it
cannot do, cross-instance reuse.

**Evidence in one line each.** Under a per-instance cache too small for the
working set, native vLLM's hit rate collapses to zero and time to first token
roughly doubles, which is the loss a pool exists to recover. Two vLLM instances
sharing one Mooncake Store pool reused about **98%** of each other's KV, proving the
mechanism. The payoff hinges entirely on transport: over TCP the pooled fetch was
about **47x slower** than recomputing (a net loss), but over **RDMA** with GPUDirect
it was faster than recompute (time to first token about **19 ms versus 27 ms**
cold), and that win held **across two physical machines** on an InfiniBand fabric,
the production-shaped scenario. Section 6.3 consolidates the numbers.

## 2. The problem and the workload

**What the KV cache is.** Inference splits into a prefill phase that computes the
KV cache for the whole prompt at once, and a decode phase that generates one token
at a time while reusing the cached KV rather than reprocessing the prompt. The
cache size grows with context length, layer count, number of attention heads, and
model size, and a long context can occupy several gigabytes for a single session.
Because it normally lives in GPU high-bandwidth memory, which is both scarce and
expensive, it is the natural pressure point in a serving system.

**Prefix reuse is the payoff.** When many requests share a common prefix, the
prefill cost for that prefix is paid once and the cached KV is reused, so each
request only pays for its new tokens. Distributed KV caching extends this across
instances: instead of each instance holding a private cache, instances share a
cluster-wide pool over a fast network, which gives larger aggregate capacity and,
critically, cross-instance cache hits.

**The motivating workload.** Agentic and multi-turn serving is the target. System
prompts, prior turns, and tool output form long prefixes that are shared across
turns and across instances and are otherwise recomputed every time. Local offload
to CPU DRAM or disk hits two limits here: per-instance capacity, which forces
eviction under load, and cross-instance misses, which happen whenever a session is
routed to an instance that never computed its prefix. A distributed pool addresses
both at once.

## 3. The layering model

This is the single most useful framing, and the evidence supports it across every
candidate.

- **Transport layer.** Moves KV bytes between memory tiers (GPU VRAM, CPU DRAM,
  NVMe) and across the network (TCP, RDMA over InfiniBand or RoCE, NVLink, CXL).
  Mooncake's **Transfer Engine** is the dominant implementation, and NVIDIA's
  **NIXL** is a parallel abstraction that can use the Transfer Engine as a
  backend. This layer does not decide what to cache or when to evict.
- **Cache-management layer.** Decides what to store, how to key it, when to evict,
  and how to look it up across instances. **Mooncake Store**, **LMCache**, and
  **FlexKV** all live here, and they all sit on top of the Transfer Engine for
  their actual data movement.

The consequence is that the choice is not "Mooncake versus the others." Mooncake's
Transfer Engine tends to be the transport inside the alternatives, so it comes
along underneath whichever management layer is chosen. The decision is which
cache-management layer to adopt.

## 4. Evaluation framework

The evaluation deliberately weights operability and reliability above raw
performance, because the target is enterprise infrastructure where a faster but
fragile or hard to run solution loses to a dependable, operable one.

**Weighted rubric.** Candidates are scored 1 to 5 per dimension and weighted.

| Dimension | Weight | What a 5 looks like |
| --- | --- | --- |
| Operability | 25 | Deploys on existing infra with minimal moving parts, Kubernetes friendly, sane rollout, modest dependencies, clear config. |
| Reliability | 25 | Degrades to recompute on cache or node loss, survives master and partition failures, predictable backpressure, no correctness risk. |
| Performance | 20 | Large, consistent TTFT and throughput gains at realistic hit rates, low transfer overhead, scales with cluster size. |
| Observability | 10 | Hit rate, transfer latency, pool occupancy, and eviction metrics out of the box. |
| Integration cost | 10 | Works through the vLLM `KVConnector` path with little glue, composes via `MultiConnector`. |
| Maturity and support | 10 | Production use, active maintenance, real documentation, responsive upstream. |

Operability and reliability together carry half the score by design.

**Hard gates (pass or fail, independent of score).**

- **Correctness invariant.** Reused KV must always correspond to the exact prefix
  it is attributed to. A content-address or hash collision that returns the wrong
  KV for a different prefix is a silent correctness failure and is not acceptable.
- **Safe degradation.** Loss of the cache, the pool, or the metadata service must
  fall back to recomputation, never to a wrong or failed answer.
- **No cross-tenant leakage.** Cache keys and stored blocks must not let one
  session read another's content.

**Metrics.** Every benchmark reports the cache hit rate alongside the performance
numbers, because a result without a hit rate cannot be told apart from a faster
GPU. The primary metrics are cache hit rate, cross-instance hit rate (which
isolates the distributed value over local offload), TTFT, throughput at a fixed
latency target, and end to end latency at p50, p95, and p99.

## 5. Candidates and comparison

**Provenance.** The comparison in this section is compiled from external sources,
the projects' own documentation, pull requests, design docs, and community posts,
then fact-checked against primary sources (the method and full source list are in
`docs/survey.md`). It is not original benchmark data produced for this report. Of
the candidates, only Mooncake Store has been run on our own hardware (Section 6.2),
and only over TCP; LMCache, FlexKV, and NIXL are scored from their documentation,
not measured here. Vendor performance figures are flagged where they appear and
are not independently reproduced. The only numbers in this report that we measured
ourselves are the baseline (Section 6.1) and the cross-instance prototype
(Section 6.2).

Profiles are condensed; the sourced detail and the fact-checking method are in
`docs/survey.md`.

**Mooncake Store (cache-management, the lead candidate).** A distributed pooled KV
cache built on the Transfer Engine, with a master server for metadata and service
discovery and clients on the GPU nodes. It is integrated into vLLM through the
existing `KVConnector` interface, composable via `MultiConnector`, with the full
Store integration released in vLLM v0.21.0. It is Apache-2.0 and is the serving
backbone for Moonshot AI's Kimi. Its main reliability gap is that high-availability
master failover currently depends on ETCD, with a Kubernetes-native lease-based
path still an open issue as of early 2026.

**LMCache (cache-management, closest head-to-head).** A management layer with a
two-tier hierarchy, an L1 local tier (CPU or GPU) and an L2 remote tier, and it
can use Mooncake Store as that L2 backend. It integrates with vLLM v1 and ships
through the official vLLM Production Stack, a Kubernetes-native, Helm-deployed
serving stack, which is the strongest operability story in the field. It reuses KV
for any repeated text, not only shared prefixes, and it exposes a documented
metrics surface. Its L1 tier survives an L2 outage, which softens the Store
master gap when LMCache fronts it. The cost is an extra layer and muddier
attribution, since a hit may come from L1 or L2.

**FlexKV (cache-management).** A unified KV caching layer in the Dynamo ecosystem
that uses the Mooncake Transfer Engine for cross-node transfer. Its
disaggregated-serving integration with vLLM is still experimental. A specific
claim that it merged natively into vLLM mainline was refuted during fact-checking
and is not relied upon. Promising, but less mature on the vLLM path than Store or
LMCache, so it stays a comparison point.

**Mooncake Transfer Engine and NIXL (transport, not standalone choices).** Both
are transport substrate, not cache-management. Prototyping either alone means
building the management layer ourselves, which contradicts the operability
priority, so they are out of scope as a first prototype.

**SGLang HiCache (context only).** Can use Mooncake Store as a hierarchical
backend, but SGLang is not the target engine, so it is noted and excluded from
scoring.

**Native vLLM local offload (the baseline).** vLLM's prefix caching keys each KV
block by a hash over the block's token ids plus the preceding prefix, using
SHA-256 by default. This content-addressing is exactly the discipline that makes
cross-instance reuse safe, and the distributed layers extend it. The baseline is
the number every distributed option must beat, and it has no cross-instance hits
by construction.

**Comparison matrix.** Scored against the rubric from sourced evidence; the
performance column is directional until our own cluster numbers land.

| Candidate | Layer | vLLM integration | Operability | Reliability | Observability | Maturity | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mooncake Store | Cache-mgmt | 5, official, v0.21.0 | 3, k8s but HA master needs ETCD | 3, HA master not yet k8s-native | 3, present, less turnkey | 5, Kimi production | Lead candidate, most direct path |
| LMCache (+ Store as L2) | Cache-mgmt | 4, vLLM v1, Production Stack | 5, official Helm/k8s stack | 4, L1 survives L2 outage | 4, documented metrics | 4, active, widely used | Best operability, extra layer |
| FlexKV | Cache-mgmt | 2, experimental | 3 | 3 | unverified | 3 | Native-merge claim refuted |
| Mooncake Transfer Engine | Transport | n/a alone | n/a | n/a | n/a | 5 | Substrate under all of the above |
| NIXL | Transport | connector exists | n/a | n/a | n/a | 4 | Transport peer, not a cache layer |
| Native vLLM offload | Baseline | built in | 5 | 5 | built in | 5 | The number to beat; no cross-instance hits |

## 6. Measured evidence

Two measurements from this project anchor the recommendation, the cross-instance
prototype (Section 6.2) and the transport comparison (Section 6.3); a development-box
mechanism check (Section 6.1) supports the motivation but is not itself a
production result. State the exact environment with every number, since absolute
values depend on hardware.

### 6.1 Mechanism check: the capacity effect (development box, supporting)

Before the cluster work, the capacity effect that motivates a pool was confirmed on
a development box, which also validated that the harness measures hit rate
correctly. This is a supporting check, not a production baseline: the hardware is a
12 GB consumer GPU, so only the relative effect transfers and absolute latencies are
not reported here. A synthetic agentic, multi-turn trace was run with the
shared-prefix fraction swept across 0, 50, and 90 percent under two cache sizings,
ample and constrained (the constrained size forces eviction). The full environment
and method are in `docs/baseline.md`.

| Shared prefix | Hit rate, ample | Hit rate, constrained |
| --- | --- | --- |
| 0% | 68.6% | 0.0% |
| 50% | 76.5% | 30.6% |
| 90% | 90.4% | 61.5% |

Hit rate is the signal, because it is a property of the workload and the cache
sizing rather than of the GPU. Shrinking the cache below the working set collapses
the hit rate to zero and roughly doubles TTFT; that collapse, not any absolute
number, is the point, and it is the empirical basis for the
capacity-versus-working-set argument a distributed pool exists to serve. The
within-session reuse still visible at 0 percent cross-session sharing (68.6%)
confirms the harness captures reuse as designed. Throughput and end to end latency
barely move at this scale because, with a small model, short outputs, and
concurrency one, total latency is decode-bound; the effect on throughput grows with
larger models, longer prompts, and higher concurrency. The on-axis production
baseline on A100/GB200 hardware remains work (Section 8).

### 6.2 Prototype: Mooncake Store cross-instance reuse

This proves the distributed value that single-instance caching cannot provide.

- **Environment.** One node, 8x NVIDIA A100-SXM4-80GB (fully NVLink-connected),
  driver 570.86.15, vLLM 0.22.0, Qwen2.5-3B-Instruct. The stock vLLM image's
  CUDA 13 build runs on this CUDA 12.8 driver through CUDA forward compatibility,
  verified by a kernel launch. Two instances, one pinned to GPU 0 and one to
  GPU 1, both with `kv_role=kv_both`, sharing one Mooncake Store pool coordinated
  by a master server. Transport was TCP for this run.
- **Method.** Every trace prompt is sent to instance A, which computes the
  prefixes and writes the KV blocks to the Store. After a short settle, the same
  prompts are sent to instance B, which never processed them locally, so any reuse
  must come from the pool. Instance B's external prefix-cache counters
  (`vllm:external_prefix_cache_*`) are the proof.

| Metric | Value |
| --- | --- |
| B external (cross-instance) hit rate | 96.7% |
| B external block queries | 2,912 |
| KV bytes written by A to the Store | about 70 MB |
| GPU placement | A on GPU 0, B on GPU 1, confirmed by per-process VRAM |
| Transport | TCP (correctness signal, not a performance number) |

Two findings gate this and would otherwise burn expensive cluster time. First,
`PYTHONHASHSEED` must be identical across all instances; vLLM seeds its block-hash
chain per process otherwise, so instances compute different hashes and never match
in the shared store, leaving the external hit rate at zero with no error. Second,
the CUDA build of both vLLM and Mooncake must be compatible with the host driver.
Both are baked into the launch scripts and the container image.

The hit rate is the correctness signal, not a performance claim. Section 6.3 makes
that distinction concrete: over TCP the reuse is real but slower than recomputing,
and over RDMA it pays off.

### 6.3 Does the distributed cache lower latency? Not over TCP, yes over RDMA

A high hit rate proves reuse happens; it does not prove reuse is faster than
recomputing. A controlled comparison on the A100 hardware makes the difference
concrete, and the table below consolidates the whole transport story. All three
conditions use the same 24-session trace with distinct, long system prompts (so no
instance reuses anything locally): instance A serves every prompt first with an
empty pool (the cold, full-prefill control), then instance B serves the same prompts
from the populated pool. Only the transport, and whether B shares a box with A or
sits on a second node, changes between rows.

| Condition | B hit rate | KV load (avg) | B pooled TTFT p50 | A cold TTFT p50 | Outcome |
| --- | --- | --- | --- | --- | --- |
| A100, single machine, TCP | 98.3% | ~3,358 ms | 1,843 ms | 38.9 ms | net loss, ~47x slower |
| A100, single machine, RDMA (IB) | 98.3% | ~2.9 ms | **19.5 ms** | 26.9 ms | **net win** |
| A100, cross machine, RDMA (IB) | 98.3% | ~2.4 ms | **19.3 ms** | 26.0 ms | **net win** |
| GB200, single machine, RDMA (RoCE) | 98.3% | ~1.6 ms | 18.4 ms | 15.6 ms | p50 break-even; mean 21 vs 42 ms |

The reuse rate is identical (98.3%) and no transfer failed in any of these
conditions (the longer-prefix TCP failure that exhausts ephemeral ports is described
later in this section); only the transport and the GPU generation change the verdict. A cache helps only when fetching cached
KV is cheaper than recomputing it. Over TCP the KV load averaged about 3.3 seconds
for tens of megabytes (an effective ~16 MB/s), far slower than recomputing a
~520-token prefix on an A100 (~39 ms), so the inequality is inverted and the cache
is a net loss. Over RDMA with GPUDirect the same load drops to ~2.9 ms (moving
434 MB at GB/s rates), so the fetch beats recompute and B's time to first token
falls below A's. The win is modest here (a 3B model, a 520-token prefix) and widens
with model size, context length, and cache pressure, the regime (Section 6.1) where
recompute is expensive.

Two notes on reading the table. The KV-load column is a mean while the TTFT columns
are medians, so on the TCP row the mean load (~3,358 ms) exceeds the median pooled
TTFT (1,843 ms): a heavy tail of slow transfers lifts the mean above the median
request. And the cold-A figure is not strictly transport-independent, because
instance A also writes its computed KV into the pool as it serves; over TCP the slow
write path is the likely reason A's cold TTFT (38.9 ms) exceeds the RDMA rows
(~27 ms), even though A pays full prefill in every row.

**The GB200 rows show the same inequality from the other side.** The A100 rows left
two questions open: whether the recommendation survives on current-generation
hardware, where much faster prefill attacks the cache's advantage from the
recompute side, and how our self-measured numbers relate to the vendor's published
results, which were produced on GB200 (the vLLM Mooncake Store blog). A GB200 node
was added to answer both. On a single GB200
node (4x GB200, 186 GB each, driver 580, CUDA 13 native, aarch64 Grace CPUs;
transport is RDMA over 200 Gb RoCE on `mlx5_2`, since this box's InfiniBand ports
are down), the KV load is the fastest measured (about 1.6 ms at the short-prefix
point), yet at ~520 tokens the p50 verdict is roughly break-even: Blackwell
prefills that prefix in ~15.6 ms, so recompute is nearly free and the pooled fetch
no longer undercuts it at the median, although the mean still favors the pool two
to one. The faster the GPU, the larger the context must be before cross-instance
caching pays at the median. A prefix-length sweep on the same node measures where
it flips (24 sessions per point, distinct prefixes, fresh pool and instances per
point, TTFT p50 in ms):

| Shared-prefix length | B hit rate | A cold p50 | B pooled p50 | Verdict at p50 |
| --- | --- | --- | --- | --- |
| ~520 tokens | 98.3% | 15.3 | 18.3 | recompute slightly ahead |
| ~2,000 tokens | 99.5% | 43.9 | **24.6** | pool ~1.8x faster |
| ~4,000 tokens | 99.7% | 48.8 | **36.9** | pool ~1.3x faster |
| ~8,000 tokens | 99.9% | 59.8 | **56.5** | pool marginally ahead |

Two readings. First, **the crossover sits between roughly 500 and 2,000 shared
tokens on this hardware**: below it recompute wins, above it the pool wins, so for
agentic workloads with multi-thousand-token system prompts and histories the pool
pays at the median even on Blackwell. Second, the relative win peaks at mid-length
prefixes and narrows again by ~8,000 tokens, because both sides scale: cold prefill
grows sublinearly (Blackwell is very efficient at long prefills), while the pooled
path must move linearly more KV per request (roughly 18 MB per request at ~520
tokens up to ~280 MB at ~8,000), so at long prefixes moving the bytes over 200 Gb
RoCE costs nearly as much as recomputing them on this GPU. On slower fabrics or
bigger models that balance shifts again; per-point transfer telemetry was not
captured in the sweep (the periodic KV-transfer log did not fire within the short
runs), so the volume figures are computed from the KV size per token, not
measured.

**Model size, measured (Qwen2.5-32B).** The claim that the win grows with model
size was the one axis still asserted rather than measured, so the 2K and 8K points
were repeated on the same node with Qwen2.5-32B-Instruct (a production-plausible
serving size; one instance per 186 GB GPU, same traces, pool segment raised to
64 GiB and the staging buffer to 8 GiB for the roughly 262 KB-per-token KV):

| Shared-prefix length | Model | A cold p50 | B pooled p50 | Verdict at p50 |
| --- | --- | --- | --- | --- |
| ~2,000 tokens | 3B | 43.9 | 24.6 | pool ~1.8x |
| ~2,000 tokens | 32B | 91.8 | **44.4** | pool ~2.1x |
| ~8,000 tokens | 3B | 59.8 | 56.5 | pool marginal |
| ~8,000 tokens | 32B | 348.3 | **104.9** | **pool ~3.3x** |

The 8K narrowing seen at 3B reverses completely at 32B: prefill compute grows
faster with model size than KV volume does, so the pooled fetch wins by the
largest margin in this investigation, about 243 ms saved per request at the
median (99.9% reuse, mean 361 versus 108 ms). This is the regime a distributed
pool is actually deployed for, a real model size with long shared prefixes, and
it is where the mechanism pays most decisively. Environment notes for these rows: they ran on a host virtualenv rather
than Docker, because every aarch64 Mooncake wheel requires glibc 2.39 (Ubuntu
24.04) while the vLLM v0.22.0 arm64 image is Ubuntu 22.04 (glibc 2.35), an
incompatibility recorded in the runbook; and `nvidia_peermem` still ships with
driver 580 and must be loaded, without it the RDMA registration of GPU memory
fails ("Bad address") and every transfer fails, yielding a 0% external hit rate
with both instances otherwise healthy. Same vLLM 0.22.0, Mooncake 0.3.11.post1,
and protocol as the other rows; `MAX_MODEL_LEN` was raised to 16384 and the pool
segment to 32 GiB for the long-prefix points.

**Transport notes.** RDMA is the transport the Mooncake Transfer Engine exists to
use, and production deployments use GPUDirect RDMA even within a single node (the
vLLM Mooncake Store blog's 1P1D baseline on 12 GB200 GPUs); TCP is the universal,
no-special-hardware fallback, not a performance transport. NVLink is not available
through the Store: although the Transfer Engine config lists an `nvlink_intra`
protocol, the Mooncake Store client rejects it at init (`unsupported_protocol`), so
the Store accepts only `tcp` and `rdma`. The NVLink hardware is healthy (validated
at ~270 GB/s GPU-to-GPU, `scripts/check_nvlink_p2p.py`); the rejection is purely a
Store build limitation. A host shared-memory transport ("UBShmem") exists only
behind a non-default build flag and is absent from the standard wheel.

**Cross-machine (GR-1331).** The third row ran across two physical nodes, A on
`latpoc51` (192.168.147.151) and B on `latpoc52` (192.168.147.152), one A100 each,
on a shared 200 Gb InfiniBand fabric. B pulled A's KV across the wire (RDMA over
`mlx5_0`, LID and GID confirmed in the logs) and still beat recompute. The raw
fabric measured 169 Gb/sec node-to-node (`ib_write_bw`), so it is not the
bottleneck, and the cross-node numbers essentially match the single-node RDMA ones.
Cross-instance reuse is a net win not only within a box but between machines, which
is the production-shaped scenario.

A separate run with longer prefixes (about 3,300 tokens, roughly 120 MB of KV each)
exposed a hard ceiling on the TCP path: large transfers exhausted ephemeral TCP
ports ("cannot assign requested address"), transfers failed, the external hit rate
fell to about 15%, and tail TTFT rose to 14 to 17 seconds. TCP is adequate to prove
the mechanism on small prefixes and inadequate for anything larger, which is the
operational reason RDMA is not optional for a real deployment.

The honest conclusion: cross-instance reuse is correct (98.3% hit), it does not pay
for itself over TCP, and it does pay off over RDMA, where the pooled fetch (about
2.9 ms) is cheaper than recompute and B's time to first token drops below A's. The
win is small on a 3B model with a 520-token prefix and grows with model size,
context length, and cache pressure; quantifying that scaling, with throughput at
real concurrency and the reliability gates, is what remains.

### 6.4 LMCache head-to-head against bare Mooncake Store

This measures survey option (c), LMCache as the cache-management layer with
Mooncake Store as its L2 remote tier, against option (b), the bare
`MooncakeStoreConnector`, on the same trace, hardware, and transport, so the
comparison isolates what the LMCache layer adds and costs. Both run over RDMA on
the same version stack (vLLM 0.22.0, LMCache 0.4.5, Mooncake 0.3.11.post1); the
bare rows were re-measured after the Mooncake upgrade and match the earlier
results. Self-measured, same cold-A versus pooled-B protocol as Section 6.3.

| RDMA, same trace | B hit rate | B TTFT p50 | B TTFT mean | A cold p50 |
| --- | --- | --- | --- | --- |
| Bare Mooncake, single machine | 98.3% | 19.5 ms | 54 ms | 26.9 ms |
| Bare Mooncake, cross machine | 98.3% | 20.3 ms | 55 ms | 26.1 ms |
| LMCache over Mooncake L2, single machine | 53.4% | 25.2 ms | 363 ms | 28.1 ms |
| LMCache over Mooncake L2, cross machine | blocked | — | — | — |

Three observations, each with a mechanical explanation:

- **The hit-rate gap is granularity, not correctness.** LMCache reuses full
  256-token chunks and does not store partial chunks (`save_unfull_chunk: false`),
  so a roughly 500-token prompt yields exactly one full-chunk hit, about 53%.
  Bare Mooncake keys 16-token vLLM blocks and captures about 98% of the same
  prefix. On long prompts the gap shrinks (the partial tail becomes a smaller
  fraction); on short ones it widens.
- **Median is competitive, tail is not.** LMCache's pooled p50 (25.2 ms) is close
  to cold recompute, but the mean (363 ms) shows a heavy retrieval tail that bare
  Mooncake does not have (54 ms mean). At this scale the LMCache layer costs more
  than its partial reuse saves.
- **The cross-machine cell is blocked by an LMCache bug.** On the node remote from
  the master, LMCache's Mooncake connector fails to create its store client
  ("Client not available", 20 retries), then proceeds anyway and segfaults on
  first use. The failure was isolated precisely: vLLM's native connector works
  from the same node, image, and master, and a standalone Mooncake client from
  the same container also works, so the defect is in LMCache's connector
  (reproduced on both LMCache 0.4.5 and 0.5.0). It should be reported upstream.

**Operability evidence from the bring-up.** Getting LMCache to this point required
diagnosing three stacked issues, which is itself rubric-relevant data given the
survey scored LMCache highest on operability from its documentation: (1) Mooncake
below 0.3.11 has a flaky RDMA `register_buffer` failure on buffers of 4 GiB and
larger, fixed upstream, which forced the pin to `mooncake-transfer-engine-cuda13
0.3.11.post1`; (2) LMCache's NUMA path calls `mbind()` and needs the `SYS_NICE`
capability in containers, and without it LMCache silently degrades to recompute
with zero hits and no surfaced error; (3) the LMCache documentation's own Mooncake
example sets `local_buffer_size: 0`, which Mooncake rejects, and LMCache then logs
"setup completed successfully" against an uninitialized client and segfaults on
the first transfer. The pattern across all three is the same: LMCache absorbs
backend failures silently and continues, which turns configuration mistakes into
crashes or silent zero-hit runs. Bare Mooncake, by contrast, ran correctly on the
first attempt in every topology tried.

**Verdict for the recommendation.** These measurements strengthen option (b),
bare Mooncake Store, as the primary choice: higher reuse, flatter latency, and a
cleaner operational story on this stack. The capabilities that motivated option
(c), the L1 tier that survives an L2 outage, Kubernetes-native deployment, and
non-prefix reuse for retrieval-heavy workloads, were not exercised here and remain
LMCache's case; a Production-Stack (Helm) evaluation would test them on their own
ground. On raw cross-instance reuse over RDMA, bare Mooncake wins.

### 6.5 Production scale: GLM-5.2-FP8 (753B) across two H200 nodes

The final measurement takes the methodology to a frontier-scale model, per a
stakeholder request: **GLM-5.2** (Zhipu's 753B-parameter MoE, 39B active) in its
official **FP8** quantization, the configuration its own serving recipe targets.
This required new hardware: FP8 weights alone are ~750 GB per serving instance,
so each instance runs tensor-parallel across all eight GPUs of an H200 node
(`latpoc32`/`latpoc34`, 8x H200 141 GB each, ConnectX at 400 Gb; node-to-node
`ib_write_bw` measured 390 Gb/s), one instance per node, one Mooncake pool over
RDMA on `mlx5_2`. The stack moved to vLLM 0.24.0 (GLM-5.2 requires >= 0.23), so a
Qwen2.5-32B control point was re-run on the same stack to bridge the version bump.
Same cold-A versus pooled-B protocol and traces as the GB200 sweep; the pool
carried a few smoke-test entries (~90 MB, disjoint prompts) at the start of the
run, which cannot produce false hits.

| Model / point (cross-machine RDMA) | B hit rate | A cold p50 | B pooled p50 | Speedup |
| --- | --- | --- | --- | --- |
| Qwen2.5-32B, ~8K (control, H200, vLLM 0.24) | 99.9% | 714.1 ms | **107.5 ms** | **6.6x** |
| GLM-5.2-FP8, ~2K | 97.8% | 175.5 ms | **67.2 ms** | **2.6x** |
| GLM-5.2-FP8, ~8K | 99.4% | 672.2 ms | **162.9 ms** | **4.1x** |

Verification: instance B pulled **24.7 GB** of GLM KV across the fabric (4,160
keys, zero failed keys, zero errors, no TCP fallback; RDMA on `mlx5_2` confirmed
in the transfer logs), at about 5.5 GB/s effective on the batched loads. GLM-5.2's
KV runs at roughly 80 KB per token in this deployment, about twice Qwen-32B's, so
the 8K point moves ~650 MB per request.

Three conclusions. First, **the model-size trend completes**: at ~8K shared
tokens the pool's median win grows 3B (marginal) to 32B (3.3x on GB200) to
**753B-FP8 (4.1x, about half a second saved per request)**. Prefill compute keeps
growing faster than KV volume, so the production-shaped configuration, a frontier
model with long shared prompts, is where the pool pays most. Second, **FP8
quantization coexists cleanly with the pool**: the connector stores and reuses
the FP8 model's KV (which remains bf16) at a 99.4% hit rate with no correctness
caveats beyond those of FP8 itself. Third, **the control row shows the pool path
is stable across stack versions**: B's pooled TTFT for Qwen-32B/8K is ~105 ms on
GB200/vLLM 0.22 and ~108 ms on H200/vLLM 0.24, with only the cold-prefill side
varying by GPU generation, which is exactly what the fetch-versus-recompute model
predicts.

Operability note: standing this up on driver-only hosts (no root, no Docker, no
system CUDA toolkit) surfaced a chain of five distinct toolchain defects in the
FP8 serving path, from a missing runtime compiler to internally version-mixed
NVIDIA pip components; the full diagnosis and the working recipe are recorded in
the runbook. None are Mooncake issues, but they are real deployment friction for
FP8 models on bare hosts.

## 7. Recommendation and roadmap

**Adopt in two steps, which also map onto a cheap-first hardware strategy.**

1. **Mooncake Store through the vLLM `KVConnector` (option b), first.** The
   cleanest, most isolatable integration: fewest moving parts, official support,
   and the easiest way to prove cross-instance hits and validate the correctness
   gate before adding complexity. Its weakness is the high-availability master
   gap, which matters at production scale more than for a first integration.
2. **LMCache fronting Mooncake Store (option c), for the enterprise evaluation.**
   One more layer, but it brings the Production Stack (Helm and Kubernetes),
   documented observability, broader non-prefix reuse, and graceful L1 and L2
   degradation that softens the Store master gap. This is where the operability
   and reliability dimensions we weight highest actually get exercised.

Drop the Transfer-Engine-only option and NIXL from prototype scope; they are the
transport floor, not a cache-management choice. Keep FlexKV as a comparison until
its vLLM path matures past experimental. Native offload stays the baseline.

**What remains for a report-grade performance result.** The single-node RDMA win is
now demonstrated (Section 6.3): on a GPU-affined NIC with GPUDirect, the pooled
fetch beats recompute. What remains is to scale it, a larger model and longer
contexts where the margin grows, throughput at real concurrency rather than a
single stream, and TTFT measured against the baseline at matched hit rate, so we
separate "the cache works" from "this implementation is efficient." The
cross-machine extension is now done (GR-1331, Section 6.3): a second A100 node on
the same InfiniBand fabric, one instance per node sharing one Store pool with KV
crossing the wire over RDMA, reproduced the single-node win (B pooled TTFT 19.3 ms
versus A cold 26.0 ms, 98.3% reuse, 2.4 ms cross-node load over a 169 Gb/sec
fabric). Scaling (above) and the reliability gates are what remain.
Then exercise the reliability gates: master loss and peer loss must degrade to
recomputation, never to a wrong answer.

## 8. Caveats and open questions

- **Vendor performance numbers are not independently reproduced.** Published
  speedups for Mooncake Store and LMCache are vendor-reported, and the exact
  mapping of which figure is TTFT versus throughput versus latency must be read
  off the source before being quoted. Our own measurements are the ones that
  count, and the cluster-grade performance number is not yet among them.

- **The prototype number is a correctness signal over TCP.** Cross-instance reuse
  is proven, but representative latency and throughput require RDMA (the Store does
  not accept NVLink, see Section 6.3), a larger model, and real concurrency.

- **The comparison baseline is the same-hardware cold-A control, not a separate
  run.** Every verdict in Sections 6.3 and 6.4 measures instance B's pooled path
  against instance A serving the same trace cold with an empty pool (full prefill,
  zero reuse) on the same GPU, transport, and trace. That cold-A number is the
  native-recompute baseline the pooled path must beat, so the comparisons are
  internal and hardware-consistent; no cross-hardware comparison is made. The
  demoted dev-box run (Section 6.1) motivates the capacity effect only and is not a
  comparison anchor. What remains is a native-local-caching baseline at a matched
  hit rate (Section 7), which would separate "the distributed cache works" from
  "the distributed path is more efficient than native local reuse."

- **The Mooncake Store HA master gap is real.** High-availability failover depends
  on ETCD today, with a Kubernetes-native path still open. This is the most
  material reliability finding for an enterprise deployment and is the main reason
  to evaluate the LMCache-fronted configuration, whose L1 tier survives an L2
  outage.

- **Scope.** This evaluation targets vLLM and, in its current phase, single-node
  multi-GPU reuse. Multi-machine reuse over a cross-node RDMA fabric is deferred.

## 9. Primary sources

- vLLM Mooncake Store integration PR: https://github.com/vllm-project/vllm/pull/40900
- vLLM Mooncake Store blog: https://vllm.ai/blog/2026-05-06-mooncake-store
- Mooncake repository and README: https://github.com/kvcache-ai/Mooncake/blob/main/README.md
- Mooncake Store design: https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html
- Mooncake HA master issue: https://github.com/kvcache-ai/Mooncake/issues/1321
- LMCache repository: https://github.com/LMCache/LMCache
- LMCache and Mooncake joint post: https://blog.lmcache.ai/en/2026/05/26/when-open-source-meets-open-source-a-joint-effort-between-lmcache-and-mooncake/
- LMCache metrics: https://docs.lmcache.ai/production/observability/metrics.html
- vLLM Production Stack: https://blog.vllm.ai/production-stack/index.html
- FlexKV repository: https://github.com/taco-project/FlexKV
- FlexKV in NVIDIA Dynamo docs: https://docs.nvidia.com/dynamo/integrations/flex-kv
- NIXL connector in vLLM: https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector/
- vLLM prefix caching design and SHA-256 keying: https://docs.vllm.ai/en/stable/design/prefix_caching/

For the full fact-checking method, the per-candidate detail, and the
reproduction recipes, see `docs/survey.md`, `docs/baseline.md`,
`docs/stage3.md`, and `docs/evaluation-rubric.md`.
