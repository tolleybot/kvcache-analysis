# Distributed KV Caching for vLLM Inference Serving: Investigation and Recommendation

A standalone report on what distributed KV caching solutions are available for
large language model inference serving, how they compare, and which to adopt. It
consolidates the staged work in this repository (the evaluation rubric, the
sourced survey, the baseline measurement, and the cross-instance prototype) into
a single document, and it calls out the points most likely to interest a machine
learning group. The detailed working docs remain in `docs/` for reproduction and
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
sharing one Mooncake Store pool, one per GPU on an 8x A100 node, reached a
**96.7% cross-instance hit rate** on the instance that never computed the
prefixes, proving the mechanism end to end.

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

Two measurements from this project anchor the recommendation. State the exact
environment with every number, since absolute values depend on hardware.

### 6.1 Baseline: native vLLM prefix caching

This establishes the capacity effect that motivates a pool, and it confirms the
harness measures hit rate correctly.

- **Environment.** NVIDIA GeForce RTX 5070, 12 GB, vLLM 0.22.0, Qwen2.5-3B-Instruct
  in bf16, greedy decoding, single instance, local only. A development-tier box;
  relative effects transfer, absolute latencies do not.
- **Method.** A synthetic agentic, multi-turn trace (16 sessions, 4 turns each, a
  shared system prompt, round-robin interleaving) with the shared-prefix fraction
  swept across 0, 50, and 90 percent. Two cache sizings isolate capacity: ample
  (about 99,440 tokens) and constrained (4,096 tokens, forcing eviction).

| Shared prefix | Hit rate, ample | Hit rate, constrained | TTFT mean, ample (ms) | TTFT mean, constrained (ms) |
| --- | --- | --- | --- | --- |
| 0% | 68.6% | 0.0% | 32.1 | 72.5 |
| 50% | 76.5% | 30.6% | 26.8 | 53.5 |
| 90% | 90.4% | 61.5% | 22.1 | 36.0 |

The reading: hit rate tracks prefix sharing as designed, within-session reuse is
significant on its own (68.6% even at 0 percent cross-session sharing), and
shrinking the cache collapses the hit rate while roughly doubling TTFT. The gap
between the constrained and ample columns is the prize a distributed pool exists
to recover. Throughput and end to end latency barely move at this scale because,
with a small model, short outputs, and concurrency one, total latency is dominated
by decode rather than prefill; the throughput effect grows with larger models,
longer prompts, and higher concurrency.

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
that distinction concrete: over TCP the reuse is real but slower than recomputing.

### 6.3 Does the distributed cache lower latency over TCP? Not here

A high hit rate proves reuse happens; it does not prove reuse is faster than
recomputing. A controlled comparison on the A100 node makes the difference
concrete. Using 24 sessions with distinct, long system prompts (so no instance can
reuse anything locally), instance A served every prompt first with an empty pool,
which is the cold full-prefill control, and instance B then served the same prompts
from the now-populated pool.

| Instance | Role | External hit rate | TTFT p50 | TTFT mean |
| --- | --- | --- | --- | --- |
| A | cold, full prefill | 0.0% | 38.9 ms | 75 ms |
| B | pooled, cache hit | 98.3% | 1,843 ms | 2,085 ms |

B reused 98.3% of A's KV with zero transfer failures, yet its time to first token
was roughly 47 times worse. The cause is the transport: B's measured KV load
averaged about 3.3 seconds for tens of megabytes, an effective rate near 16 MB/s,
which is far slower than recomputing a roughly 520-token prefix on an A100 (about
39 ms).

This is the central caveat stated as a measurement. A cache helps only when
fetching cached KV is cheaper than recomputing it. On a fast GPU with a small model
and TCP transport, that inequality is inverted, so the distributed cache is a net
latency loss. It flips in favour of the cache on two axes, both of which this run
deliberately sits on the wrong side of:

- **Cheap fetch.** RDMA is the transport the Mooncake Transfer Engine exists to
  use; it moves KV at fabric speeds rather than ~16 MB/s, and production deployments
  use GPUDirect RDMA even within a single node (the vLLM Mooncake Store blog's 1P1D
  baseline on 12 GB200 GPUs). This is the deferred next step and what makes the
  fetch competitive. A correction from testing: although the Transfer Engine config
  lists an `nvlink_intra` protocol, the Mooncake Store client that vLLM uses
  **rejects it at init** (`unsupported_protocol protocol=nvlink_intra`), so the
  Store path supports only `tcp` and `rdma`. The single-node non-TCP fix is
  therefore **RDMA, not NVLink**; an NVLink path would need a different connector
  and is out of scope here. TCP is documented as the universal fallback that needs
  no special hardware, not a performance transport. A host shared-memory transport
  ("UBShmem") exists only behind a non-default build flag and is absent from the
  standard wheel.
- **Expensive recompute.** Much larger models, much longer contexts, or capacity
  pressure where the local alternative is eviction and a cascade of misses rather
  than a cheap prefill (the regime Section 6.1 isolates). When recompute is slow,
  even a moderate-speed fetch wins.

A separate run with longer prefixes (about 3,300 tokens, roughly 120 MB of KV each)
exposed a hard ceiling on the TCP path: large transfers exhausted ephemeral TCP
ports ("cannot assign requested address"), transfers failed, the external hit rate
fell to about 15%, and tail TTFT rose to 14 to 17 seconds. TCP is adequate to prove
the mechanism on small prefixes and inadequate for anything larger, which is the
operational reason RDMA is not optional for a real deployment.

The honest conclusion: the prototype proves cross-instance reuse is correct, and it
proves that over TCP that reuse does not pay for itself. No cross-instance
performance gain is claimed from this tier; demonstrating one is exactly what the
RDMA step in Section 7 is for.

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

**What remains for a report-grade performance result.** Move the prototype off TCP
to RDMA over a GPU-affined NIC (the Store accepts only `tcp` and `rdma`, not
NVLink), run a larger model under real concurrency, and measure TTFT and throughput
against the baseline at matched hit rate, so we separate "the cache works" from
"this implementation is efficient."
Then exercise the reliability gates: master loss and peer loss must degrade to
recomputation, never to a wrong answer.

## 8. Details of interest to a machine learning group

The points below are where this systems work touches modeling, training-adjacent
concerns, and experiment design, and they are the ones a machine learning group
will care about most.

- **Reuse is bit-exact, so output quality is unchanged.** This is not approximate
  or lossy caching. A block is reused only when a content hash over its token ids
  and the entire preceding prefix matches, so the reused KV is identical to what
  recomputation would produce. Under greedy decoding the outputs are
  byte-for-byte identical with and without the cache. The quality question is
  therefore moot, the only risk is a keying bug, which is exactly why the
  correctness invariant is a hard gate rather than a metric.

- **Hit rate is the lever, and prompt structure controls it.** Every downstream
  gain is bounded by the fraction of prompt tokens served from cache. That
  fraction is a property of the workload, not the hardware, and it is something a
  model or product team can engineer. Stable, shared system prompts placed at the
  front, consistent tool and few-shot ordering, and append-only conversation
  histories all raise the hit rate. Reordering or templating that perturbs an
  early token invalidates every block after it, since the hash chains forward.

- **Caching is block-level, not token-level.** KV is keyed and stored per fixed
  block of tokens (16 in vLLM by default). Reuse happens at block granularity, so
  a divergence inside a block costs the whole block, and block size trades hit
  granularity against per-block metadata and transfer overhead. This is worth
  knowing when reasoning about why a near-identical prompt got a lower hit rate
  than expected.

- **Where the savings land depends on the regime.** Prefill reuse most directly
  cuts time to first token. At low concurrency with short outputs, total latency
  is decode-bound and the win shows up almost entirely in TTFT, which is why the
  baseline's throughput barely moved. With larger models, longer prompts, and
  higher concurrency, prefill is a larger share of the work and the saved
  computation converts into throughput and capacity. Experiment design should
  match the regime to the claim being made.

- **KV cache size scales with architecture, and that sets the transfer budget.**
  Cache size grows with layers, number of key and value heads, head dimension,
  sequence length, and dtype. Grouped-query and multi-query attention shrink it
  substantially by sharing key and value heads, and fp8 or other KV quantization
  halves or quarters it again. Smaller KV per token means more fits in the pool
  and less moves on a hit, so architecture choices directly change the economics
  of distributed caching. A quantized KV cache also raises a quality question that
  exact-reuse prefix caching does not, and the two should not be conflated.

- **Determinism is a hard requirement across instances.** Because keys are
  content hashes seeded per process, two instances only agree if their hash seed
  agrees. In practice this means pinning `PYTHONHASHSEED` identically everywhere.
  More generally, anything that makes block hashing nondeterministic across
  instances silently defeats cross-instance reuse without raising an error, so it
  is a thing to watch when reproducing or scaling experiments.

- **Cross-instance reuse decouples routing from cache locality.** Once the pool is
  shared, any instance can serve any session and still find its prefix, which
  removes the need for sticky session routing and enables prefill and decode
  disaggregation. For a research team this is the structural difference from
  local offload: the cache stops being a per-replica resource and becomes a
  cluster-wide one.

- **Non-prefix reuse is relevant to retrieval-augmented generation.** LMCache can
  reuse KV for any repeated span, not only a shared prefix. In a RAG setting the
  same document chunks recur across requests in different positions, where
  prefix-only caching misses them. This is a capability difference worth weighing
  if the workload is retrieval-heavy rather than purely conversational.

- **Capacity versus working set is the whole argument.** The baseline shows that
  when the working set exceeds the per-instance cache, eviction drives the hit
  rate to zero under interleaved load. A distributed pool gives a larger effective
  cache than any single instance can hold, so the relevant quantity is the ratio
  of aggregate pool capacity to the working set of active prefixes, not the
  per-GPU cache size.

## 9. Caveats and open questions

- **Vendor performance numbers are not independently reproduced.** Published
  speedups for Mooncake Store and LMCache are vendor-reported, and the exact
  mapping of which figure is TTFT versus throughput versus latency must be read
  off the source before being quoted. Our own measurements are the ones that
  count, and the cluster-grade performance number is not yet among them.

- **The prototype number is a correctness signal over TCP.** Cross-instance reuse
  is proven, but representative latency and throughput require RDMA (the Store does
  not accept NVLink, see Section 6.3), a larger model, and real concurrency.

- **Baseline and prototype ran on different hardware.** The baseline is from a
  12 GB consumer GPU and the prototype from 8x A100, so the two are not directly
  comparable in absolute terms. The baseline establishes the capacity effect; the
  prototype establishes the cross-instance mechanism. A re-baseline on the A100
  node is the clean way to put both on one axis.

- **The Mooncake Store HA master gap is real.** High-availability failover depends
  on ETCD today, with a Kubernetes-native path still open. This is the most
  material reliability finding for an enterprise deployment and is the main reason
  to evaluate the LMCache-fronted configuration, whose L1 tier survives an L2
  outage.

- **Scope.** This evaluation targets vLLM and, in its current phase, single-node
  multi-GPU reuse. Multi-machine reuse over a cross-node RDMA fabric is deferred.

## 10. Primary sources

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
