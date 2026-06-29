# Stage 1 Survey: Distributed KV Caching Solutions for vLLM

> Stage 1 working detail (the sourced candidate survey and fact-checking method). The consolidated investigation and results are in [`report.md`](report.md).

Sourced survey and comparison of distributed KV caching solutions for vLLM
serving, current as of June 2026, scored against the Stage 0 rubric
(`evaluation-rubric.md`). The purpose is the Stage 1 gate: choose which
configuration to prototype first.

Method: fan-out web search across six angles, 18 primary and forum sources
fetched, 84 candidate claims extracted, top 25 adversarially fact-checked with a
three-vote scheme. 24 confirmed, 1 refuted. Claims that could not be tied to a
primary source are marked unverified. Published performance numbers are flagged
vendor-reported unless an independent source corroborates them.

## 1. The layering model

The single most useful framing, and one the evidence supports across every
candidate, is that these are not all competing at the same level. There are two
layers:

- **Transport layer.** Moves KV bytes between memory tiers and across the
  network. Mooncake's **Transfer Engine** is the dominant implementation here,
  and NIXL is a parallel abstraction that can use the Transfer Engine as a
  backend. This layer does not decide what to cache or when to evict.
- **Cache-management layer.** Decides what to store, how to key it, when to
  evict, and how to look it up across instances. **Mooncake Store**, **LMCache**,
  and **FlexKV** all live here, and they all sit *on top of* the Transfer Engine
  for their actual data movement.

This is why the choice is not "Mooncake versus the others." Mooncake's Transfer
Engine tends to be the transport *inside* the alternatives. The real decision is
which cache-management layer to adopt, and the Transfer Engine comes along
underneath either way.

Sources: Mooncake Transfer Engine provides a unified transport interface
(https://github.com/kvcache-ai/Mooncake/blob/main/README.md); FlexKV uses the
Mooncake Transfer Engine for cross-node transfer
(https://github.com/taco-project/FlexKV, https://docs.nvidia.com/dynamo/integrations/flex-kv);
NIXL connector in vLLM (https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector/).

## 2. Candidate profiles

### Mooncake Transfer Engine (transport)

- **What it is.** The transport substrate: batched transfer across DRAM, VRAM,
  and NVMe over TCP, RDMA (InfiniBand, RoCEv2, eRDMA), GPUDirect, NVLink, and CXL.
- **Status.** Apache-2.0, the serving backbone for Moonshot AI's Kimi, part of
  the PyTorch ecosystem. vLLM officially supports the Transfer Engine.
- **Verdict.** Not a standalone solution. It is the floor every option is built
  on. Prototyping it alone (option a) means building the cache-management layer
  ourselves, which contradicts the operability priority.

Sources: https://github.com/kvcache-ai/Mooncake/blob/main/README.md, Apache-2.0
and Kimi production use confirmed.

### Mooncake Store (cache-management, the lead candidate)

- **What it is.** Distributed pooled KV cache built on the Transfer Engine, with
  a master server managing metadata and service discovery and clients on GPU
  nodes.
- **vLLM integration.** Integrated through the existing `KVConnector` interface,
  composable via `MultiConnector`. The full-Store integration landed in vLLM
  **PR #40900**, released in **v0.21.0**. This is the most direct, officially
  supported path. (https://github.com/vllm-project/vllm/pull/40900)
- **Reliability gap.** The high-availability master failover currently depends on
  **ETCD**, and a Kubernetes-native lease-based HA path is still an **open issue
  (#1321)** as of January 2026. This is the most material reliability finding for
  our weighting. (https://github.com/kvcache-ai/Mooncake/issues/1321)
- **Performance (vendor-reported, vLLM blog).** On a Codex agentic-trace
  benchmark on 12 GB200 GPUs, reported cache hit rate spanning 1.7% to 92.2% and
  reported speedups of roughly 3.8x, 46x, and 8.6x on different metrics. A
  scaling test on 60 GB200 GPUs with round-robin routing reported greater than
  95% cross-node cache hit rate. These are vendor numbers and the exact
  metric-to-number mapping (which figure is TTFT vs throughput vs latency) must
  be confirmed against the source before quoting in the final report.
  (https://vllm.ai/blog/2026-05-06-mooncake-store)

Sources: PR #40900, vLLM Mooncake Store blog, Mooncake Store design doc
(https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html), HA issue #1321.

### LMCache (cache-management, closest head-to-head)

- **What it is.** A cache-management layer with a two-tier hierarchy: **L1** local
  (CPU/GPU) and **L2** remote. It can use **Mooncake Store as its L2 backend**,
  documented in a joint LMCache and Mooncake write-up (May 26 2026).
- **vLLM integration.** Integrates with vLLM v1, providing CPU KV offload and
  cross-instance reuse. Ships via the official **vLLM Production Stack**, a
  Kubernetes-native, Helm-deployed serving stack. This is the strongest
  operability story in the field.
- **Capability edge.** Reuses KV for **any reused text, not only shared
  prefixes**, which is broader than prefix-only reuse.
- **Reliability shape.** The L1/L2 hierarchy is relevant to our safe-degradation
  gate: if the L2 (Mooncake Store) backend is unavailable, L1 local cache still
  serves. This softens the Store HA-master gap when LMCache fronts it.
- **Observability.** Exposes a documented metrics surface (hit rate and related)
  out of the box, plus a vLLM metrics endpoint.
- **Performance (vendor-reported).** Claims 3x to 10x delay savings and GPU cycle
  reduction. Vendor numbers, not independently verified.

Sources: https://github.com/LMCache/LMCache,
https://blog.lmcache.ai/en/2026/05/26/when-open-source-meets-open-source-a-joint-effort-between-lmcache-and-mooncake/,
https://blog.vllm.ai/production-stack/index.html,
https://docs.lmcache.ai/production/observability/metrics.html,
https://docs.lmcache.ai/production/observability/vllm_endpoint.html.

### FlexKV (cache-management)

- **What it is.** A unified KV caching layer (Tencent, in the Dynamo ecosystem)
  that uses the Mooncake Transfer Engine for cross-node RDMA transfer.
- **vLLM integration.** Its disaggregated-serving integration with vLLM is
  **still experimental**. A specific claim that FlexKV merged natively into vLLM
  mainline as `FlexKVConnectorV1` in v0.17.2 on March 17 2026 was **refuted** by
  the verification step (1 confirm, 2 refute), so do not rely on it.
- **Verdict.** Promising but less mature on the vLLM path than Store or LMCache.
  Keep as comparison, not as first prototype.

Sources: https://github.com/taco-project/FlexKV,
https://docs.nvidia.com/dynamo/integrations/flex-kv.

### NIXL (transport)

- **What it is.** NVIDIA's data-movement abstraction. A transport-layer peer to
  the Transfer Engine, exposed in vLLM via a NIXL connector. It can use the
  Mooncake Transfer Engine as a backend plugin. Not a full cache solution.
- **Verdict.** Same layer as the Transfer Engine, not a cache-management choice.
  Out of scope as a first prototype for the same reason as option a.

Source: https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector/.

### SGLang HiCache (context only)

Surveyed for comparison only. SGLang is not a target engine, so HiCache (which
can use Mooncake Store as a hierarchical backend) is noted and excluded from
scoring.

### Native vLLM local offload (the baseline)

The baseline every distributed option must beat. vLLM's prefix caching keys each
KV block by a hash computed from the block's token ids plus the preceding
prefix, using **SHA-256** as the default block-hash algorithm. This matters for
our correctness gate: content-addressing with SHA-256 is the mechanism that makes
cross-instance reuse safe, and it is the same keying discipline the distributed
layers extend. Baseline performance numbers are deliberately deferred to Stage 2,
where we measure them on our own trace.

Sources: https://docs.vllm.ai/en/stable/design/prefix_caching/,
https://github.com/vllm-project/vllm/issues/38474.

## 3. Comparison matrix

Scored 1 to 5 against the Stage 0 rubric. Operability and reliability carry half
the weighted score by design. Scores are directional from sourced evidence, not
yet from our own measurement.

| Candidate | Layer | vLLM integration | Operability | Reliability | Observability | Maturity | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mooncake Store | Cache-mgmt | 5, official, PR #40900, v0.21.0 | 3, k8s but HA master needs ETCD | 3, HA master not yet k8s-native (#1321 open) | 3, present, less turnkey | 5, Kimi production | Lead candidate, most direct path |
| LMCache (+ Store as L2) | Cache-mgmt | 4, vLLM v1, Production Stack | 5, official Helm/k8s stack | 4, L1 survives L2 outage | 4, documented metrics + endpoint | 4, active, widely used | Best operability, extra layer |
| FlexKV | Cache-mgmt | 2, experimental on vLLM | 3 | 3 | unverified | 3 | Native-merge claim refuted |
| Mooncake Transfer Engine | Transport | n/a alone | n/a | n/a | n/a | 5 | Substrate under all of the above |
| NIXL | Transport | connector exists | n/a | n/a | n/a | 4 | Transport peer, not a cache layer |
| Native vLLM offload | Baseline | built in | 5 | 5 | built in | 5 | The number to beat; no cross-instance hits |

## 4. The three-way architectural choice

The prototype question was framed as three options:

- **(a) Transfer Engine only, beneath a layer we build.** Rejected as a first
  prototype. It is the transport floor, and choosing it means writing our own
  cache-management, which directly contradicts the operability priority. It is a
  later optimization, not a starting point.
- **(b) Full Mooncake Store via vLLM `KVConnector`.** The most direct, officially
  integrated path (PR #40900, v0.21.0). Fewest layers, so results are easiest to
  attribute and the mechanism is cleanest to validate. Its weakness is the
  HA-master reliability gap (#1321), which matters most at production scale, less
  so for a first prototype.
- **(c) Mooncake Store via LMCache.** One more layer, but it brings the official
  Production Stack (Helm/k8s), documented observability, broader non-prefix
  reuse, and graceful L1/L2 degradation that softens the Store HA gap. This is
  the strongest fit for the operability-and-reliability weighting, at the cost of
  muddier attribution (a hit may come from L1 or L2).

The generic ranking from the research was b > c > a. Under our
operability-and-reliability-first weighting, b and c are much closer, because
LMCache's Production Stack and graceful degradation are exactly the dimensions we
overweight.

## 5. Recommendation

Prototype in two steps, which also maps cleanly onto our local-first hardware
strategy:

1. **Start with (b), full Mooncake Store via the vLLM `KVConnector`, on the local
   box.** It is the cleanest, most isolatable integration: fewest moving parts,
   official support, and the easiest way to prove cross-instance hits and
   validate the correctness gate before adding any complexity. This is ideal for
   the local correctness phase, where the goal is mechanism, not performance.
2. **Then evaluate (c), LMCache fronting Mooncake Store, for the cluster and
   enterprise evaluation.** This is where the operability and reliability
   dimensions we weight highest actually get exercised: Helm/k8s deployment,
   observability metrics, and graceful degradation when the L2 backend or its
   master is unavailable.

Drop (a) and NIXL from prototype scope; they are the transport floor, not a
cache-management choice. Keep FlexKV as comparison only until its vLLM path
matures past experimental.

The Stage 1 gate decision: **proceed to Stage 2 (baseline harness) targeting
option (b) as the first integration**, with option (c) as the planned follow-on
for the cluster tier.

## 6. Caveats and coverage gaps

- **Vendor numbers.** The Mooncake Store speedups (3.8x / 46x / 8.6x, hit rate
  1.7% to 92.2%, greater than 95% cross-node on 60 GB200) and LMCache's 3x to 10x
  are vendor-reported. No independent reproduction was found. Our own Stage 2 and
  Stage 3 numbers are the ones that count.
- **Unresolved metric mapping.** The specific Mooncake Store figures were
  compressed in synthesis; which number is TTFT vs throughput vs latency must be
  read off the source before any of them is quoted in the final report.
- **Refuted claim.** The FlexKV native-vLLM-merge claim (v0.17.2, March 17 2026)
  failed verification and must not be cited.
- **Thin coverage.** NIXL was covered only as a transport peer, and the native
  vLLM offload baseline was covered for keying and correctness but not for
  performance, which Stage 2 will measure directly.
- **Forum-quality sources.** Two practitioner caveats came from GitHub issues
  (LMCache #2232, vllm-ascend #5044) and are weaker evidence than the primary
  docs; treat them as leads, not facts.

## 7. Primary sources

- vLLM Mooncake Store integration PR: https://github.com/vllm-project/vllm/pull/40900
- vLLM Mooncake Store blog: https://vllm.ai/blog/2026-05-06-mooncake-store
- Mooncake repo / README: https://github.com/kvcache-ai/Mooncake/blob/main/README.md
- Mooncake Store design: https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html
- Mooncake HA master issue: https://github.com/kvcache-ai/Mooncake/issues/1321
- Mooncake keying/correctness issue: https://github.com/kvcache-ai/Mooncake/issues/1408
- LMCache repo: https://github.com/LMCache/LMCache
- LMCache + Mooncake joint post: https://blog.lmcache.ai/en/2026/05/26/when-open-source-meets-open-source-a-joint-effort-between-lmcache-and-mooncake/
- LMCache metrics: https://docs.lmcache.ai/production/observability/metrics.html
- LMCache vLLM endpoint: https://docs.lmcache.ai/production/observability/vllm_endpoint.html
- vLLM Production Stack: https://blog.vllm.ai/production-stack/index.html
- FlexKV repo: https://github.com/taco-project/FlexKV
- FlexKV (NVIDIA Dynamo docs): https://docs.nvidia.com/dynamo/integrations/flex-kv
- NIXL connector (vLLM): https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector/
- vLLM prefix caching design (SHA-256 keying): https://docs.vllm.ai/en/stable/design/prefix_caching/
- vLLM keying/observability issue: https://github.com/vllm-project/vllm/issues/38474
- Benchmark paper: https://arxiv.org/pdf/2510.09665
