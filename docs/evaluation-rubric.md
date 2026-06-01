# Evaluation Rubric and Success Definition

Stage 0 of the project. This document defines what we are deciding, the workload
we measure against, the metrics that count, and the weighted criteria used to
score candidate solutions. Every later stage, the survey, the benchmark harness,
the prototype, and the final recommendation, is measured against what is written
here. If a criterion or target changes, it changes in this file first.

## 1. The decision

Choose a distributed KV caching approach for vLLM serving in an enterprise
setting, then prove it with measured before and after numbers. The output of the
evaluation is one of three architectural choices, plus a recommendation on
whether to adopt at all:

1. Mooncake **Transfer Engine** only, used as the transport beneath another
   cache layer.
2. Mooncake **Store**, the full pooled cache, integrated through the vLLM
   `KVConnector` interface.
3. Mooncake **Store via LMCache**, where LMCache is the cache-management layer
   and Mooncake is the remote backend.

The honest fourth option is "stay on native vLLM local offload," which is the
baseline every distributed option must beat by enough to justify its operational
and reliability cost.

## 2. Fixed context for this evaluation

- **Serving engine:** vLLM only. SGLang and its HiCache backend are surveyed for
  comparison but are not integration targets.
- **Workload:** agentic and multi-turn serving, where long prefixes (system
  prompts, prior turns, tool output) are shared across turns and across
  instances and are otherwise recomputed.
- **Priority order for "practical in enterprise":** operability and reliability
  are weighted above raw performance. A solution that is faster but fragile or
  hard to run loses to one that is slightly slower but dependable and easy to
  operate.

## 3. Environment

- **Development loop:** local RTX workstation. Single node, used for integration
  correctness and mechanical proof of cross-instance hits. Transport may be TCP
  here. Numbers from this tier are not representative and are labelled as such.
- **Benchmark tier:** GResearch multi-GPU clusters (V100, H200, and similar).
  This tier produces report-grade numbers.
- **To confirm before Stage 3 benchmarking:** whether the benchmark cluster
  exposes an RDMA fabric (InfiniBand or RoCE) and GPUDirect. TCP-only results are
  not representative of the design and must be flagged if that is all we have.
  State the exact environment, GPU model, NIC, driver, CUDA version, and vLLM
  version, in every reported result. The questions to ask and the commands to run
  for this confirmation live in `environment-checklist.md`.

## 4. Workload and trace definition

Results are only comparable if the workload is fixed. We define a representative
trace before benchmarking and reuse it unchanged across the baseline and every
candidate.

- **Shape:** multi-turn sessions with a shared system prompt and growing
  conversation history, plus a tail of unique per-request tokens.
- **Prefix sharing:** controlled so we can sweep the shared-prefix fraction, for
  example 0 percent, 50 percent, and 90 percent shared, since hit rate is the
  variable that drives everything else.
- **Routing:** sessions are deliberately routed to instances that did not compute
  the prefix, so that cross-instance hits are exercised rather than only
  same-instance reuse.
- **Concurrency:** swept to find the point where GPU memory pressure forces
  eviction, which is where a distributed pool is supposed to help.

## 5. Metrics

Every benchmark reports the hit rate alongside the performance metrics. A result
without a hit rate is not interpretable, because it cannot be told apart from a
faster GPU.

Primary metrics:

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Cache hit rate | Fraction of prompt tokens served from cached KV rather than recomputed | The driver. Sets the ceiling on every other gain. |
| Cross-instance hit rate | Hits served from another instance's contribution to the pool | Isolates the distributed value over local offload. |
| TTFT | Time to first token | The metric prefill reuse most directly improves. |
| Throughput | Sustained requests or tokens per second at a fixed latency SLO | Capacity gain. |
| End to end latency | p50, p95, p99 | Tail behavior under load. |

Each candidate is reported against the baseline as a delta on these, at matched
hit rate where possible, so we separate "the cache works" from "this
implementation is efficient."

## 6. Scoring rubric

Candidates are scored 1 to 5 on each dimension, then weighted. Weights reflect
the operability and reliability priority. Weights are tunable but changing them
is a recorded decision.

| Dimension | Weight | What a 5 looks like |
| --- | --- | --- |
| Operability | 25 | Deploys on existing infra with minimal moving parts, k8s friendly, sane upgrade and rollout story, modest dependency footprint, clear config. |
| Reliability | 25 | Degrades to recompute on cache or node loss, survives master and network-partition failures, predictable backpressure under load, no correctness risk. |
| Performance | 20 | Large, consistent TTFT and throughput gains at realistic hit rates, low transfer overhead, scales with cluster size. |
| Observability | 10 | Exposes hit rate, transfer latency, pool occupancy, and eviction metrics out of the box. |
| Integration cost | 10 | Works through the vLLM `KVConnector` path with little custom glue, composes via `MultiConnector`. |
| Maturity and support | 10 | Production use, active maintenance, real documentation, responsive upstream. |

Operability and reliability together carry half the score by design.

### Observability signals to require

Operability is unscoreable without telemetry, so a candidate must surface at
least: cache hit rate, cross-instance hit rate, KV transfer latency and
bandwidth, pool occupancy and eviction counts, and master or metadata service
health.

## 7. Hard gates

These are pass or fail. A failure here disqualifies a candidate regardless of
its weighted score.

- **Correctness invariant:** reused KV must always correspond to the exact prefix
  it is attributed to. A content-address or hash collision that returns the wrong
  KV for a different prefix is a silent correctness failure and is not
  acceptable. We test this explicitly.
- **Safe degradation:** loss of the cache, the pool, or the metadata service must
  fall back to recomputation and never to a wrong or failed answer.
- **No secret leakage across tenants:** cache keys and stored blocks must not let
  one session read another's content. This is a baseline check even though full
  multi-tenancy hardening is out of scope for the prototype.

## 8. Success definition

The evaluation succeeds when we can state, with sourced claims and reproducible
numbers from the benchmark tier:

1. The measured hit rate, TTFT, throughput, and latency for the baseline and for
   the chosen candidate on the fixed trace.
2. A weighted rubric score for each surveyed candidate.
3. A clear recommendation: which of the three Mooncake configurations to adopt,
   or to stay on native vLLM offload, with the operability and reliability
   reasoning made explicit.
4. An honest account of operational cost, failure modes, and the hardware the
   result depends on.

## 9. Decision gates between stages

- **After Stage 1 (survey):** select the candidate or candidates to prototype.
- **After Stage 2 (baseline):** confirm the harness is sound and the baseline
  numbers are stable and reproducible before introducing any distributed layer.
- **After Stage 3 (prototype):** confirm cross-instance hits work and produce
  representative numbers before writing the final recommendation.
