# Distributed KV Caching for vLLM

Investigating distributed KV caching solutions for LLM inference serving, then
building a working prototype around the chosen solution. The motivating workload
is agentic and multi-turn serving, where long shared prefixes (system prompts,
prior turns, tool output) are recomputed across turns and across instances. A
distributed KV cache pool gives instances a larger shared cache and, critically,
cross-instance cache hits.

This README is the orientation point. It captures the strategy, the decisions and
their rationale, and the current status, so anyone (or any tooling) picking up
the work on a fresh clone has the full picture without the original conversation.
Detailed material lives in `docs/`; this file links into it.

## Status at a glance

| Stage | What | State |
| --- | --- | --- |
| 0 | Evaluation rubric and success definition | Done (`docs/evaluation-rubric.md`) |
| 0 | Environment confirmation checklist (RDMA gate) | Done (`docs/environment-checklist.md`) |
| 1 | Sourced survey and comparison of candidates | Done (`docs/survey.md`) |
| 2 | Baseline benchmark of native vLLM caching | Done (`docs/baseline.md`) |
| 3 | Mooncake Store prototype, cross-instance reuse | Proven locally (`docs/stage3.md`) |
| 3+ | Cluster tier: RDMA, multi-GPU, representative numbers | Next |

## Strategy

The work is organized as a decision pipeline with go/no-go gates, rather than
jumping straight to benchmarking. Each stage produces an artifact and a decision.

1. **Frame the decision (Stage 0).** Write the evaluation rubric first, so the
   comparison is honest. Operability and reliability are weighted above raw
   performance, because the target is enterprise infrastructure. There are hard
   gates, the most important being a correctness invariant: reused KV must always
   match the exact prefix it claims to be.
2. **Survey and choose (Stage 1).** Score candidates against the rubric and pick
   what to prototype.
3. **Measure the baseline (Stage 2).** Build the benchmark harness and measure
   native vLLM, the number any distributed option must beat. Hit rate is always
   reported alongside latency and throughput, because it is the variable that
   drives the rest.
4. **Prototype (Stage 3).** Stand up the chosen solution and prove cross-instance
   reuse with measured numbers.

Hardware strategy is local-first: prove correctness and integration cheaply on a
local box, then move to the cluster only for representative performance. The
project is containerized so the same setup runs on any machine; see
`docs/runbook.md`.

## Decisions and rationale

- **Serving engine: vLLM only.** SGLang and its HiCache backend are comparison
  context, not integration targets. The work centers on vLLM's `KVConnector`
  path.
- **Priorities: operability and reliability over raw performance.** In the rubric
  these two carry half the weighted score. A faster but fragile or hard to run
  solution loses to a dependable, operable one.
- **The field splits into two layers.** A transport layer (Mooncake Transfer
  Engine, NIXL) and a cache-management layer (Mooncake Store, LMCache, FlexKV).
  The cache-management layers all sit on top of the Transfer Engine, so the real
  choice is which cache-management layer to adopt, not Mooncake versus the rest.
- **Lead candidate and the three-way fork.** The prototype options were (a)
  Transfer Engine only beneath our own layer, (b) full Mooncake Store via the
  vLLM `KVConnector`, and (c) Mooncake Store fronted by LMCache. Option (a) was
  dropped as too much custom work. The plan is (b) first for its directness and
  clean attribution, then (c) on the cluster where LMCache's operability and
  graceful degradation are worth evaluating. Details and sources in
  `docs/survey.md`.
- **Reproducibility via Docker, repo as the source of truth.** The environment is
  parameterized in a single image so it runs across the local box and cluster
  hardware. Because tooling memory does not travel with a clone, durable
  environment facts and run instructions live in the repo (`docs/runbook.md`),
  not anywhere machine-local.

## Results so far

- **Baseline (Stage 2), native vLLM on Qwen2.5-3B.** Hit rate rises with prefix
  sharing (68.6% to 90.4% across 0/50/90% shared) when the cache is ample. Under
  a constrained per-instance cache it collapses (0%, 30.6%, 61.5%) and time to
  first token roughly doubles. That gap is the motivation for a distributed pool.
  Full numbers in `docs/baseline.md`.
- **Prototype (Stage 3), Mooncake Store.** Two vLLM instances sharing one Store
  pool achieved a 96.7% cross-instance hit rate on instance B for prefixes
  computed by instance A, over TCP on a single GPU. This is a correctness proof
  of the mechanism, not a performance result. Full detail in `docs/stage3.md`.

## Two findings that gate distributed reuse

Both are documented in `docs/runbook.md` and baked into the scripts, but they are
the kind of thing worth knowing up front:

- **`PYTHONHASHSEED` must be identical across all instances.** vLLM seeds its
  block hashes per process otherwise, so instances never match in the shared
  store and cross-instance hits stay at zero with no error.
- **CUDA build must match the host driver** for both vLLM and Mooncake. The wrong
  build fails to import with a `libcudart.so.NN` error. The image is parameterized
  for this; the driver to CUDA matrix is in the runbook.

## How to run

See `docs/runbook.md` for the full quick start (Docker and host virtualenv), the
hardware portability matrix, and the Stage 3 multi-instance instructions. In
short:

```bash
docker build -f docker/Dockerfile -t mloss-vllm-kvcache:latest .
bash docker/run-server.sh        # baseline server
bash docker/run-bench.sh         # baseline sweep
```

## Repository map

- `docs/evaluation-rubric.md` the Stage 0 success definition and weighted rubric
- `docs/environment-checklist.md` RDMA fabric questions and diagnostic commands
- `docs/survey.md` the sourced candidate comparison and recommendation
- `docs/baseline.md` Stage 2 baseline results and method
- `docs/stage3.md` the Mooncake Store prototype, recipe, and result
- `docs/runbook.md` how to run anywhere, the portable environment notes
- `bench/` trace generator and benchmark harnesses, with unit tests
- `scripts/` host-side serve, generate, sweep, stop, and Mooncake launch
- `docker/` Dockerfile, run wrappers, and the multi-GPU Compose topology
- `CLAUDE.md` project conventions and contribution rules

## Continuing the work

The next step is the cluster tier: run on at least two GPUs with one instance per
GPU and a real model over RDMA, to get representative time to first token and
throughput against the Stage 2 baseline, then exercise the reliability gates
(master or peer failure degrading to recompute). The prototype is ready for this:
`git clone`, then build with `--build-arg INSTALL_MOONCAKE=1` and bring up
`docker/compose.mooncake.yml`. Confirm the RDMA fabric first using
`docs/environment-checklist.md`.
