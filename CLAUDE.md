# CLAUDE.md

## Project

Investigate distributed KV caching solutions for LLM inference serving, then build a working prototype around the chosen solution.

The work runs in two phases:

1. **Research and report.** Survey available distributed KV cache solutions, compare them on capability, maturity, performance, and integration cost, and produce a written recommendation.
2. **Prototype.** Stand up the recommended solution against a serving engine (vLLM is the assumed target) and demonstrate cross-instance KV cache reuse with measured before/after numbers.

Do not start prototype work until the report exists and a candidate has been selected.

**Orientation: read `README.md` first, then `docs/`.** `README.md` is the current
status, the strategy, and the decision log; it is the entry point for picking up
this work on any machine. `docs/runbook.md` covers how to run anywhere and the
environment gotchas. As of the latest commit, Stages 0 through 2 are complete and
the Stage 3 Mooncake Store prototype has proven cross-instance reuse locally; the
cluster tier is next. The repository is the single source of truth, since tooling
memory does not travel with a clone.

## Background: what a KV caching solution is

During transformer inference, the attention mechanism computes a key and a value vector for every token in the context. These key/value tensors are cached so that already-processed tokens are not recomputed on each step. This store is the KV cache. Inference splits into a prefill phase, which computes the KV cache for the whole prompt at once, and a decode phase, which generates one token at a time and reuses the cached KV rather than reprocessing the prompt.

The cache is large. Its size grows with context length, layer count, and model size, and it normally lives in GPU high-bandwidth memory, which is scarce and expensive. A 100K-token context can occupy several gigabytes for a single session.

A KV caching solution is a system that manages this cache beyond a single request on a single GPU. Such systems store and reuse KV blocks, offload them from GPU memory to CPU DRAM, SSD, or remote storage when GPU memory is full, and transfer them efficiently between tiers and machines. The payoff is prefix reuse: when many requests share a common prefix, the prefill cost for that prefix is paid once and the cached KV is reused, so each request only pays for its new tokens.

Distributed KV caching extends this across multiple serving instances. Instead of each instance holding its own private cache, instances share a cluster-wide pool, usually over a fast network such as RDMA. This gives larger aggregate capacity and, critically, cross-instance cache hits, so a session can be served by any instance and still find its prefix already cached. That is the class of solution this project evaluates.

## Context

The motivating problem is agentic and multi-turn workloads, where long shared prefixes (system prompts, prior turns, tool output) are recomputed across turns and across instances. Local KV cache offload to CPU DRAM or disk hits two limits: per-instance capacity with eviction under load, and cross-instance misses when a session is routed to an instance that never computed the prefix. A distributed KV cache pool addresses both by giving instances a larger shared pool and cross-instance hits.

## Lead candidate: Mooncake

Mooncake (https://github.com/kvcache-ai/Mooncake) is an open-source KV cache transfer and distributed storage library, Apache-2.0, now part of the PyTorch ecosystem. Relevant components:

- **Transfer Engine.** Batched data transfer across DRAM, VRAM, and NVMe over TCP, RDMA (InfiniBand/RoCEv2/eRDMA/GPUDirect), NVMe-of, NVLink, and CXL. This is the core building block.
- **Mooncake Store.** Distributed pooled KV cache built on the Transfer Engine, with a master server for metadata and service discovery and clients on GPU nodes.

vLLM integrates Mooncake through the `KVConnector` interface, both for prefill/decode disaggregation (`MooncakeConnector`) and for the distributed KV cache pool via Mooncake Store, composable through `MultiConnector`. Reference write-up: https://vllm.ai/blog/2026-05-06-mooncake-store

Install path for the engine is `pip install mooncake-transfer-engine` (CUDA < 13.0) or the `-cuda13` / `-non-cuda` variants. RDMA drivers and CUDA 12.1+ are expected for realistic evaluation; TCP-only works but is not representative.

## Other solutions to cover in the report

Survey these for comparison rather than assuming Mooncake wins. Confirm current status during research; do not rely on memory:

- LMCache (Mooncake usable as a remote connector)
- FlexKV (Tencent/NVIDIA; supports distributed reuse over the Mooncake Transfer Engine)
- NIXL (supports Mooncake Transfer Engine as a backend plugin)
- SGLang HiCache (Mooncake Store as a hierarchical backend)
- Native vLLM/SGLang local offload, as the baseline to beat

## Tech stack

- Python for tooling, benchmarks, glue, and the vLLM integration layer.
- C++ for any work touching the Transfer Engine or Store internals.
- Target serving engine: vLLM.
- Hardware assumptions for benchmarks: NVIDIA GPUs, RDMA NICs, CUDA 12.1+. State the exact environment in any reported result.

## Repository layout (proposed)

Fresh repo. Suggested structure, adjust as the work takes shape:

```
docs/        research notes, comparison matrix, the final report
bench/       benchmark scripts, traces, result data
prototype/   integration code (Python + any C++)
scripts/     setup and environment helpers
```

## Build, test, and lint conventions

These are starting defaults for a fresh repo. Confirm and update this section once tooling is committed.

- **Python:** format and lint with `ruff`; test with `pytest`.
- **C++:** follow `.clang-format`; build with CMake; test with `ctest`/GoogleTest.
- Record the exact commands here as they are established so they can be run before every commit.

## Working notes

- Benchmarks must report the cache hit rate alongside throughput, TTFT, and end-to-end latency. Hit rate is the variable that drives the rest, so a result without it is not interpretable.
- Keep claims about external projects sourced. Link the PR, doc, or commit rather than paraphrasing from memory, since this space moves quickly.

## Coding Rules

1. If the repository has linting rules, always check against them before a commit.
2. Cover new or modified behavior with unit tests where applicable. Configuration, documentation, and trivial glue changes that cannot be meaningfully unit tested are exempt.
3. Run the full test suite before reporting work as complete. Report failures rather than working around them, stubbing them out, or weakening assertions to make tests pass.
4. Do not make changes beyond what was requested. Ask before large refactors, architectural changes, or modifying files unrelated to the task.
5. Read the surrounding code before writing. Follow the existing style, naming, and patterns of the file rather than introducing new ones.
6. Never commit credentials, API keys, tokens, or .env files. Do not hardcode secrets.
7. Flag before adding new dependencies or upgrading existing ones.
8. Do not commit directly to main or master, force-push shared branches, or amend or rewrite other contributors' commits.

## Contribution Rules

1. No comments, documentation, commit messages, or any output may mention Claude, Claude AI, Anthropic, or any AI assistant. All work must appear as if authored entirely by a human contributor.
2. All commits must be signed by Donald Tolley with email tolleybot@gmail.com. Never add a Co-Authored-By line or any other attribution to an AI.
3. Comments, documentation, and commit messages must follow the conventions of the repository being contributed to. Use proper grammar with commas and complete sentences. Do not use dashes as separators in prose; use commas or rewrite the sentence instead.
