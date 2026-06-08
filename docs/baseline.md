# Stage 2 Baseline: Native vLLM KV Caching

The number every distributed option must beat. This records the baseline
measurement of native vLLM prefix caching on the local development box, with
cache hit rate reported alongside latency and throughput, per the project rubric.

These are single-instance results. Cross-instance hits, the value a distributed
pool adds, cannot be measured on one GPU and are the subject of Stage 3.

## Environment

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070, 12 GB (Blackwell, sm_120) |
| Driver / CUDA | 595.58.03 / CUDA 13.2 |
| vLLM | 0.22.0 |
| PyTorch | 2.11.0+cu130 |
| Model | Qwen/Qwen2.5-3B-Instruct (bf16) |
| Sampler | greedy (temperature 0); FlashInfer sampler disabled, it cannot JIT for sm_120 |
| Transport | local only, no RDMA (development tier) |

This is the development tier from `environment-checklist.md`. Numbers are for
correctness and relative comparison, not for representative performance claims.

## Method

The trace generator (`bench/trace.py`) builds a synthetic agentic, multi-turn
workload: 16 sessions, 4 turns each, a 400-word shared system prompt, and ~40
word user and assistant messages per turn, issued round-robin so sessions
interleave. The shared-system-prompt fraction is swept across 0.0, 0.5, and 0.9
to move the hit rate. The harness (`bench/run_baseline.py`) streams each request
to the vLLM OpenAI-compatible server, measures time to first token and end to end
latency directly, and reads the prefix cache hit rate from the Prometheus
`/metrics` counters `vllm:prefix_cache_queries_total` and
`vllm:prefix_cache_hits_total` as a delta across the run. Requests use greedy
decoding with 64 output tokens. Concurrency is 1, so latency reflects per-request
cost without queueing.

Two configurations isolate the effect of cache capacity:

- **B1, ample cache.** Default sizing, 99,440-token KV cache (about 6,200 blocks,
  12x concurrency headroom). The working set fits, so reuse is near its ceiling.
- **B2, constrained cache.** `--num-gpu-blocks-override 256`, a 4,096-token KV
  cache (2x concurrency headroom), with `--max-model-len 2048`. The working set
  no longer fits, forcing eviction. This models the per-instance capacity limit
  that motivates a distributed pool.

## Results

### B1, ample cache (99,440-token KV cache)

| Shared prefix | Hit rate | TTFT mean (ms) | TTFT p50 (ms) | e2e p50 (ms) | Output tok/s |
| --- | --- | --- | --- | --- | --- |
| 0% | 68.6% | 32.1 | 22.7 | 718 | 88.2 |
| 50% | 76.5% | 26.8 | 22.4 | 717 | 88.8 |
| 90% | 90.4% | 22.1 | 20.6 | 715 | 89.4 |

### B2, constrained cache (4,096-token KV cache)

| Shared prefix | Hit rate | TTFT mean (ms) | TTFT p50 (ms) | Output tok/s |
| --- | --- | --- | --- | --- |
| 0% | 0.0% | 72.5 | 73.0 | 84.2 |
| 50% | 30.6% | 53.5 | 54.4 | 86.4 |
| 90% | 61.5% | 36.0 | 33.3 | 88.5 |

## Interpretation

1. **Hit rate tracks prefix sharing, as designed.** In B1 it rises from 68.6% to
   90.4% as the shared-prompt fraction grows. The harness measures the variable
   that drives everything else.
2. **Within-session reuse is significant on its own.** At 0% cross-session
   sharing, B1 still hits 68.6%, because each session's later turns reuse the
   growing prefix of its earlier turns. Cross-session sharing of the system
   prompt adds on top of that.
3. **Cache capacity is the lever, and it is the whole argument for a pool.**
   Shrinking the cache from 99,440 to 4,096 tokens collapses the hit rate: 0%
   sharing falls from 68.6% to 0.0%, 50% from 76.5% to 30.6%, 90% from 90.4% to
   61.5%. At 0% sharing the round-robin interleaving evicts each session's blocks
   before its next turn arrives, so nothing survives to be reused.
4. **TTFT pays for the misses.** Under the constrained cache, mean TTFT roughly
   doubles (32 to 72 ms at 0%, 27 to 53 ms at 50%, 22 to 36 ms at 90%), because
   the prefill that the cache would have skipped is recomputed.
5. **Throughput and end to end latency barely move here.** With a 3B model, 64
   output tokens, and concurrency 1, total latency is dominated by decode, not
   prefill, so the prefill savings show up in TTFT rather than tokens per second.
   On larger models, longer prompts, and higher concurrency the throughput effect
   grows; that regime is for the multi-GPU tier.

The headline for the project: the gap between B2 and B1 is the prize. A larger
shared cache recovers the evicted hits, and a distributed pool is how a cluster
gets a larger effective cache than any single instance can hold. Stage 3 measures
whether Mooncake closes that gap across instances.

## Reproduce

```bash
# 1. Generate the trace sweep
bash scripts/gen_traces.sh

# 2a. Baseline B1 (ample cache)
MAX_MODEL_LEN=8192 bash scripts/serve_baseline.sh        # in one shell
LABEL=baseline_b1 bash scripts/run_sweep.sh              # in another
bash scripts/stop_server.sh

# 2b. Baseline B2 (constrained cache, eviction pressure)
MAX_MODEL_LEN=2048 NUM_GPU_BLOCKS=256 bash scripts/serve_baseline.sh
LABEL=baseline_b2 MAX_MODEL_LEN=2048 bash scripts/run_sweep.sh
bash scripts/stop_server.sh
```

Per-run JSON is written to `bench/results/`.

## Caveats

- Development-tier hardware: a single 12 GB consumer GPU, no RDMA, a small model.
  Absolute latencies and the throughput regime are not representative of a serving
  cluster. The relative effect of cache capacity on hit rate and TTFT is the
  transferable result.
- Concurrency 1 isolates per-request cost but understates throughput gains.
  Higher concurrency is a cluster-tier measurement.
- The constrained-cache sizes (B2) are chosen to force eviction for illustration,
  not calibrated to a specific production cache-to-working-set ratio.
