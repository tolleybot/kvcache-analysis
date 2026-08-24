# SGLang Radix Caching — Research Design

**Status:** scoped 2026-08-24, not started.
**Owner:** Donald Tolley. **Asked by:** Adam Reeve. **Reporting to:** Adam, then Jonathan.

Companion to the GR-1331 distributed KV caching report (`docs/report.md`), which measured
the vLLM + Mooncake side of the same design axis. Read `README.md` for the parent project's
background on KV caching and the Mooncake stack; this document does not repeat it.

This is the research design: what question is being asked, what would count as an answer,
and how the effort is bounded. Findings go in a separate writeup.

## Goal

**Determine whether SGLang's radix-tree prefix cache can be shared across instances, and
what that reveals about the tradeoff between match granularity and distributability.**

Done when we can say with evidence which of three outcomes holds:

- **(a) Distributes, keeps the advantage.** Radix beats hash blocks on hit rate under
  partial overlap *and* survives pooling. Significant, because GR-1331's engine choice deserves
  revisiting.
- **(b) Distributes, loses the advantage.** Sharing forces coarser granularity or a
  content-addressed lower tier, collapsing it toward what vLLM already does. The
  crossover result, and the one to bet on.
- **(c) Does not distribute.** Structural reason, stated precisely. Confirms GR-1331's
  vLLM-only decision was right, and explains *why* rather than by luck.

Every outcome is reportable. There is no incentive to stretch for a positive, and this is
also the stop condition: once one of the three is evidenced, the work is done.

## Origin

Adam Reeve, Slack, ~2026-08-19, replying to a note that SGLang's prefix caching uses a
radix tree rather than vLLM's hash blocks:

> That point about SGLang using radix tree caching is interesting, could be worth looking
> into that and whether it can be used in a distributed way.

## Framing

This is a **research project**, not a Stage 5 of GR-1331 and not a procurement evaluation.

An evaluation asks "should we adopt this?" and a negative answer ends it. Research asks
"how does this class of design behave, and why?" and a negative answer is a result.

- **A "no" is an output, not an exit.** If radix trees resist distribution, the deliverable
  is *why*, stated precisely enough to apply to the next system.
- **Feeding GR-1331 is a byproduct.** Anything learned about granularity and hit rate
  applies to the Mooncake work, and Jonathan gets a report. A benefit, not the justification.
- **Research framing changes what counts as a result, not how long it runs.** The stages
  and the hardware gate still bound it.

## The central idea

This is the reason the question is non-trivial. Keep it in view.

**vLLM's prefix cache is content-addressed.** A block is identified by the hash of the
tokens it contains, so any node can hold any block and a lookup is a hash probe. That
property is *why* Mooncake Store worked so well in GR-1331, since pooling content-addressed
blocks across machines is nearly free. Measured: 98% cross-instance reuse, with
cross-machine RDMA adding no meaningful overhead over single-machine.

**A radix tree is structural.** Entries are nodes in an ordered tree with parent-child
dependencies; a lookup is a walk from the root. Sharing that across machines means
replicating a mutable tree, sharding a structure whose value *is* its connectedness, or
decomposing into a distributed index over a content-addressed store, at which point ask
whether the radix layer still earns anything.

**Working hypothesis: radix wins on hit rate, loses on distributability.** The useful
output is therefore not a winner but where the crossover sits.

**Prior evidence this axis is real.** In GR-1331, with the *same* Mooncake backend
underneath, LMCache reused only 53% against bare Mooncake's 98%, purely because its
256-token chunking was too coarse to match partial prefixes. **Granularity dominated
transport.** Radix matching is the fine-grained end of that same axis.

## Stages

Effort is deliberately weighted toward the architectural question. It is what decides (b)
versus (c), and it is the half Adam actually asked about.

### Stage 1 — Architecture read. The main event. (~1.5 days, no hardware)

Outcomes (b) and (c) are both settled by reading, not benchmarking.

- How RadixAttention's prefix cache is structured; what eviction does to shared prefixes.
- **Whether HiCache's remote tier preserves radix matching or degrades to block/chunk
  granularity at the boundary.** This is the crux. If the shared tier is content-addressed
  underneath, the radix advantage stops at the node edge and the answer is (b).
- What granularity HiCache uses when targeting Mooncake Store. Note the GR-1331 survey
  already records SGLang HiCache as supporting Mooncake Store as a hierarchical backend,
  so the open question is not *whether* a remote tier exists but *what it does to matching*.
- Whether the tree is per-instance by construction, or has an upstream cross-instance story.

**Output:** a written architectural answer naming (a), (b), or (c), with the mechanism.
If it lands on (b) or (c) with confidence, Stage 3 does not run.

### Stage 2 — Targeted hit-rate measurement. (~1 day, single node)

Narrow by design. Sizes the granularity advantage. **Not a bake-off.**

- One question: how much better is radix matching than hash blocks **under partial prefix
  overlap**? Equal-prefix workloads do not separate the two designs and are not worth running.
- Reuse the 0/50/90% shared-prefix workloads from `docs/baseline.md` so numbers sit directly
  alongside what we already reported.
- **Hit rate is the metric. TTFT is context only.** The engines differ in scheduling,
  batching, and attention kernels, so a raw latency delta does not isolate the cache.
- This number is what makes (b) meaningful: it quantifies exactly what is lost at the boundary.

### Stage 3 — Distributed prototype. Conditional. (~3 days)

Runs **only** if Stage 1 points to (a), or leaves (b) genuinely ambiguous in a way that
measurement resolves.

- Two SGLang instances sharing a cache tier, same shape as the GR-1331 Stage 3 test.
- Same transports where applicable: TCP as control, RDMA single-machine, RDMA cross-machine.
- Same comparison table format, so rows read directly against the vLLM ones.

## What carries over from GR-1331

The measurement half is already built. This is why the scope is small.

| Asset | Where | Reuse |
|---|---|---|
| Evaluation rubric | `docs/evaluation-rubric.md` | As-is |
| Benchmark harness and methodology | `bench/`, Stages 2–3 | As-is, new engine behind it |
| Baseline, native vLLM caching | `docs/baseline.md` | Direct comparison target |
| Environment gotchas | `docs/runbook.md`, `docs/environment-checklist.md` | Saves a day |
| Mooncake Store deployment | `prototype/` | Reuse if HiCache can target it |

**Hardware** (unchanged from GR-1331):

- A100 box: `192.168.147.151` (`latpoc51`), 8× A100-SXM4-80GB, SM80
- H200 pair: `latpoc32` (`192.168.147.132`) / `latpoc34` (`192.168.147.133`), 4 IB links
  (`mlx5_2`, `mlx5_3`, `mlx5_4`, `mlx5_7`)

**Models**: Qwen2.5-3B, Qwen2.5-32B, GLM-5.2-FP8. Use the same ones so results compare.

**Known gotchas that will bite again:**

- `PYTHONHASHSEED` must be identical across instances, or cross-instance hits stay at zero
  with no error. Verify the SGLang equivalent early.
- CUDA build must match the host driver, or imports fail on `libcudart.so.NN`.

## Out of scope

- Migrating any GResearch workload to SGLang. This is an evaluation, not an adoption proposal.
- The 32K/64K/100K sweep and mixed-size concurrency test offered to James. Those belong to
  GR-1331 and reopen that file if he takes them up.
- SGLang correctness, throughput, or general feature comparison. The AI Stack Challenge
  covers the serving-engine experience separately.
- Anything needing new hardware. If it does not run on the A100 box or the H200 pair, it is out.

## Deliverable

A standalone writeup in `docs/`, sitting alongside the GR-1331 report rather than inside it:

1. **The design axis, stated plainly.** Content-addressed versus structural prefix caching:
   what each buys, what each costs, why distributability and match granularity pull opposite.
2. **The architectural answer.** Can radix caching be distributed, by what mechanism, at what cost.
3. **Head-to-head hit rate** on workloads where granularity is actually visible.
4. **Where the crossover sits**, not a winner.

**Reporting:** Adam first, then Jonathan as the KV cache stakeholder. Jonathan's group already
has the GR-1331 report, so this reads as a follow-on chapter on the same axis, not a new topic.
