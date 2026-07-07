# Follow-up work from the results brief review

Open items arising from review of `results-brief.md`. Several are gated on input
from the requesting team about their target workload and the metrics they care
about, so they are grouped by that dependency.

## Highest value

1. **Throughput under concurrent load.** The brief measures time to first token at
   concurrency 1, which isolates per-request cost. The next question is throughput
   and capacity at realistic concurrency, where prefill savings convert into
   sustained tokens per second rather than a single-stream latency win. This is the
   most useful follow-up measurement.

## Gated on workload confirmation

2. **Confirm the target workload.** Shared prefixes (agentic and multi-turn: system
   prompts, prior turns, tool output) versus repeated spans that recur mid-prompt
   (retrieval-augmented generation). The answer drives both the metric choice in
   item 1 and whether item 3 is worth doing. The working assumption is that shared
   prefixes dominate for the agentic target.

3. **Non-prefix, repeated-span reuse test (conditional on item 2).** Only if
   repeated spans turn out to matter. LMCache can reuse KV for any repeated span,
   not only a shared prefix, which prefix-only caching misses in a retrieval-heavy
   workload. Likely low priority given the agentic target.

## Independent

4. **Write-overhead control run (optional, low priority).** A dedicated run with the
   store write disabled would quantify how much instance A's pool write adds to its
   cold time to first token. It is mostly academic on the recommended transport,
   where the write is a few milliseconds over RDMA, so it is worth doing only if a
   precise number is wanted.

5. **File the LMCache cross-node defect upstream.** No existing issue matches the
   exact symptom (on a node remote from the master, the Mooncake connector fails to
   create its store client, then segfaults on first use). The closest is LMCache
   issue 2232, which documents `MooncakeLookupClient` hardcoding `localhost` and
   `tcp` instead of reading them from config, a plausible root cause. Raise a new
   issue with the reproduction on LMCache 0.4.5 and 0.5.0.
