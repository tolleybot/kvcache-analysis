"""Cross-instance KV reuse test for the Mooncake Store prototype (Stage 3).

The claim under test: a prefix computed by instance A and written to the shared
Mooncake Store can be reused by instance B without recomputation. This is the
distributed value that single-instance prefix caching cannot provide.

Procedure:
  1. Send every trace prompt to instance A only. A computes the prefixes and
     writes the KV blocks to the Store.
  2. After a short settle, send the same prompts to instance B. B has never seen
     them locally, so any reuse must come from the Store.
  3. Read instance B's external prefix cache counters. A non-zero external hit
     rate is the proof of cross-instance reuse.

External hits are reported separately from local hits because vLLM exposes
vllm:external_prefix_cache_* for connector-backed caches, which is exactly the
cross-instance path. A run with no Mooncake connector would show zero external
queries, which is the control.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from bench.run_baseline import (
    load_trace,
    parse_prometheus_counter,
    stream_one,
    summarize,
)

_LOCAL_QUERIES = "vllm:prefix_cache_queries_total"
_LOCAL_HITS = "vllm:prefix_cache_hits_total"
_EXT_QUERIES = "vllm:external_prefix_cache_queries_total"
_EXT_HITS = "vllm:external_prefix_cache_hits_total"


def scrape(metrics_url: str) -> dict[str, float | None]:
    text = httpx.get(metrics_url, timeout=10.0).text
    return {
        "local_queries": parse_prometheus_counter(text, _LOCAL_QUERIES),
        "local_hits": parse_prometheus_counter(text, _LOCAL_HITS),
        "ext_queries": parse_prometheus_counter(text, _EXT_QUERIES),
        "ext_hits": parse_prometheus_counter(text, _EXT_HITS),
    }


def hit_rate(before: dict, after: dict, q_key: str, h_key: str) -> float | None:
    if before[q_key] is None or after[q_key] is None:
        return None
    dq = after[q_key] - before[q_key]
    dh = (after[h_key] or 0.0) - (before[h_key] or 0.0)
    return (dh / dq) if dq > 0 else 0.0


def send_all(base_url: str, model: str, trace: list[dict], tag: str) -> list:
    client = httpx.Client(timeout=httpx.Timeout(300.0))
    results = [
        stream_one(
            client, base_url, model, r["prompt"], r["max_tokens"], f"{tag}-{r['request_id']}"
        )
        for r in trace
    ]
    client.close()
    return results


def run(trace: list[dict], model: str, a_url: str, b_url: str, b_metrics: str,
        settle_s: float) -> dict:
    # Phase 1: populate the shared store from instance A.
    a_results = send_all(a_url, model, trace, "A")

    # Let asynchronous store writes settle before reading from B.
    time.sleep(settle_s)

    # Phase 2: serve the same prompts from instance B and measure its reuse.
    before = scrape(b_metrics)
    b_results = send_all(b_url, model, trace, "B")
    after = scrape(b_metrics)

    b_ok = [r for r in b_results if r.ok]
    metrics = {
        "requests": len(trace),
        "a_requests_ok": sum(1 for r in a_results if r.ok),
        "b_requests_ok": len(b_ok),
        "b_external_hit_rate": hit_rate(before, after, "ext_queries", "ext_hits"),
        "b_local_hit_rate": hit_rate(before, after, "local_queries", "local_hits"),
        "b_external_queries": (
            None if before["ext_queries"] is None
            else after["ext_queries"] - before["ext_queries"]
        ),
        "settle_s": settle_s,
    }
    metrics.update(summarize("b_ttft", [r.ttft_s for r in b_ok]))
    metrics.update(summarize("b_e2e", [r.e2e_s for r in b_ok]))

    if before["ext_queries"] is None:
        metrics["note"] = (
            "vllm:external_prefix_cache_* absent on instance B; the Mooncake "
            "connector is likely not active. This is the no-connector control."
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-instance KV reuse test (Mooncake Store)")
    parser.add_argument("--trace", required=True, help="Trace JSONL from bench/trace.py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host-a", default="localhost")
    parser.add_argument("--port-a", type=int, default=8000)
    parser.add_argument("--host-b", default="localhost")
    parser.add_argument("--port-b", type=int, default=8001)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--out", help="Path to write results JSON")
    args = parser.parse_args()

    a_url = f"http://{args.host_a}:{args.port_a}"
    b_url = f"http://{args.host_b}:{args.port_b}"
    b_metrics = f"{b_url}/metrics"

    metrics = run(
        trace=load_trace(args.trace),
        model=args.model,
        a_url=a_url,
        b_url=b_url,
        b_metrics=b_metrics,
        settle_s=args.settle_s,
    )
    output = {
        "label": "xinstance",
        "config": {"trace": args.trace, "model": args.model, "a": a_url, "b": b_url},
        "metrics": metrics,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)

    ext = metrics["b_external_hit_rate"]
    if ext is None:
        print("\nRESULT: external cache metrics absent (no-connector control).")
    elif ext > 0:
        print(f"\nRESULT: cross-instance reuse CONFIRMED, B external hit rate = {ext:.1%}")
    else:
        print("\nRESULT: no cross-instance hits (external hit rate 0); check store wiring.")

    if args.out:
        with open(args.out, "w") as f:
            f.write(rendered + "\n")
        print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()
