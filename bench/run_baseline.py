"""Baseline benchmark harness for native vLLM KV caching.

Drives a running vLLM OpenAI-compatible server with a trace produced by
``bench/trace.py``, streaming each request so time-to-first-token is measured
directly. The prefix cache hit rate is read from the server's Prometheus
``/metrics`` endpoint, taken as the delta across the run so it reflects only this
benchmark's requests.

This harness is engine-agnostic on purpose: it speaks HTTP, so the same code
measures the native baseline now and any KVConnector-backed configuration later.
Hit rate is reported alongside latency and throughput, because per the project
rubric a result without it is not interpretable.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

# Prometheus counter names exposed by vLLM v1 for prefix caching (verified
# against vLLM 0.22.0 /metrics). Summed across all label sets, since a
# single-model server still tags samples with engine and model_name labels.
# vLLM also exposes vllm:external_prefix_cache_* for connector-backed caches,
# which is where cross-instance hits will surface in the Stage 3 prototype.
_QUERIES_METRIC = "vllm:prefix_cache_queries_total"
_HITS_METRIC = "vllm:prefix_cache_hits_total"


@dataclass
class RequestResult:
    request_id: str
    ttft_s: float
    e2e_s: float
    output_tokens: int
    ok: bool


def parse_prometheus_counter(text: str, metric: str) -> float | None:
    """Sum all samples of a Prometheus counter across label sets.

    Returns None if the metric is absent, so the caller can distinguish "zero
    hits" from "this vLLM build does not export the metric".
    """
    total = None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name == metric:
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            total = value if total is None else total + value
    return total


def scrape_cache_counters(metrics_url: str) -> dict[str, float | None]:
    text = httpx.get(metrics_url, timeout=10.0).text
    return {
        "queries": parse_prometheus_counter(text, _QUERIES_METRIC),
        "hits": parse_prometheus_counter(text, _HITS_METRIC),
    }


def stream_one(client: httpx.Client, base_url: str, model: str, prompt: str,
               max_tokens: int, request_id: str) -> RequestResult:
    """Send one streaming completion and time first token and total."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    output_tokens = 0
    ok = True
    try:
        with client.stream("POST", f"{base_url}/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    output_tokens = usage["completion_tokens"]
    except (httpx.HTTPError, json.JSONDecodeError):
        ok = False
    e2e = time.perf_counter() - start
    return RequestResult(
        request_id=request_id,
        ttft_s=ttft if ttft is not None else e2e,
        e2e_s=e2e,
        output_tokens=output_tokens,
        ok=ok,
    )


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; ``pct`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def summarize(name: str, values_s: list[float]) -> dict[str, float]:
    ms = [v * 1000.0 for v in values_s]
    return {
        f"{name}_ms_p50": percentile(ms, 50),
        f"{name}_ms_p95": percentile(ms, 95),
        f"{name}_ms_p99": percentile(ms, 99),
        f"{name}_ms_mean": sum(ms) / len(ms) if ms else 0.0,
    }


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(trace: list[dict], base_url: str, metrics_url: str, model: str,
        concurrency: int, warmup: bool) -> dict:
    client = httpx.Client(timeout=httpx.Timeout(300.0))

    if warmup:
        stream_one(client, base_url, model, "warmup", 4, "warmup")

    before = scrape_cache_counters(metrics_url)

    wall_start = time.perf_counter()
    if concurrency <= 1:
        results = [
            stream_one(client, base_url, model, r["prompt"], r["max_tokens"], r["request_id"])
            for r in trace
        ]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(stream_one, client, base_url, model,
                            r["prompt"], r["max_tokens"], r["request_id"])
                for r in trace
            ]
            results = [f.result() for f in futures]
    wall = time.perf_counter() - wall_start

    after = scrape_cache_counters(metrics_url)
    client.close()

    ok_results = [r for r in results if r.ok]
    total_out = sum(r.output_tokens for r in ok_results)

    hit_rate: float | None = None
    if before["queries"] is not None and after["queries"] is not None:
        dq = after["queries"] - before["queries"]
        dh = (after["hits"] or 0.0) - (before["hits"] or 0.0)
        hit_rate = (dh / dq) if dq > 0 else 0.0

    metrics = {
        "requests": len(results),
        "requests_ok": len(ok_results),
        "prefix_cache_hit_rate": hit_rate,
        "prefix_cache_queries": (
            None if before["queries"] is None else after["queries"] - before["queries"]
        ),
        "wall_time_s": wall,
        "throughput_req_per_s": len(ok_results) / wall if wall > 0 else 0.0,
        "throughput_output_tok_per_s": total_out / wall if wall > 0 else 0.0,
        "output_tokens_total": total_out,
    }
    metrics.update(summarize("ttft", [r.ttft_s for r in ok_results]))
    metrics.update(summarize("e2e", [r.e2e_s for r in ok_results]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vLLM KV cache benchmark")
    parser.add_argument("--trace", required=True, help="Trace JSONL from bench/trace.py")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--label", default="baseline", help="Run label recorded in output")
    parser.add_argument("--out", help="Path to write results JSON")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    metrics_url = f"{base_url}/metrics"
    trace = load_trace(args.trace)

    metrics = run(
        trace=trace,
        base_url=base_url,
        metrics_url=metrics_url,
        model=args.model,
        concurrency=args.concurrency,
        warmup=not args.no_warmup,
    )
    output = {
        "label": args.label,
        "config": {
            "trace": args.trace,
            "model": args.model,
            "concurrency": args.concurrency,
            "requests": len(trace),
        },
        "metrics": metrics,
    }

    if metrics["prefix_cache_hit_rate"] is None:
        print("WARNING: prefix cache metrics not found at /metrics; hit rate unavailable")

    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.out:
        with open(args.out, "w") as f:
            f.write(rendered + "\n")
        print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
