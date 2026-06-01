"""Unit tests for the pure logic in the baseline harness."""

from __future__ import annotations

from bench.run_baseline import parse_prometheus_counter, percentile, summarize


def test_parse_counter_sums_across_labels():
    text = (
        "# HELP vllm:prefix_cache_hits_total Hits\n"
        "# TYPE vllm:prefix_cache_hits_total counter\n"
        'vllm:prefix_cache_hits_total{model_name="a"} 10.0\n'
        'vllm:prefix_cache_hits_total{model_name="b"} 5.0\n'
    )
    assert parse_prometheus_counter(text, "vllm:prefix_cache_hits_total") == 15.0


def test_parse_counter_absent_returns_none():
    text = 'other_metric{x="1"} 3.0\n'
    assert parse_prometheus_counter(text, "vllm:prefix_cache_hits_total") is None


def test_parse_counter_no_labels():
    text = "vllm:prefix_cache_queries_total 42.0\n"
    assert parse_prometheus_counter(text, "vllm:prefix_cache_queries_total") == 42.0


def test_parse_counter_does_not_match_external_prefix():
    # The external_prefix_cache_* family must not be mistaken for the local one.
    text = 'vllm:external_prefix_cache_hits_total{model_name="a"} 9.0\n'
    assert parse_prometheus_counter(text, "vllm:prefix_cache_hits_total") is None


def test_percentile_endpoints_and_midpoint():
    values = [0.0, 10.0]
    assert percentile(values, 0) == 0.0
    assert percentile(values, 100) == 10.0
    assert percentile(values, 50) == 5.0


def test_percentile_single_value():
    assert percentile([7.0], 95) == 7.0


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_summarize_converts_to_ms_keys():
    out = summarize("ttft", [0.1, 0.2, 0.3])
    assert set(out) == {"ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99", "ttft_ms_mean"}
    assert abs(out["ttft_ms_mean"] - 200.0) < 1e-6
