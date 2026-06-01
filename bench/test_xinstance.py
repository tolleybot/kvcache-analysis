"""Unit tests for the cross-instance test's pure logic."""

from __future__ import annotations

from bench.run_xinstance import hit_rate


def _counters(local_q, local_h, ext_q, ext_h):
    return {
        "local_queries": local_q,
        "local_hits": local_h,
        "ext_queries": ext_q,
        "ext_hits": ext_h,
    }


def test_external_hit_rate_delta():
    before = _counters(0, 0, 0, 0)
    after = _counters(0, 0, 10, 8)
    assert hit_rate(before, after, "ext_queries", "ext_hits") == 0.8


def test_hit_rate_none_when_metric_absent():
    before = _counters(0, 0, None, None)
    after = _counters(0, 0, None, None)
    assert hit_rate(before, after, "ext_queries", "ext_hits") is None


def test_hit_rate_zero_queries_is_zero_not_error():
    before = _counters(0, 0, 5, 5)
    after = _counters(0, 0, 5, 5)
    assert hit_rate(before, after, "ext_queries", "ext_hits") == 0.0


def test_hit_rate_uses_delta_not_absolute():
    # Pre-existing hits before the window must not count toward this run.
    before = _counters(0, 0, 100, 100)
    after = _counters(0, 0, 110, 104)
    assert hit_rate(before, after, "ext_queries", "ext_hits") == 0.4
