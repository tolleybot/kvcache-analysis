"""Unit tests for the synthetic trace generator."""

from __future__ import annotations

import pytest

from bench.trace import build_trace


def test_request_count_matches_sessions_times_turns():
    trace = build_trace(num_sessions=8, turns_per_session=3)
    assert len(trace) == 8 * 3


def test_deterministic_for_fixed_seed():
    a = build_trace(num_sessions=5, turns_per_session=3, seed=42)
    b = build_trace(num_sessions=5, turns_per_session=3, seed=42)
    assert [r.prompt for r in a] == [r.prompt for r in b]


def test_seed_changes_content():
    a = build_trace(num_sessions=5, turns_per_session=3, seed=1)
    b = build_trace(num_sessions=5, turns_per_session=3, seed=2)
    assert [r.prompt for r in a] != [r.prompt for r in b]


def test_within_session_prompts_grow_as_prefixes():
    # Turn t's prompt must contain turn t-1's prompt as a literal prefix, which
    # is the property the KV cache exploits for within-session reuse.
    trace = build_trace(num_sessions=2, turns_per_session=4, order="by_session")
    by_session: dict[int, list] = {}
    for r in trace:
        by_session.setdefault(r.session_id, []).append(r)
    for reqs in by_session.values():
        reqs.sort(key=lambda r: r.turn_index)
        for prev, cur in zip(reqs, reqs[1:], strict=False):
            assert cur.prompt.startswith(prev.prompt)


def test_full_sharing_gives_common_system_prefix():
    trace = build_trace(num_sessions=6, turns_per_session=1, shared_system_fraction=1.0)
    first_turn = [r.prompt for r in trace if r.turn_index == 0]
    common_prefix = first_turn[0].split("\nUser:")[0]
    assert all(p.startswith(common_prefix) for p in first_turn)


def test_zero_sharing_gives_distinct_system_prefixes():
    trace = build_trace(num_sessions=6, turns_per_session=1, shared_system_fraction=0.0)
    systems = [r.prompt.split("\nUser:")[0] for r in trace if r.turn_index == 0]
    assert len(set(systems)) == len(systems)


def test_round_robin_interleaves_sessions():
    trace = build_trace(num_sessions=3, turns_per_session=2, order="round_robin")
    # First three requests should be turn 0 of sessions 0, 1, 2.
    assert [(r.session_id, r.turn_index) for r in trace[:3]] == [(0, 0), (1, 0), (2, 0)]


def test_invalid_shared_fraction_rejected():
    with pytest.raises(ValueError):
        build_trace(shared_system_fraction=1.5)


def test_invalid_order_rejected():
    with pytest.raises(ValueError):
        build_trace(order="nonsense")
