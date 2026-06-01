"""Synthetic agentic, multi-turn workload generator for KV cache benchmarking.

The trace models the motivating workload: multi-turn sessions that share a long
system prompt and grow a conversation history across turns. Two reuse effects
are produced deliberately, because they are what a KV cache is supposed to
exploit:

* within a session, turn ``t`` repeats the entire prompt of turn ``t - 1`` plus
  that turn's response, so each turn's prompt is a prefix superset of the last;
* across sessions, a configurable fraction share one common system prompt, which
  is the large shared prefix typical of agentic serving (system instructions and
  tool definitions).

Token counts are approximated by words. The harness reports vLLM's own
token-level hit rate, so this generator only needs to control relative prefix
lengths and sharing in a reproducible way, not exact token boundaries.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

# A small fixed vocabulary keeps traces reproducible and human-readable. The
# content is meaningless on purpose; only its length and reuse structure matter.
_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega vector matrix tensor "
    "gradient kernel buffer pointer socket thread cache prefix suffix latency "
    "throughput context session prompt token decode prefill attention serving"
).split()


@dataclass
class Request:
    """A single request issued to the serving engine, in issue order."""

    request_id: str
    session_id: int
    turn_index: int
    prompt: str
    max_tokens: int


def _filler(rng: random.Random, n_words: int) -> str:
    """Deterministic pseudo-text of ``n_words`` words drawn from the vocabulary."""
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def build_trace(
    *,
    num_sessions: int = 16,
    turns_per_session: int = 4,
    system_words: int = 400,
    turn_words: int = 40,
    response_words: int = 40,
    shared_system_fraction: float = 0.9,
    max_tokens: int = 64,
    order: str = "round_robin",
    seed: int = 0,
) -> list[Request]:
    """Build a reproducible list of requests in issue order.

    ``shared_system_fraction`` is the fraction of sessions that reuse one common
    system prompt; the rest receive a unique system prompt. This is the main
    knob for cross-session (and, later, cross-instance) hit rate. Within-session
    reuse always occurs because turns accumulate history.

    ``order`` is ``"round_robin"`` (turn 0 of every session, then turn 1, ...),
    which interleaves sessions and stresses eviction, or ``"by_session"`` (all
    turns of a session before the next), which is the easy case for the cache.
    """
    if not 0.0 <= shared_system_fraction <= 1.0:
        raise ValueError("shared_system_fraction must be in [0, 1]")
    if order not in ("round_robin", "by_session"):
        raise ValueError("order must be 'round_robin' or 'by_session'")

    rng = random.Random(seed)
    common_system = "System: " + _filler(rng, system_words)
    n_shared = round(shared_system_fraction * num_sessions)

    # Build each session's system prompt and its per-turn user messages up front,
    # so generation order does not change the content.
    sessions: list[dict] = []
    for s in range(num_sessions):
        if s < n_shared:
            system = common_system
        else:
            system = f"System[{s}]: " + _filler(rng, system_words)
        user_msgs = [_filler(rng, turn_words) for _ in range(turns_per_session)]
        responses = [_filler(rng, response_words) for _ in range(turns_per_session)]
        sessions.append({"system": system, "users": user_msgs, "responses": responses})

    # Materialize the prompt seen at each turn, accumulating history per session.
    per_session_prompts: list[list[str]] = []
    for s in range(num_sessions):
        conversation = sessions[s]["system"]
        prompts: list[str] = []
        for t in range(turns_per_session):
            prompt = f"{conversation}\nUser: {sessions[s]['users'][t]}\nAssistant:"
            prompts.append(prompt)
            conversation = f"{prompt} {sessions[s]['responses'][t]}"
        per_session_prompts.append(prompts)

    requests: list[Request] = []

    def emit(s: int, t: int) -> None:
        requests.append(
            Request(
                request_id=f"s{s}-t{t}",
                session_id=s,
                turn_index=t,
                prompt=per_session_prompts[s][t],
                max_tokens=max_tokens,
            )
        )

    if order == "round_robin":
        for t in range(turns_per_session):
            for s in range(num_sessions):
                emit(s, t)
    else:
        for s in range(num_sessions):
            for t in range(turns_per_session):
                emit(s, t)

    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic KV cache benchmark trace")
    parser.add_argument("--num-sessions", type=int, default=16)
    parser.add_argument("--turns-per-session", type=int, default=4)
    parser.add_argument("--system-words", type=int, default=400)
    parser.add_argument("--turn-words", type=int, default=40)
    parser.add_argument("--response-words", type=int, default=40)
    parser.add_argument("--shared-system-fraction", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--order", choices=["round_robin", "by_session"], default="round_robin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Path to write the trace as JSON lines")
    args = parser.parse_args()

    trace = build_trace(
        num_sessions=args.num_sessions,
        turns_per_session=args.turns_per_session,
        system_words=args.system_words,
        turn_words=args.turn_words,
        response_words=args.response_words,
        shared_system_fraction=args.shared_system_fraction,
        max_tokens=args.max_tokens,
        order=args.order,
        seed=args.seed,
    )
    with open(args.out, "w") as f:
        for req in trace:
            f.write(json.dumps(asdict(req)) + "\n")
    print(f"wrote {len(trace)} requests to {args.out}")


if __name__ == "__main__":
    main()
