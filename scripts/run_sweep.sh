#!/usr/bin/env bash
# Run the baseline harness across the prefix-sharing trace sweep against an
# already-running vLLM server. Generates traces first if absent.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1}"
LABEL="${LABEL:-baseline}"
PY="${PY:-.venv/bin/python}"

mkdir -p bench/results

if [[ ! -f data/trace_shared90.jsonl ]]; then
  bash scripts/gen_traces.sh
fi

for tag in 00 50 90; do
  echo "=== shared=${tag}% ==="
  "$PY" -m bench.run_baseline \
    --trace "data/trace_shared${tag}.jsonl" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --concurrency "$CONCURRENCY" \
    --label "${LABEL}_shared${tag}" \
    --out "bench/results/${LABEL}_shared${tag}.json"
done

echo "Results in bench/results/"
