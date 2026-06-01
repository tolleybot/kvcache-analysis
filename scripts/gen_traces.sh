#!/usr/bin/env bash
# Generate the standard Stage 2 trace sweep: low, medium, and high prefix
# sharing. Hit rate is the variable that drives the rest, so we sweep the
# shared-system-prompt fraction and keep everything else fixed.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT_DIR="data"
mkdir -p "$OUT_DIR"

PY="${PY:-.venv/bin/python}"
COMMON=(--num-sessions 16 --turns-per-session 4 --system-words 400 \
        --turn-words 40 --response-words 40 --max-tokens 64 \
        --order round_robin --seed 0)

# Map each sharing fraction to a two-digit percent tag without external tools.
declare -A TAGS=( [0.0]=00 [0.5]=50 [0.9]=90 )
for frac in 0.0 0.5 0.9; do
  tag="${TAGS[$frac]}"
  "$PY" -m bench.trace "${COMMON[@]}" \
    --shared-system-fraction "$frac" \
    --out "$OUT_DIR/trace_shared${tag}.jsonl"
done

echo "Traces written to $OUT_DIR/"
