#!/usr/bin/env bash
# Generate AMBER-g responses with MMaDA, validate the official JSON, and, for a
# full 1..1004 run, score it with the official AMBER evaluator.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

# shellcheck source=python_env.sh
source "$SCRIPT_DIR/python_env.sh"
resolve_python
export PYTHON

OUTPUT="$ROOT/results/amber_g/mmada_predictions.json"
AMBER_REPO="$ROOT/third_party/AMBER"
START_ID=1
END_ID=1004
SKIP_SCORE=0

args=("$@")
index=0
while [[ $index -lt ${#args[@]} ]]; do
  case "${args[$index]}" in
    --output)
      OUTPUT=${args[$((index + 1))]}
      index=$((index + 2))
      ;;
    --amber-repo)
      AMBER_REPO=${args[$((index + 1))]}
      index=$((index + 2))
      ;;
    --start-id)
      START_ID=${args[$((index + 1))]}
      index=$((index + 2))
      ;;
    --end-id)
      END_ID=${args[$((index + 1))]}
      index=$((index + 2))
      ;;
    --skip-score)
      SKIP_SCORE=1
      index=$((index + 1))
      ;;
    *)
      index=$((index + 1))
      ;;
  esac
done

forward=()
for argument in "$@"; do
  if [[ "$argument" != "--skip-score" ]]; then
    forward+=("$argument")
  fi
done

if [[ ${#forward[@]} -eq 0 ]]; then
  "$PYTHON" "$SCRIPT_DIR/run_mmada_amber_g.py"
else
  "$PYTHON" "$SCRIPT_DIR/run_mmada_amber_g.py" "${forward[@]}"
fi

SCOPE_LINE=$("$PYTHON" "$SCRIPT_DIR/run_mmada_amber_g.py" \
  --validate-only \
  --output "$OUTPUT" \
  --start-id "$START_ID" \
  --end-id "$END_ID" \
  --amber-repo "$AMBER_REPO" | tail -n 1)

if [[ "$SCOPE_LINE" != "AMBER_SCOPE full" && "$SCOPE_LINE" != "AMBER_SCOPE subset" ]]; then
  echo "Could not read validation scope from: $SCOPE_LINE" >&2
  exit 1
fi

if [[ "$SKIP_SCORE" -eq 1 || "$SCOPE_LINE" == "AMBER_SCOPE subset" ]]; then
  echo "Skipped the official AMBER aggregate scorer."
  echo "Subset CHAIR/Cover/Hal/Cog would not be comparable to full AMBER-g."
  echo "Predictions: $OUTPUT"
  exit 0
fi

bash "$SCRIPT_DIR/evaluate_amber_g.sh" \
  --predictions "$OUTPUT" \
  --amber-repo "$AMBER_REPO" \
  --metrics-out "$(dirname "$OUTPUT")/amber_g_metrics.txt"
