#!/usr/bin/env bash
# Score an official AMBER-g response file with the untouched AMBER evaluator.
# Subset files are rejected here. Five-example smoke runs are not AMBER-g scores.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=python_env.sh
source "$SCRIPT_DIR/python_env.sh"
resolve_python

PREDICTIONS=""
AMBER_REPO=${AMBER_REPO:-"$ROOT/third_party/AMBER"}
METRICS_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions|--inference-data)
      PREDICTIONS=$2
      shift 2
      ;;
    --amber-repo)
      AMBER_REPO=$2
      shift 2
      ;;
    --metrics-out)
      METRICS_OUT=$2
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash benchmarks/amber/evaluate_amber_g.sh --predictions PREDICTIONS.json [--amber-repo DIR] [--metrics-out FILE]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PREDICTIONS" ]]; then
  echo "Pass --predictions path/to/mmada_predictions.json" >&2
  exit 1
fi
if [[ ! -f "$PREDICTIONS" ]]; then
  echo "Predictions file not found: $PREDICTIONS" >&2
  exit 1
fi
if [[ ! -f "$AMBER_REPO/inference.py" ]]; then
  echo "Official AMBER inference.py not found at $AMBER_REPO. Run benchmarks/amber/download_amber.sh." >&2
  exit 1
fi

PREDICTIONS=$(cd "$(dirname "$PREDICTIONS")" && pwd)/$(basename "$PREDICTIONS")
if [[ -z "$METRICS_OUT" ]]; then
  METRICS_OUT="$(dirname "$PREDICTIONS")/amber_g_metrics.txt"
fi
mkdir -p "$(dirname "$METRICS_OUT")"

"$PYTHON" "$SCRIPT_DIR/run_mmada_amber_g.py" \
  --validate-only \
  --output "$PREDICTIONS" \
  --start-id 1 \
  --end-id 1004 \
  --amber-repo "$AMBER_REPO"

"$PYTHON" "$SCRIPT_DIR/setup_scorer_deps.py"

(
  cd "$AMBER_REPO"
  "$PYTHON" inference.py \
    --inference_data "$PREDICTIONS" \
    --evaluation_type g
) | tee "$METRICS_OUT"

"$PYTHON" - "$METRICS_OUT" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
found = {}
for name in ("CHAIR", "Cover", "Hal", "Cog"):
    match = re.search(rf"^{name}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, re.MULTILINE)
    if not match:
        sys.exit(f"Official scorer output is missing {name}. See {sys.argv[1]}")
    found[name] = match.group(1)
print(f"CHAIR {found['CHAIR']}")
print(f"Cover {found['Cover']}")
print(f"Hal {found['Hal']}")
print(f"Cog {found['Cog']}")
print(f"Saved official scorer output to {sys.argv[1]}")
PY
