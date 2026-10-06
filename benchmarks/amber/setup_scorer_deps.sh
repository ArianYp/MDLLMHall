#!/usr/bin/env bash
# Install the Python packages imported by the official AMBER scorer and download
# the NLTK and spaCy resources it loads. Does not modify AMBER metric files.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=python_env.sh
source "$SCRIPT_DIR/python_env.sh"
resolve_python

"$PYTHON" -m pip install -U spacy nltk tqdm
"$PYTHON" -m spacy download en_core_web_lg
"$PYTHON" "$SCRIPT_DIR/setup_scorer_deps.py"
