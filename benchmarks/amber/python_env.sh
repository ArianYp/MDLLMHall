# Shared interpreter resolution for the AMBER-g scripts.
# Expects ROOT to point at the MDLLM directory. Honors PYTHON when it starts.

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if ! "$PYTHON" -c "import sys" >/dev/null 2>&1; then
      echo "PYTHON=$PYTHON cannot start." >&2
      exit 1
    fi
    return
  fi

  if [[ -x "$ROOT/virtualenv/bin/python" ]] && "$ROOT/virtualenv/bin/python" -c "import sys" >/dev/null 2>&1; then
    PYTHON="$ROOT/virtualenv/bin/python"
    return
  fi

  if [[ -x "$ROOT/virtualenv/bin/python" ]] && command -v module >/dev/null 2>&1; then
    # Matches the Python module recorded in mdlm.slurm for this project's virtualenv.
    module load GCCcore/11.3.0 Python/3.11.3 >/dev/null 2>&1 || true
    if "$ROOT/virtualenv/bin/python" -c "import sys" >/dev/null 2>&1; then
      PYTHON="$ROOT/virtualenv/bin/python"
      return
    fi
  fi

  local candidate
  for candidate in python3 python; do
    if "$candidate" -c "import sys" >/dev/null 2>&1; then
      PYTHON=$candidate
      return
    fi
  done

  echo "No working Python interpreter found. Set PYTHON to the environment that runs MMaDA." >&2
  exit 1
}
