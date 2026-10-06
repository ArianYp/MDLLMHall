#!/usr/bin/env bash
# Clone the official AMBER scorer and prepare the AMBER-g query file and images.
# Images are never committed. Pass a local archive as the first argument, or set
# AMBER_IMAGE_ZIP. If neither is set and gdown is installed, the official Drive
# file is downloaded. gdown is optional.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AMBER_REPO=${AMBER_REPO:-"$ROOT/third_party/AMBER"}
DATA_DIR=${AMBER_DATA_DIR:-"$ROOT/data/amber"}
IMAGE_DIR=${AMBER_IMAGE_DIR:-"$DATA_DIR/images"}
QUERY_URL="https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/query/query_generative.json"
DRIVE_URL="https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing"
ARCHIVE=${1:-${AMBER_IMAGE_ZIP:-}}

# shellcheck source=python_env.sh
source "$SCRIPT_DIR/python_env.sh"
resolve_python

cd "$ROOT"
mkdir -p "$DATA_DIR" "$(dirname "$AMBER_REPO")"

if [[ ! -f "$AMBER_REPO/inference.py" ]]; then
  echo "Cloning official AMBER into $AMBER_REPO"
  git clone --depth 1 https://github.com/junyangwang0410/AMBER.git "$AMBER_REPO"
else
  echo "Official AMBER scorer already present at $AMBER_REPO"
fi

QUERY_SOURCE="$AMBER_REPO/data/query/query_generative.json"
QUERY_DEST="$DATA_DIR/query_generative.json"
if [[ ! -f "$QUERY_SOURCE" ]]; then
  echo "Official query file missing from the clone. Downloading $QUERY_URL"
  curl -fsSL "$QUERY_URL" -o "$QUERY_DEST"
else
  ln -sfn "$QUERY_SOURCE" "$QUERY_DEST"
fi

if [[ ! -f "$QUERY_DEST" ]]; then
  echo "Failed to place query_generative.json at $QUERY_DEST" >&2
  exit 1
fi

count_expected_images() {
  "$PYTHON" - "$IMAGE_DIR" <<'PY'
import os
import sys
image_dir = sys.argv[1]
missing = [f"AMBER_{i}.jpg" for i in range(1, 1005) if not os.path.isfile(os.path.join(image_dir, f"AMBER_{i}.jpg"))]
if missing:
    print(len(missing))
    sys.exit(1)
print(0)
PY
}

link_image_dir() {
  local source_dir=$1
  mkdir -p "$(dirname "$IMAGE_DIR")"
  if [[ -L "$IMAGE_DIR" || ! -e "$IMAGE_DIR" ]]; then
    ln -sfn "$source_dir" "$IMAGE_DIR"
  elif [[ "$(cd "$source_dir" && pwd)" != "$(cd "$IMAGE_DIR" && pwd)" ]]; then
    find "$source_dir" -maxdepth 1 -type f -name 'AMBER_*.jpg' -exec cp -n {} "$IMAGE_DIR"/ \;
  fi
}

extract_archive() {
  local archive=$1
  local staging="$DATA_DIR/archive_extract"
  rm -rf "$staging"
  mkdir -p "$staging"
  echo "Extracting $archive"
  case "$archive" in
    *.zip) unzip -q "$archive" -d "$staging" ;;
    *.tar.gz|*.tgz) tar -xzf "$archive" -C "$staging" ;;
    *.tar) tar -xf "$archive" -C "$staging" ;;
    *)
      echo "Unsupported archive type: $archive" >&2
      exit 1
      ;;
  esac
  local found
  found=$(find "$staging" -type f -name 'AMBER_1.jpg' -print -quit)
  if [[ -z "$found" ]]; then
    echo "Extracted archive does not contain AMBER_1.jpg." >&2
    exit 1
  fi
  link_image_dir "$(dirname "$found")"
}

if count_expected_images >/dev/null 2>&1; then
  echo "AMBER_1.jpg through AMBER_1004.jpg are already in $IMAGE_DIR"
else
  if [[ -n "$ARCHIVE" ]]; then
    if [[ ! -f "$ARCHIVE" ]]; then
      echo "Image archive not found: $ARCHIVE" >&2
      exit 1
    fi
    extract_archive "$ARCHIVE"
  elif command -v gdown >/dev/null 2>&1 || "$PYTHON" -c "import gdown" >/dev/null 2>&1; then
    mkdir -p "$DATA_DIR/downloads"
    ARCHIVE_PATH="$DATA_DIR/downloads/amber_images.zip"
    echo "Downloading the official image archive with gdown"
    if command -v gdown >/dev/null 2>&1; then
      gdown --fuzzy "$DRIVE_URL" -O "$ARCHIVE_PATH"
    else
      "$PYTHON" -m gdown --fuzzy "$DRIVE_URL" -O "$ARCHIVE_PATH"
    fi
    extract_archive "$ARCHIVE_PATH"
  else
    cat <<EOF
The official AMBER scorer and query_generative.json are ready.
The generative images are not.

Download the official image archive from:
  $DRIVE_URL

Then rerun:
  bash benchmarks/amber/download_amber.sh /path/to/downloaded_archive.zip

If you want this script to fetch the Drive file itself, install gdown and rerun:
  pip install gdown
  bash benchmarks/amber/download_amber.sh
EOF
    exit 1
  fi
fi

if ! missing=$(count_expected_images); then
  echo "AMBER image check failed: $missing of AMBER_1.jpg..AMBER_1004.jpg are missing from $IMAGE_DIR" >&2
  exit 1
fi

echo "AMBER-g data ready"
echo "  scorer: $AMBER_REPO"
echo "  query:  $QUERY_DEST"
echo "  images: $IMAGE_DIR"
