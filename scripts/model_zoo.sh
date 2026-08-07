#!/usr/bin/env bash

# Run lightweight model-zoo administration directly on an Arrhenius login node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$REPO/tools/model_zoo.py" "$@"
