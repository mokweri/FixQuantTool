#!/usr/bin/env bash

# Run a command in FixQuantTool's pinned Arrhenius PyTorch environment.
# This launcher must be called from an ARM GPU allocation.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
CONTAINER_TAG="${FIXQUANT_CONTAINER_TAG:-25.09-py3}"
CONTAINER="${FIXQUANT_CONTAINER:-$STORAGE_BASE/software/containers/pytorch-$CONTAINER_TAG.sif}"
VENV="${FIXQUANT_VENV:-$STORAGE_BASE/software/venvs/fixquant-ngc-$CONTAINER_TAG}"
ZOO_ROOT="${FIXQUANT_ZOO_ROOT:-$REPO/model_zoo}"

if (($# == 0)); then
    printf 'Usage: %s <command> [args...]\n' "$0" >&2
    printf 'Example: %s python -m pytest -q -m "not slow"\n' "$0" >&2
    exit 2
fi

if [[ "$(uname -m)" != "aarch64" || -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Error: run this launcher inside an Arrhenius GPU allocation.\n' >&2
    printf 'Example: interactive -A naiss2025-22-1409-gpu -p gpu --gpus 1 -n 1 -c 8 -t 01:00:00\n' >&2
    exit 1
fi

if [[ ! -s "$CONTAINER" ]]; then
    printf 'Error: container not found: %s\n' "$CONTAINER" >&2
    printf 'Run: bash %s/scripts/setup_arrhenius_env.sh\n' "$REPO" >&2
    exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    printf 'Error: Python environment not found: %s\n' "$VENV" >&2
    printf 'Run: bash %s/scripts/setup_arrhenius_env.sh\n' "$REPO" >&2
    exit 1
fi

bind_args=(--bind "$STORAGE_BASE:$STORAGE_BASE")
if [[ -d /dataset ]]; then
    bind_args+=(--bind /dataset:/dataset)
fi
if [[ -d "$ZOO_ROOT" ]]; then
    bind_args+=(--bind "$ZOO_ROOT:$ZOO_ROOT")
fi

export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_FIXQUANT_REPO="$REPO"
export APPTAINERENV_FIXQUANT_ZOO_ROOT="$ZOO_ROOT"

exec apptainer exec \
    --cleanenv \
    --nv \
    "${bind_args[@]}" \
    "$CONTAINER" \
    bash -c 'source "$1/bin/activate"; shift; cd "$FIXQUANT_REPO"; exec "$@"' \
    bash "$VENV" "$@"
