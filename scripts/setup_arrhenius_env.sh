#!/usr/bin/env bash

# Reproducible FixQuantTool environment setup for Arrhenius.
# Run from the login node with: bash scripts/setup_arrhenius_env.sh

set -euo pipefail

COMPUTE_ACCOUNT="${FIXQUANT_COMPUTE_ACCOUNT:-naiss2025-22-1409-gpu}"
STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
CONTAINER_TAG="${FIXQUANT_CONTAINER_TAG:-25.09-py3}"
CONTAINER="${FIXQUANT_CONTAINER:-$STORAGE_BASE/software/containers/pytorch-$CONTAINER_TAG.sif}"
VENV="${FIXQUANT_VENV:-$STORAGE_BASE/software/venvs/fixquant-ngc-$CONTAINER_TAG}"
SNAPSHOT="${FIXQUANT_SNAPSHOT:-$STORAGE_BASE/software/locks/fixquant-ngc-$CONTAINER_TAG.freeze.txt}"
SCRIPT_PATH="$(readlink -f "$0")"

inside_allocation=false
force_container=false

usage() {
    printf 'Usage: %s [--force-container]\n' "$0"
}

while (($#)); do
    case "$1" in
        --inside-allocation) inside_allocation=true ;;
        --force-container) force_container=true ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

mkdir -p \
    "$STORAGE_BASE/results" \
    "$STORAGE_BASE/software/containers" \
    "$STORAGE_BASE/software/venvs" \
    "$STORAGE_BASE/software/locks"

if [[ "$(uname -m)" != "aarch64" ]]; then
    if [[ "$inside_allocation" == true ]]; then
        printf 'Error: setup job is not running on an ARM GPU node.\n' >&2
        exit 1
    fi

    submit_args=(
        -A "$COMPUTE_ACCOUNT"
        -p gpu
        --gpus=1
        -n 1
        -c 32
        -t 01:00:00
        -J fixquant-setup
        -o "$STORAGE_BASE/results/fixquant-setup-%j.out"
        "$SCRIPT_PATH"
        --inside-allocation
    )
    if [[ "$force_container" == true ]]; then
        submit_args+=(--force-container)
    fi

    printf 'Submitting FixQuantTool setup to an Arrhenius GPU node...\n'
    submission="$(sbatch "${submit_args[@]}")"
    printf '%s\n' "$submission"
    job_id="${submission##* }"
    printf 'Monitor with: squeue -j %s\n' "$job_id"
    printf 'Log file: %s/results/fixquant-setup-%s.out\n' "$STORAGE_BASE" "$job_id"
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Error: run this on the login node or inside a Slurm GPU allocation.\n' >&2
    exit 1
fi

if [[ ! -d "$REPO/.git" ]]; then
    printf 'Error: FixQuantTool checkout not found at %s\n' "$REPO" >&2
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
    printf 'Error: apptainer is not available on this node.\n' >&2
    exit 1
fi

printf 'Host: %s (%s)\n' "$(hostname)" "$(uname -m)"
printf 'Repository: %s\n' "$REPO"
printf 'Container: %s\n' "$CONTAINER"
printf 'Environment: %s\n' "$VENV"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ "$force_container" == true && -e "$CONTAINER" ]]; then
    rm -f -- "$CONTAINER"
fi

if [[ ! -s "$CONTAINER" ]]; then
    export APPTAINER_TMPDIR="${SNIC_TMP:-/scratch/local}"
    export APPTAINER_MKSQUASHFS_ARGS="${APPTAINER_MKSQUASHFS_ARGS:--processors 4}"
    printf 'Pulling NVIDIA PyTorch container %s...\n' "$CONTAINER_TAG"
    apptainer pull --disable-cache \
        "$CONTAINER" \
        "docker://nvcr.io/nvidia/pytorch:$CONTAINER_TAG"
else
    printf 'Using existing container.\n'
fi

bind_args=(--bind "$STORAGE_BASE:$STORAGE_BASE")
if [[ -d /dataset ]]; then
    bind_args+=(--bind /dataset:/dataset)
fi

container_exec=(
    apptainer exec
    --cleanenv
    --nv
    "${bind_args[@]}"
    "$CONTAINER"
)

if [[ ! -x "$VENV/bin/python" ]]; then
    printf 'Creating Python environment...\n'
    "${container_exec[@]}" python -m venv --system-site-packages "$VENV"
else
    printf 'Using existing Python environment.\n'
fi

printf 'Installing the pinned ImageNet Arrow reader...\n'
"${container_exec[@]}" \
    "$VENV/bin/python" -m pip install "datasets==4.1.1"

printf 'Installing FixQuantTool without changing pinned container dependencies...\n'
"${container_exec[@]}" \
    "$VENV/bin/python" -m pip install --no-deps -e "$REPO"

printf 'Checking installed dependencies...\n'
"${container_exec[@]}" "$VENV/bin/python" -m pip check

printf 'Checking PyTorch, CUDA, and FixQuantTool imports...\n'
"${container_exec[@]}" "$VENV/bin/python" -c '
import onnx
import datasets
import pytest
import torch
import torchvision
import yaml
import fixquant

assert torch.cuda.is_available(), "PyTorch cannot see the allocated GPU"
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("datasets:", datasets.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("FixQuant import: OK")
'

printf 'Running the fast test suite...\n'
(
    cd "$REPO"
    "${container_exec[@]}" "$VENV/bin/python" -m pytest -q -m 'not slow'
)

"${container_exec[@]}" "$VENV/bin/python" -m pip freeze > "$SNAPSHOT"

printf '\nSetup complete.\n'
printf 'Dependency snapshot: %s\n' "$SNAPSHOT"
printf 'Run commands with: %s/scripts/run_arrhenius.sh <command> [args...]\n' "$REPO"
