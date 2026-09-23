#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=fixquant-vgg16tilecnn-smoke
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

# Structural and end-to-end legality checks for the vgg16_tilecnn variant.
# Runs the full QAT -> hardware-model -> ModelPackage export on random weights
# and applies tools/check_tilecnn_legality.py, so the architecture is proven
# exportable before any GPU hours go into training it.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"

mkdir -p "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'FixQuant vgg16_tilecnn smoke test\n'
printf 'Slurm job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$SLURMD_NODENAME"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python -m pytest -q -rs tests/test_vgg16_tilecnn.py

printf '\nRegression suite (must stay green after the exporter change)...\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python -m pytest -q -m 'not slow'

printf '\nvgg16_tilecnn smoke test completed.\n'
