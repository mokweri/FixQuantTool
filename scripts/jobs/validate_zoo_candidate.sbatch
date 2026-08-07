#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=fixquant-zoo-validate
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

set -euo pipefail

if (($# != 1)); then
    printf 'Usage: sbatch %s <candidate-id>\n' "$0" >&2
    exit 2
fi

CANDIDATE_ID="$1"
STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
ZOO_ROOT="${FIXQUANT_ZOO_ROOT:-$REPO/model_zoo}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"

cd "$REPO"
export APPTAINERENV_FIXQUANT_ZOO_ROOT="$ZOO_ROOT"
export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'Validating model-zoo candidate: %s\n' "$CANDIDATE_ID"
printf 'Zoo root: %s\n' "$ZOO_ROOT"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/validate_zoo_candidate.py \
    "$CANDIDATE_ID" \
    --root "$ZOO_ROOT" \
    --workers 12
