#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=fixquant-vgg16tilecnn-export
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

# Export the vgg16_tilecnn ModelPackage (step 8) and run the acceptance check.
#
# The package carries the INT8 graph, every parameter, one preprocessed
# validation input and the bit-exact integer reference output the accelerator
# must reproduce. tools/check_deeptile_legality.py then verifies the graph
# against the fabric's rules and every manifest checksum against the files on
# disk.
#
# Usage: sbatch scripts/jobs/vgg16_tilecnn_export.sbatch [release-id]

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
ZOO_ROOT="${FIXQUANT_ZOO_ROOT:-$REPO/model_zoo}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"

MODEL=vgg16_tilecnn
RELEASE_ID="${1:-${FIXQUANT_RELEASE_ID:-$MODEL/imagenet1k/int8-tqt@v1.0.0}}"
IMAGE="${FIXQUANT_EXPORT_IMAGE:-$REPO/assets/new.JPEG}"
OUT_DIR="${FIXQUANT_EXPORT_DIR:-$REPO/outputs/${MODEL}_int8_deeptile}"

cd "$REPO"
mkdir -p "$TORCH_CACHE"

export APPTAINERENV_FIXQUANT_ZOO_ROOT="$ZOO_ROOT"
export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'FixQuant %s ModelPackage export\n' "$MODEL"
printf 'Slurm job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$SLURMD_NODENAME"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Release: %s\n' "$RELEASE_ID"
printf 'Validation input: %s\n' "$IMAGE"
printf 'Output: %s\n' "$OUT_DIR"

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/export_deeptile_graph.py \
    --zoo-model "$RELEASE_ID" \
    --zoo-root "$ZOO_ROOT" \
    --image "$IMAGE" \
    --out_dir "$OUT_DIR"

printf '\n=== Acceptance check ===\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/check_deeptile_legality.py "$OUT_DIR"

printf '\nModelPackage exported and accepted: %s\n' "$OUT_DIR"
