#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=fixquant-imagenet-smoke
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

set -euo pipefail

REPO="${FIXQUANT_REPO:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"

cd "$REPO"
srun scripts/run_arrhenius.sh \
    python tools/imagenet_smoke.py \
    --dataroot "$DATASET" \
    --batch-size 8 \
    --workers 4
