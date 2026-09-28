#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --array=0-1%2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=fixquant-vgg16-pool-ablation
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%A_%a.out

# Free pooling ablation (step 1 of the vgg16_tilecnn plan).
#
# Task 0: stock torchvision vgg16_bn, the FP32 reference point.
# Task 1: the same pretrained weights with all five MaxPool2d(2, 2) replaced by
#         the one pooling geometry the DeepTile fabric implements (3x3/s2/p1).
#
# Pooling is parameter-free, so task 1 costs a single evaluation pass and
# isolates the architectural cost of the pooling change from the cost of the
# new classifier head.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"
WORKERS="${FIXQUANT_WORKERS:-12}"
BATCH_SIZE="${FIXQUANT_BATCH_SIZE:-64}"

MODELS=(vgg16_bn vgg16_bn_pool3)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
if ((TASK_ID < 0 || TASK_ID >= ${#MODELS[@]})); then
    printf 'Error: unsupported array task ID: %s\n' "$TASK_ID" >&2
    exit 1
fi

MODEL="${MODELS[$TASK_ID]}"
RUN_DIR="${FIXQUANT_EVAL_DIR:-$STORAGE_BASE/results/eval/pool_ablation/$SLURM_ARRAY_JOB_ID}"

if [[ ! -f "$DATASET/dataset_dict.json" ]]; then
    printf 'Error: ImageNet dataset not found: %s\n' "$DATASET" >&2
    exit 1
fi
if ((WORKERS > SLURM_CPUS_PER_TASK)); then
    printf 'Error: workers (%s) exceed allocated CPUs (%s).\n' \
        "$WORKERS" "$SLURM_CPUS_PER_TASK" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/$MODEL" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'FixQuant VGG-16 pooling ablation\n'
printf 'Array job/task: %s/%s\n' "$SLURM_ARRAY_JOB_ID" "$TASK_ID"
printf 'Model: %s\n' "$MODEL"
printf 'Dataset: %s\n' "$DATASET"
printf 'Batch size: %s\n' "$BATCH_SIZE"
printf 'Results: %s\n' "$RUN_DIR"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/fp32_eval.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    --test_batch_size "$BATCH_SIZE" \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/$MODEL" \
    --metrics-output "$RUN_DIR/${MODEL}_fp32_metrics.json"

printf '\nPooling ablation evaluation completed for %s.\n' "$MODEL"
