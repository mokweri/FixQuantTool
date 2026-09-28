#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --array=0-3%4
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=fixquant-eval-fp32
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%A_%a.out

# FP32 reference accuracy for every released model, on the same ImageNet
# validation set and preprocessing as the released QAT and DeepTile metrics.
# Writes <model>_fp32_metrics.json beneath results/eval/fp32/<array-job-id>/.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"
WORKERS="${FIXQUANT_WORKERS:-12}"

MODELS=(resnet18 resnet50 vgg16 mobilenet_v2)
BATCH_SIZES=(256 128 64 256)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
if ((TASK_ID < 0 || TASK_ID >= ${#MODELS[@]})); then
    printf 'Error: unsupported array task ID: %s\n' "$TASK_ID" >&2
    exit 1
fi

MODEL="${MODELS[$TASK_ID]}"
BATCH_SIZE="${FIXQUANT_BATCH_SIZE:-${BATCH_SIZES[$TASK_ID]}}"
RUN_DIR="${FIXQUANT_EVAL_DIR:-$STORAGE_BASE/results/eval/fp32/$SLURM_ARRAY_JOB_ID}"

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

printf 'FixQuant FP32 reference evaluation\n'
printf 'Array job/task: %s/%s\n' "$SLURM_ARRAY_JOB_ID" "$TASK_ID"
printf 'Model: %s\n' "$MODEL"
printf 'Dataset: %s\n' "$DATASET"
printf 'Batch size: %s\n' "$BATCH_SIZE"
printf 'Results: %s\n' "$RUN_DIR"
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

printf '\nFP32 evaluation completed successfully for %s.\n' "$MODEL"
