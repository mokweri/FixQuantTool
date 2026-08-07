#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --array=0-2%3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fixquant-qat-sweep
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%A_%a.out

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"

MODELS=(vgg16 resnet18 resnet50)
TRAIN_BATCH_SIZES=(32 128 64)
TEST_BATCH_SIZES=(64 256 128)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
if ((TASK_ID < 0 || TASK_ID >= ${#MODELS[@]})); then
    printf 'Error: unsupported array task ID: %s\n' "$TASK_ID" >&2
    exit 1
fi

MODEL="${MODELS[$TASK_ID]}"
TRAIN_BATCH_SIZE="${FIXQUANT_TRAIN_BATCH_SIZE:-${TRAIN_BATCH_SIZES[$TASK_ID]}}"
TEST_BATCH_SIZE="${FIXQUANT_TEST_BATCH_SIZE:-${TEST_BATCH_SIZES[$TASK_ID]}}"
EPOCHS="${FIXQUANT_EPOCHS:-5}"
WORKERS="${FIXQUANT_WORKERS:-12}"
CALIB_BATCHES="${FIXQUANT_CALIB_BATCHES:-20}"
CALIB_SCOPE="${FIXQUANT_CALIB_SCOPE:-5}"
INIT_LR="${FIXQUANT_INIT_LR:-1e-5}"
QUANTIZER_LR="${FIXQUANT_QUANTIZER_LR:-1e-2}"
SEED="${FIXQUANT_SEED:-0}"
RUN_DIR="${FIXQUANT_RUN_DIR:-$STORAGE_BASE/results/qat/$MODEL/$SLURM_JOB_ID}"

if [[ ! -d "$REPO/.git" ]]; then
    printf 'Error: repository not found: %s\n' "$REPO" >&2
    exit 1
fi
if [[ ! -f "$DATASET/dataset_dict.json" ]]; then
    printf 'Error: ImageNet dataset not found: %s\n' "$DATASET" >&2
    exit 1
fi
if ((WORKERS > SLURM_CPUS_PER_TASK)); then
    printf 'Error: workers (%s) exceed allocated CPUs (%s).\n' \
        "$WORKERS" "$SLURM_CPUS_PER_TASK" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_PYTHONHASHSEED="$SEED"

printf 'FixQuant ImageNet QAT model sweep\n'
printf 'Array job/task: %s/%s\n' "$SLURM_ARRAY_JOB_ID" "$TASK_ID"
printf 'Slurm job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$SLURMD_NODENAME"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Model: %s\n' "$MODEL"
printf 'Dataset: %s\n' "$DATASET"
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Epochs: %s\n' "$EPOCHS"
printf 'Train/test batch: %s/%s\n' "$TRAIN_BATCH_SIZE" "$TEST_BATCH_SIZE"
printf 'Workers: %s\n' "$WORKERS"
printf 'Calibration batches/scope: %s/%s\n' "$CALIB_BATCHES" "$CALIB_SCOPE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/qat_train.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --n_worker "$WORKERS" \
    --n_epochs "$EPOCHS" \
    --init_lr "$INIT_LR" \
    --quantizer_lr "$QUANTIZER_LR" \
    --threshold_freeze_frac 0.7 \
    --calib_batches "$CALIB_BATCHES" \
    --calib_scope "$CALIB_SCOPE" \
    --validation_frequency 1 \
    --display_freq 100 \
    --manual_seed "$SEED" \
    --gpus 0 \
    --save_dir "$RUN_DIR"

printf 'QAT completed for %s. Results: %s/%s\n' "$MODEL" "$RUN_DIR" "$MODEL"
