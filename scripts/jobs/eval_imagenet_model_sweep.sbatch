#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --array=0-2%3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=fixquant-eval-sweep
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%A_%a.out

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"
DEPLOY_BACKEND="${FIXQUANT_DEPLOY_BACKEND:-tilecnn}"
WORKERS="${FIXQUANT_WORKERS:-12}"

MODELS=(vgg16 resnet18 resnet50)
TRAIN_JOB_IDS=(949945 949946 949944)
QAT_BATCH_SIZES=(64 256 128)
DEPLOY_BATCH_SIZES=(32 128 64)

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"
if ((TASK_ID < 0 || TASK_ID >= ${#MODELS[@]})); then
    printf 'Error: unsupported array task ID: %s\n' "$TASK_ID" >&2
    exit 1
fi

MODEL="${MODELS[$TASK_ID]}"
TRAIN_JOB_ID="${TRAIN_JOB_IDS[$TASK_ID]}"
QAT_BATCH_SIZE="${FIXQUANT_QAT_BATCH_SIZE:-${QAT_BATCH_SIZES[$TASK_ID]}}"
DEPLOY_BATCH_SIZE="${FIXQUANT_DEPLOY_BATCH_SIZE:-${DEPLOY_BATCH_SIZES[$TASK_ID]}}"
CHECKPOINT="${FIXQUANT_CHECKPOINT:-$STORAGE_BASE/results/qat/$MODEL/$TRAIN_JOB_ID/$MODEL/checkpoint/model_best.pth.tar}"
RUN_DIR="${FIXQUANT_EVAL_DIR:-$STORAGE_BASE/results/eval/$MODEL/$SLURM_JOB_ID}"

if [[ ! -f "$CHECKPOINT" ]]; then
    printf 'Error: checkpoint not found: %s\n' "$CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$DATASET/dataset_dict.json" ]]; then
    printf 'Error: ImageNet dataset not found: %s\n' "$DATASET" >&2
    exit 1
fi
if [[ "$DEPLOY_BACKEND" != "emu" && "$DEPLOY_BACKEND" != "tilecnn" ]]; then
    printf 'Error: FIXQUANT_DEPLOY_BACKEND must be emu or tilecnn.\n' >&2
    exit 1
fi
if ((WORKERS > SLURM_CPUS_PER_TASK)); then
    printf 'Error: workers (%s) exceed allocated CPUs (%s).\n' \
        "$WORKERS" "$SLURM_CPUS_PER_TASK" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/qat" "$RUN_DIR/deploy" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'FixQuant ImageNet checkpoint evaluation sweep\n'
printf 'Array job/task: %s/%s\n' "$SLURM_ARRAY_JOB_ID" "$TASK_ID"
printf 'Slurm job ID: %s\n' "$SLURM_JOB_ID"
printf 'Model: %s\n' "$MODEL"
printf 'Checkpoint: %s\n' "$CHECKPOINT"
printf 'Dataset: %s\n' "$DATASET"
printf 'Deploy backend: %s\n' "$DEPLOY_BACKEND"
printf 'QAT/deploy batch sizes: %s/%s\n' "$QAT_BATCH_SIZE" "$DEPLOY_BATCH_SIZE"
printf 'Results: %s\n' "$RUN_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

printf '\n=== QAT model evaluation ===\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/qat_test.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --test_batch_size "$QAT_BATCH_SIZE" \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/qat" \
    --metrics-output "$RUN_DIR/qat_metrics.json"

printf '\n=== Hardware deployment evaluation ===\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/deploy_eval.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --model_type "$DEPLOY_BACKEND" \
    --test_batch_size "$DEPLOY_BATCH_SIZE" \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/deploy" \
    --metrics-output "$RUN_DIR/${DEPLOY_BACKEND}_metrics.json" \
    --qconfig-output "$RUN_DIR/qconfig.json"

printf '\nCheckpoint evaluations completed successfully for %s.\n' "$MODEL"
