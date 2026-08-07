#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=fixquant-eval-mobilenet
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
CHECKPOINT="${FIXQUANT_CHECKPOINT:-$STORAGE_BASE/results/qat/mobilenet_v2/904116/mobilenet_v2/checkpoint/model_best.pth.tar}"
DEPLOY_BACKEND="${FIXQUANT_DEPLOY_BACKEND:-tilecnn}"
RUN_DIR="${FIXQUANT_EVAL_DIR:-$STORAGE_BASE/results/eval/mobilenet_v2/$SLURM_JOB_ID}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"
WORKERS="${FIXQUANT_WORKERS:-12}"

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

printf 'FixQuant MobileNetV2 checkpoint evaluation\n'
printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Checkpoint: %s\n' "$CHECKPOINT"
printf 'Dataset: %s\n' "$DATASET"
printf 'Deploy backend: %s\n' "$DEPLOY_BACKEND"
printf 'Results: %s\n' "$RUN_DIR"

printf '\n=== QAT model evaluation ===\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/qat_test.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model mobilenet_v2 \
    --checkpoint "$CHECKPOINT" \
    --cle \
    --test_batch_size 128 \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/qat"

printf '\n=== Hardware deployment evaluation ===\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/deploy_eval.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model mobilenet_v2 \
    --checkpoint "$CHECKPOINT" \
    --cle \
    --model_type "$DEPLOY_BACKEND" \
    --test_batch_size 64 \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/deploy"

printf '\nCheckpoint evaluations completed successfully.\n'
