#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fixquant-vgg16tilecnn-ft
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

# Float training for vgg16_tilecnn (steps 3-5 of the plan), in two stages
# inside one allocation.
#
# Stage 1, head warm-up: the feature stack is frozen (weights and BN running
# statistics) and only the new head trains, so gradients from the randomly
# initialized head cannot damage the pretrained features.
#
# Stage 2, full fine-tune: the backbone is unfrozen and the whole network
# trains at a low learning rate. This stage is not optional -- the pooling
# change perturbs activation statistics at every stage, so the conv stack has
# to adapt. train.py's defaults (--n_epochs 150 --init_lr 0.1) are for training
# from scratch and are wrong here.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"

MODEL=vgg16_tilecnn
WORKERS="${FIXQUANT_WORKERS:-12}"
TRAIN_BATCH_SIZE="${FIXQUANT_TRAIN_BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${FIXQUANT_TEST_BATCH_SIZE:-64}"
SEED="${FIXQUANT_SEED:-0}"

WARMUP_EPOCHS="${FIXQUANT_WARMUP_EPOCHS:-2}"
WARMUP_LR="${FIXQUANT_WARMUP_INIT_LR:-0.01}"
WARMUP_WD="${FIXQUANT_WARMUP_WD:-1e-4}"

FINETUNE_EPOCHS="${FIXQUANT_FINETUNE_EPOCHS:-12}"
FINETUNE_LR="${FIXQUANT_FINETUNE_INIT_LR:-0.005}"
FINETUNE_WD="${FIXQUANT_FINETUNE_WD:-1e-4}"

RUN_DIR="${FIXQUANT_RUN_DIR:-$STORAGE_BASE/results/float/$MODEL/$SLURM_JOB_ID}"
WARMUP_DIR="$RUN_DIR/warmup"
FINETUNE_DIR="$RUN_DIR/finetune"
WARMUP_CKPT="$WARMUP_DIR/$MODEL/checkpoint/model_best.pth.tar"
FINETUNE_CKPT="$FINETUNE_DIR/$MODEL/checkpoint/model_best.pth.tar"

# Set to a checkpoint path to skip stage 1 and start the fine-tune from it.
RESUME_FROM="${FIXQUANT_RESUME_FROM:-}"

if [[ ! -f "$DATASET/dataset_dict.json" ]]; then
    printf 'Error: ImageNet dataset not found: %s\n' "$DATASET" >&2
    exit 1
fi
if ((WORKERS > SLURM_CPUS_PER_TASK)); then
    printf 'Error: workers (%s) exceed allocated CPUs (%s).\n' \
        "$WORKERS" "$SLURM_CPUS_PER_TASK" >&2
    exit 1
fi

mkdir -p "$WARMUP_DIR" "$FINETUNE_DIR" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_PYTHONHASHSEED="$SEED"

printf 'FixQuant %s float training\n' "$MODEL"
printf 'Slurm job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$SLURMD_NODENAME"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Dataset: %s\n' "$DATASET"
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Train/test batch: %s/%s, workers: %s, seed: %s\n' \
    "$TRAIN_BATCH_SIZE" "$TEST_BATCH_SIZE" "$WORKERS" "$SEED"
printf 'Stage 1 (head warm-up): %s epochs, lr %s, wd %s\n' \
    "$WARMUP_EPOCHS" "$WARMUP_LR" "$WARMUP_WD"
printf 'Stage 2 (full fine-tune): %s epochs, lr %s, wd %s\n' \
    "$FINETUNE_EPOCHS" "$FINETUNE_LR" "$FINETUNE_WD"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -z "$RESUME_FROM" ]]; then
    printf '\n=== Stage 1: head warm-up (backbone frozen) ===\n'
    srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
        python tools/train.py \
        --dataset imagenet \
        --dataroot "$DATASET" \
        --model "$MODEL" \
        --pretrained \
        --freeze_backbone \
        --n_epochs "$WARMUP_EPOCHS" \
        --init_lr "$WARMUP_LR" \
        --weight_decay "$WARMUP_WD" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --test_batch_size "$TEST_BATCH_SIZE" \
        --n_worker "$WORKERS" \
        --validation_frequency 1 \
        --display_freq 200 \
        --manual_seed "$SEED" \
        --gpus 0 \
        --save_dir "$WARMUP_DIR"

    if [[ ! -f "$WARMUP_CKPT" ]]; then
        printf 'Error: head warm-up produced no checkpoint: %s\n' "$WARMUP_CKPT" >&2
        exit 1
    fi
    RESUME_FROM="$WARMUP_CKPT"
fi

printf '\n=== Stage 2: full fine-tune (backbone trainable) ===\n'
printf 'Resuming from: %s\n' "$RESUME_FROM"
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/train.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    --pretrained \
    --resume "$RESUME_FROM" \
    --n_epochs "$FINETUNE_EPOCHS" \
    --init_lr "$FINETUNE_LR" \
    --weight_decay "$FINETUNE_WD" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --n_worker "$WORKERS" \
    --validation_frequency 1 \
    --display_freq 200 \
    --manual_seed "$SEED" \
    --gpus 0 \
    --save_dir "$FINETUNE_DIR"

if [[ ! -f "$FINETUNE_CKPT" ]]; then
    printf 'Error: fine-tune produced no checkpoint: %s\n' "$FINETUNE_CKPT" >&2
    exit 1
fi

printf '\n=== Step 5: FP32 accuracy of the converged variant ===\n'
srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/fp32_eval.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    --checkpoint "$FINETUNE_CKPT" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/fp32_eval" \
    --metrics-output "$RUN_DIR/${MODEL}_fp32_metrics.json"

printf '\nFloat training completed.\n'
printf 'Head warm-up checkpoint: %s\n' "$WARMUP_CKPT"
printf 'Fine-tuned checkpoint:   %s\n' "$FINETUNE_CKPT"
printf 'FP32 metrics:            %s\n' "$RUN_DIR/${MODEL}_fp32_metrics.json"
