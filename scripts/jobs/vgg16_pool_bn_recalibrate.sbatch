#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=fixquant-vgg16-pool-bnrecal
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

# Follow-up to the pooling ablation: how much of the drop is stale BatchNorm
# running statistics rather than damaged features? Re-estimates the statistics
# over a few hundred training batches without a single gradient step. No
# weights change, so this bounds from below what fine-tuning will recover.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"
WORKERS="${FIXQUANT_WORKERS:-12}"
BATCHES="${FIXQUANT_BN_BATCHES:-400}"
RUN_DIR="${FIXQUANT_EVAL_DIR:-$STORAGE_BASE/results/eval/pool_ablation/bn_recal/$SLURM_JOB_ID}"

mkdir -p "$RUN_DIR" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'FixQuant VGG-16 pooling ablation: BatchNorm re-estimation\n'
printf 'Slurm job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$SLURMD_NODENAME"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Re-estimation batches: %s\n' "$BATCHES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/bn_recalibrate.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model vgg16_bn_pool3 \
    --batches "$BATCHES" \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --n_worker "$WORKERS" \
    --gpus 0 \
    --save_dir "$RUN_DIR/run" \
    --metrics-output "$RUN_DIR/vgg16_bn_pool3_bnrecal_metrics.json"

printf '\nBatchNorm re-estimation completed.\n'
