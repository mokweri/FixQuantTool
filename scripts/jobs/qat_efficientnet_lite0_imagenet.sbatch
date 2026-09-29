#!/usr/bin/env bash
#SBATCH --account=naiss2025-22-1409-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fixquant-qat-effnetlite0
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/%x-%j.out

# INT8 TQT quantization-aware training for efficientnet_lite0 with cross-layer
# equalization that keeps ReLU6 (profile int8-tqt-cle-relu6). Starts from the
# converted timm weights, which must be copied to checkpoints/ beforehand; they
# are not version-controlled and this job does not reconvert them. See
# docs/efficientnet_lite0.md.

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"
ZOO_ROOT="${FIXQUANT_ZOO_ROOT:-$REPO/model_zoo}"
AUTO_REGISTER="${FIXQUANT_AUTO_REGISTER:-1}"

MODEL=efficientnet_lite0
PRETRAINED="$REPO/checkpoints/efficientnet_lite0_ra_in1k.pth"
PRETRAINED_SHA256=5b0e7a075e51a77ac9ed39def38ee0ef1de11288a7901aa2c4dcbe6038b85a22
RUN_DIR="${FIXQUANT_RUN_DIR:-$STORAGE_BASE/results/qat/$MODEL/$SLURM_JOB_ID}"

# The ImageNet-mini development run peaked at epoch 4 of 30 (~140k images);
# one full-ImageNet epoch is ~1.28M images, so a few epochs suffice.
EPOCHS="${FIXQUANT_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${FIXQUANT_TRAIN_BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${FIXQUANT_TEST_BATCH_SIZE:-128}"
WORKERS="${FIXQUANT_WORKERS:-12}"
CALIB_BATCHES="${FIXQUANT_CALIB_BATCHES:-20}"
CALIB_SCOPE="${FIXQUANT_CALIB_SCOPE:-5}"
INIT_LR="${FIXQUANT_INIT_LR:-1e-5}"
QUANTIZER_LR="${FIXQUANT_QUANTIZER_LR:-1e-2}"
SEED="${FIXQUANT_SEED:-0}"

if [[ ! -d "$REPO/.git" ]]; then
    printf 'Error: repository not found at %s\n' "$REPO" >&2
    exit 1
fi
if [[ ! -f "$DATASET/dataset_dict.json" ]]; then
    printf 'Error: ImageNet Arrow dataset not found at %s\n' "$DATASET" >&2
    exit 1
fi
if [[ ! -f "$PRETRAINED" ]]; then
    printf 'Error: converted timm weights not found: %s\n' "$PRETRAINED" >&2
    printf 'Copy checkpoints/efficientnet_lite0_ra_in1k.pth from the machine that converted it.\n' >&2
    exit 1
fi
if [[ "$(sha256sum "$PRETRAINED" | cut -d' ' -f1)" != "$PRETRAINED_SHA256" ]]; then
    printf 'Error: %s does not match the recorded SHA-256 %s\n' \
        "$PRETRAINED" "$PRETRAINED_SHA256" >&2
    exit 1
fi
if ((WORKERS > SLURM_CPUS_PER_TASK)); then
    printf 'Error: FIXQUANT_WORKERS (%s) exceeds allocated CPUs (%s).\n' \
        "$WORKERS" "$SLURM_CPUS_PER_TASK" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_PYTHONHASHSEED="$SEED"
export APPTAINERENV_FIXQUANT_ZOO_ROOT="$ZOO_ROOT"

printf 'FixQuant %s INT8 TQT QAT (CLE keeping ReLU6)\n' "$MODEL"
printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$SLURMD_NODENAME"
printf 'Repository: %s\n' "$REPO"
printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
printf 'Pretrained weights: %s (sha256 verified)\n' "$PRETRAINED"
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
    --cle \
    --cle_keep_relu6 \
    --validation_frequency 1 \
    --display_freq 200 \
    --manual_seed "$SEED" \
    --gpus 0 \
    --save_dir "$RUN_DIR"

if [[ "$AUTO_REGISTER" == "1" ]]; then
    # The profile (int8-tqt-cle-relu6) is derived from the run manifest.
    printf 'Registering model-zoo candidate...\n'
    srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
        python tools/model_zoo.py --root "$ZOO_ROOT" register \
        "$RUN_DIR/$MODEL" \
        --model "$MODEL" \
        --dataset-name imagenet1k \
        --dataset-path "$DATASET" \
        --cle
fi

printf 'QAT job completed successfully. Results: %s/%s\n' "$RUN_DIR" "$MODEL"
