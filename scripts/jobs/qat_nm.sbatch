#!/usr/bin/env bash
# QAT with a fixed N:M weight mask (or the dense reference with the same
# schedule). Submit through scripts/nm_submit.sh, which supplies the account
# and the settings below as environment variables:
#
#   NM_MODEL      resnet18 | resnet50 | mobilenet_v2 | efficientnet_lite0
#   NM_PATTERN    dense | filter24 | pair24 | oc_n16 | filter24+perm | pair24+perm
#   NM_LAYERS     optional JSON layer list (selective pruning); default all
#   NM_EPOCHS, NM_LR, NM_SEED, NM_TAG (optional label for the run directory)
#
# The flags per network are those of the v1.0.0 releases: CLE for
# MobileNetV2, CLE keeping ReLU6 for EfficientNet-Lite0, none for the ResNets.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fixquant-qat-nm
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka/results/qat-nm/logs/%x-%j.out

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
DATASET="${FIXQUANT_DATA_DIR:-/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b}"
TORCH_CACHE="${FIXQUANT_TORCH_CACHE:-$STORAGE_BASE/software/torch-cache}"

MODEL="${NM_MODEL:?NM_MODEL is required}"
PATTERN="${NM_PATTERN:?NM_PATTERN is required (dense for the reference)}"
LAYERS="${NM_LAYERS:-}"
EPOCHS="${NM_EPOCHS:?NM_EPOCHS is required}"
INIT_LR="${NM_LR:?NM_LR is required}"
SEED="${NM_SEED:-0}"
QUANTIZER_LR="${NM_QUANTIZER_LR:-1e-2}"
WORKERS="${NM_WORKERS:-14}"
CALIB_BATCHES="${NM_CALIB_BATCHES:-20}"
CALIB_SCOPE="${NM_CALIB_SCOPE:-5}"

case "$MODEL" in
    resnet18)           TRAIN_BS=128; TEST_BS=256; MODEL_FLAGS=() ;;
    resnet50)           TRAIN_BS=64;  TEST_BS=128; MODEL_FLAGS=() ;;
    mobilenet_v2)       TRAIN_BS=64;  TEST_BS=128; MODEL_FLAGS=(--cle) ;;
    efficientnet_lite0) TRAIN_BS=64;  TEST_BS=128; MODEL_FLAGS=(--cle --cle_keep_relu6) ;;
    *) printf 'Error: unsupported model %s\n' "$MODEL" >&2; exit 1 ;;
esac
TRAIN_BS="${NM_TRAIN_BATCH_SIZE:-$TRAIN_BS}"
TEST_BS="${NM_TEST_BATCH_SIZE:-$TEST_BS}"

NM_FLAGS=()
if [[ "$PATTERN" != "dense" ]]; then
    NM_FLAGS=(--nm_pattern "$PATTERN")
    if [[ -n "$LAYERS" ]]; then
        [[ -f "$LAYERS" ]] || { printf 'Error: layer list not found: %s\n' "$LAYERS" >&2; exit 1; }
        NM_FLAGS+=(--nm_layers "$LAYERS")
    fi
elif [[ -n "$LAYERS" ]]; then
    printf 'Error: NM_LAYERS given with the dense pattern\n' >&2; exit 1
fi

TAG="${NM_TAG:-${PATTERN/+/_}-lr${INIT_LR}-e${EPOCHS}-s${SEED}}"
RUN_DIR="$STORAGE_BASE/results/qat-nm/$MODEL/$TAG/$SLURM_JOB_ID"

[[ -d "$REPO/.git" ]] || { printf 'Error: repository not found: %s\n' "$REPO" >&2; exit 1; }
[[ -f "$DATASET/dataset_dict.json" ]] || { printf 'Error: ImageNet not found: %s\n' "$DATASET" >&2; exit 1; }
if [[ "$MODEL" == efficientnet_lite0 && ! -f "$REPO/checkpoints/efficientnet_lite0_ra_in1k.pth" ]]; then
    printf 'Error: converted EfficientNet-Lite0 weights missing under %s/checkpoints\n' "$REPO" >&2; exit 1
fi
((WORKERS <= SLURM_CPUS_PER_TASK)) || { printf 'Error: workers exceed CPUs\n' >&2; exit 1; }

mkdir -p "$RUN_DIR" "$TORCH_CACHE"
cd "$REPO"

export APPTAINERENV_TORCH_HOME="$TORCH_CACHE"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_PYTHONHASHSEED="$SEED"

printf 'FixQuant N:M QAT\n'
printf 'Job: %s on %s, account %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "${SLURM_JOB_ACCOUNT:-?}"
printf 'Git commit: %s (%s)\n' "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)"
printf 'Model / pattern / layers: %s / %s / %s\n' "$MODEL" "$PATTERN" "${LAYERS:-all}"
printf 'Epochs / lr / seed: %s / %s / %s\n' "$EPOCHS" "$INIT_LR" "$SEED"
printf 'Train/test batch: %s/%s\n' "$TRAIN_BS" "$TEST_BS"
printf 'Run directory: %s\n' "$RUN_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun --kill-on-bad-exit=1 scripts/run_arrhenius.sh \
    python tools/qat_train.py \
    --dataset imagenet \
    --dataroot "$DATASET" \
    --model "$MODEL" \
    "${MODEL_FLAGS[@]}" \
    "${NM_FLAGS[@]}" \
    --train_batch_size "$TRAIN_BS" \
    --test_batch_size "$TEST_BS" \
    --n_worker "$WORKERS" \
    --n_epochs "$EPOCHS" \
    --init_lr "$INIT_LR" \
    --quantizer_lr "$QUANTIZER_LR" \
    --threshold_freeze_frac 0.7 \
    --calib_batches "$CALIB_BATCHES" \
    --calib_scope "$CALIB_SCOPE" \
    --validation_frequency 1 \
    --display_freq 500 \
    --manual_seed "$SEED" \
    --gpus 0 \
    --save_dir "$RUN_DIR"

printf 'Done: %s/%s\n' "$RUN_DIR" "$MODEL"
