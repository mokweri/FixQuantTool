#!/bin/env bash
# =============================================================================
# FixQuantTool QAT training on Arrhenius (NAISS) — single-GPU GH200 job
#
# Usage:
#   sbatch scripts/jobscript_arrhenius.sh                        # mobilenet_v2 + CLE
#   MODEL=resnet50 EXTRA_ARGS="" sbatch scripts/jobscript_arrhenius.sh
#
# Before first use (see docs/arrhenius_gpu_guide.md):
#   * replace <project> below with your SUPR project id (account MUST end in -gpu)
#   * set PROJ_DIR to your project storage directory
#   * build the environment ONCE on a GPU node (compute nodes are aarch64/ARM;
#     x86 envs and default-PyPI torch wheels do NOT work):
#         interactive -p gpu --gpus 1 -A <project>-gpu -t 01:00:00
#         apptainer pull $PROJ_DIR/containers/pytorch.sif \
#                        docker://nvcr.io/nvidia/pytorch:25.06-py3
#   * verify with the repo test suite (CPU-only, seconds):
#         apptainer exec pytorch.sif python -m pytest FixQuantTool/tests -m "not slow"
# =============================================================================

#SBATCH -A naiss2026-XX-YYY-gpu       # <project>-gpu account  [EDIT ME]
#SBATCH -p gpu                        # GPU partition (4x GH200 per node)
#SBATCH -n 1                          # single task: qat_train.py is single-process
#SBATCH -c 16                         # Grace cores for dataloading/calibration (max 72 per GPU)
#SBATCH --gpus 1                      # one GH200 (96 GB HBM3)
#SBATCH -t 08:00:00                   # walltime (partition max is 3-00:00:00)
#SBATCH -J fixquant-qat
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

# ----------------------------- configuration --------------------------------
PROJ_DIR=/nobackup/proj/disk/naiss2026-XX-YYY          # [EDIT ME] project storage
REPO=$PROJ_DIR/FixQuantTool
CONTAINER=$PROJ_DIR/containers/pytorch.sif             # aarch64 NGC PyTorch image
DATASET_TAR=$PROJ_DIR/datasets/imagenet-mini.tar       # dataset archive on Lustre

MODEL=${MODEL:-mobilenet_v2}
EXTRA_ARGS=${EXTRA_ARGS:---cle}                        # CLE recommended for mobilenet_v2
EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-128}

# ------------------------- stage data to node-local NVMe --------------------
# Each node has ~1.8 TB local NVMe; reading the dataset from it avoids
# hammering Lustre with small-file I/O. TMPDIR fallback: verify the site's
# scratch variable on first login (docs/arrhenius_gpu_guide.md §8).
SCRATCH=${TMPDIR:-/tmp}/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
echo "[jobscript] staging dataset to $SCRATCH"
tar -xf "$DATASET_TAR" -C "$SCRATCH"
export FIXQUANT_DATA_DIR=$SCRATCH/imagenet-mini

# ------------------------------- run ----------------------------------------
cd "$REPO"
echo "[jobscript] node=$(hostname) arch=$(uname -m) gpus=${SLURM_GPUS:-?}"

srun apptainer exec --nv \
    --bind "$PROJ_DIR","$SCRATCH" \
    "$CONTAINER" \
    python tools/qat_train.py \
        --model "$MODEL" \
        --dataset imagenet \
        --n_epochs "$EPOCHS" \
        --train_batch_size "$BATCH" \
        --test_batch_size "$BATCH" \
        --init_lr 1e-5 \
        --quantizer_lr 1e-2 \
        --calib_batches 20 \
        --threshold_freeze_frac 0.7 \
        --n_worker "$SLURM_CPUS_PER_TASK" \
        --manual_seed 0 \
        --save_dir "$PROJ_DIR/qat_models" \
        $EXTRA_ARGS

# Checkpoints land in $PROJ_DIR/qat_models/$MODEL/checkpoint/ (RunManager saves
# every epoch, so a job hitting the walltime can be resumed from the last one).
# Per-epoch threshold logs: $PROJ_DIR/qat_models/$MODEL/logs/quant_thresholds.csv
echo "[jobscript] done"
