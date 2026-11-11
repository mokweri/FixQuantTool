#!/usr/bin/env bash
# Submit one N:M QAT run (scripts/jobs/qat_nm.sbatch) on Arrhenius.
#
#   scripts/nm_submit.sh MODEL PATTERN EPOCHS LR [SEED] [LAYERS.json]
#
#   scripts/nm_submit.sh resnet18 dense       2 1e-4
#   scripts/nm_submit.sh resnet18 pair24+perm 2 1e-4 0
#
#   NM_FT_EPOCHS=5 NM_FT_LR=1e-2 scripts/nm_submit.sh resnet18 pair24+perm 5 1e-5
#                  (a float fine-tune of 5 epochs from lr 1e-2, then QAT)
#
# The compute account is FIXQUANT_ACCOUNT (default the current allocation);
# extra sbatch options go in NM_SBATCH_OPTS, e.g. NM_SBATCH_OPTS="--time=06:00:00".
set -euo pipefail

if (($# < 4 || $# > 6)); then
    sed -n '2,14p' "$0" >&2
    exit 1
fi
ACCOUNT="${FIXQUANT_ACCOUNT:-naiss2026-4-1836-gpu}"
STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$STORAGE_BASE/results/qat-nm/logs"

LAYERS="${6:-}"
FT=""
((${NM_FT_EPOCHS:-0} == 0)) || FT="-ft${NM_FT_LR:-1e-2}x${NM_FT_EPOCHS}"
[[ -z "$LAYERS" ]] || LAYERS="$(realpath "$LAYERS")"

# shellcheck disable=SC2086
NM_MODEL="$1" NM_PATTERN="$2" NM_EPOCHS="$3" NM_LR="$4" NM_SEED="${5:-0}" NM_LAYERS="$LAYERS" \
    sbatch --account="$ACCOUNT" ${NM_SBATCH_OPTS:-} \
    --job-name="nm-$1-${2/+/_}${FT}-s${5:-0}" \
    --export=ALL \
    "$HERE/jobs/qat_nm.sbatch"
