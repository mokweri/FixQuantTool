#!/usr/bin/env bash

# Collect every vgg16_tilecnn artifact into one self-describing directory.
#
# The artifacts are produced by four different jobs and land in four different
# places under $STORAGE_BASE. This gathers them so a single rsync moves what a
# given destination needs, and writes SHA256SUMS so the far side can prove the
# transfer was clean.
#
# Files are hard-linked when the destination is on the same filesystem, so the
# bundle normally costs no extra space; it falls back to copying otherwise.
#
# Usage: scripts/collect_vgg16_tilecnn_handoff.sh [destination]

set -euo pipefail

STORAGE_BASE="${FIXQUANT_STORAGE_BASE:-/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka}"
REPO="${FIXQUANT_REPO:-$STORAGE_BASE/projects/FixQuantTool}"
RELEASE_ID="vgg16_tilecnn/imagenet1k/int8-tqt@v1.0.0"
VERSION="v1.0.0"
DEST="${1:-$STORAGE_BASE/handoff/vgg16_tilecnn-$VERSION}"

FLOAT_RUN="${FIXQUANT_FLOAT_RUN:-$STORAGE_BASE/results/float/vgg16_tilecnn/2859325}"
QAT_RUN="${FIXQUANT_QAT_RUN:-$STORAGE_BASE/results/qat/vgg16_tilecnn/2884234/vgg16_tilecnn}"
ABLATION="${FIXQUANT_ABLATION:-$STORAGE_BASE/results/eval/pool_ablation}"
PACKAGE="${FIXQUANT_EXPORT_DIR:-$REPO/outputs/vgg16_tilecnn_int8_tilecnn}"
RELEASE_DIR="$REPO/model_zoo/releases/vgg16_tilecnn/imagenet1k/int8-tqt/$VERSION"

# Hard-link when we can (same filesystem, zero extra bytes), copy when we cannot.
place() {
    local src="$1" dst="$2"
    if [[ ! -e "$src" ]]; then
        printf 'Error: missing artifact: %s\n' "$src" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    rm -rf -- "$dst"
    if [[ -d "$src" ]]; then
        cp -al "$src" "$dst" 2>/dev/null || cp -a "$src" "$dst"
    else
        ln "$src" "$dst" 2>/dev/null || cp -a "$src" "$dst"
    fi
}

printf 'Collecting vgg16_tilecnn %s into %s\n' "$VERSION" "$DEST"
rm -rf -- "$DEST"
mkdir -p "$DEST"

# --- INT8: what the accelerator consumes -------------------------------------
place "$PACKAGE" "$DEST/int8/modelpackage"

# --- INT8: the zoo release (metadata + QAT checkpoint) -----------------------
place "$RELEASE_DIR" "$DEST/int8/release"

# --- FP32: for evaluation and for re-running QAT -----------------------------
place "$FLOAT_RUN/finetune/vgg16_tilecnn/checkpoint/model_best.pth.tar" \
      "$DEST/fp32/vgg16_tilecnn_fp32.pth.tar"
place "$FLOAT_RUN/vgg16_tilecnn_fp32_metrics.json" \
      "$DEST/fp32/vgg16_tilecnn_fp32_metrics.json"

# --- FP32: the pooling ablation, so the architecture cost stays attributable --
for metrics in "$ABLATION"/*/vgg16_bn_fp32_metrics.json \
               "$ABLATION"/*/vgg16_bn_pool3_fp32_metrics.json \
               "$ABLATION"/bn_recal/*/vgg16_bn_pool3_bnrecal_metrics.json; do
    [[ -e "$metrics" ]] && place "$metrics" "$DEST/fp32/ablation/$(basename "$metrics")"
done

cat > "$DEST/README.md" <<README
# vgg16_tilecnn $VERSION handoff

Release \`$RELEASE_ID\`, collected $(date -u +%Y-%m-%dT%H:%M:%SZ) from
FixQuant commit \`$(git -C "$REPO" rev-parse HEAD)\`.

VGG-16 rebuilt to satisfy the TileCNN fabric: 3x3/s2/p1 pooling throughout and
a convolutional classifier head that fits the weight buffer. See
\`docs/vgg16_tilecnn.md\` in the repository for the full run log.

| Number | top-1 | top-5 |
|---|---:|---:|
| stock vgg16_bn FP32 | 73.378 | 91.500 |
| stock + pooling swap, no retraining | 48.534 | 73.508 |
| vgg16_tilecnn FP32, fine-tuned | 71.978 | 90.662 |
| vgg16_tilecnn INT8 QAT | 72.252 | 90.762 |
| vgg16_tilecnn INT8 TileCNN deploy | 72.204 | 90.706 |

## What to take where

\`int8/modelpackage/\` (53 MB) is the only thing the accelerator needs: the INT8
graph, all 32 parameter blobs, one preprocessed validation input and the
bit-exact integer reference output it must reproduce. Keep the directory layout
intact -- the paths inside \`graph.json\` are relative to it.

\`int8/release/\` is the model-zoo release: metadata, evaluation reports, the
quantizer threshold log, and the 624 MB QAT checkpoint. Needed only to re-run
\`qat_test.py\` / \`deploy_eval.py\` or to re-export the package.

\`fp32/\` is the fine-tuned float checkpoint and its metrics, plus the pooling
ablation numbers that separate the architecture cost from the quantization
cost. Needed for float evaluation and to re-run QAT from scratch.

## Verify after transfer

\`\`\`bash
sha256sum -c SHA256SUMS
python tools/check_tilecnn_legality.py int8/modelpackage
\`\`\`

The legality checker re-reads every artifact and recomputes every manifest
checksum, so a clean run also proves the transfer was intact.

## Re-run the evaluations

\`\`\`bash
# FP32
python tools/fp32_eval.py --model vgg16_tilecnn \\
    --checkpoint fp32/vgg16_tilecnn_fp32.pth.tar --dataroot <imagenet>

# INT8 fake-quant, and the bit-exact integer twin
python tools/qat_test.py    --zoo-model $RELEASE_ID
python tools/deploy_eval.py --zoo-model $RELEASE_ID --model_type tilecnn
\`\`\`
README

printf 'Writing SHA256SUMS...\n'
( cd "$DEST" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS )

printf '\nCollected %s in %s\n' "$(du -sh --apparent-size "$DEST" | cut -f1)" "$DEST"
printf 'Files: %s\n' "$(grep -c '' "$DEST/SHA256SUMS")"
printf '\nTo move only what the accelerator needs (53 MB):\n'
printf '  rsync -avP %s/int8/modelpackage/ <tilecnn-host>:<path>/\n' "$DEST"
printf '\nTo move everything:\n'
printf '  rsync -avP %s/ <host>:<path>/\n' "$DEST"
