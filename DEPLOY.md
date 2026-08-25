# Deployment Guide

*Rewritten 2026-07: earlier versions described `deploy_eval.py` as a parameter
extraction tool. Since the `a30fc02` refactor it is a pure evaluation script;
hardware artifact export lives in the exporter tools (see below).*

## Overview

Deployment has two independent steps:

1. **Evaluate the hardware model** (`tools/deploy_eval.py`) — rebuild the QAT
   model, load the trained checkpoint, convert it to the bit-exact integer
   digital twin, and measure Top-1/Top-5 on the validation set. This is the
   accuracy the FPGA will deliver (see
   [docs/tilecnn_exporter_and_digital_twin.md](docs/tilecnn_exporter_and_digital_twin.md)).
2. **Export hardware artifacts** — produce `graph.json` + int8 binaries per the
   [TileCNN Graph Handoff Specification](graph_handoff_spec.md):
   * `tools/export_tilecnn_graph.py` — full-model graph bundle (`--model`).
   * `tools/export_hw_testcases.py` — predefined ResNet-50 subgraph test cases
     for the C-simulation testbench.
   * `tools/export_mobilenet_testcases.py` — 5 real MobileNet-V2 inverted-residual
     blocks (varying stride, channel width, residual add) for the depthwise /
     pw→dw fusion path. Exports the plain sequential order (expand pw, depthwise,
     project pw, optional residual add); the TileCNN graph compiler does the
     pw→dw fusion. Uses the CLE checkpoint by default.

## Evaluating a trained model

```bash
python tools/deploy_eval.py \
    --model resnet50 \
    --dataroot /path/to/imagenet \
    --model_type tilecnn        # or "emu" (identical arithmetic, different
                                # exporter classification)
```

Key options:

* `--model` — `resnet18 | resnet50 | vgg16 | mobilenet_v2`
* `--checkpoint` — QAT checkpoint; default
  `qat_models/<model>/checkpoint/model_best.pth.tar`
* `--dataset` / `--dataroot` (or `FIXQUANT_DATA_DIR`), `--test_batch_size`,
  `--gpus`, `--manual_seed`

What it does: `QatProcessor.quantize()` → `load_qat_weights()` → `freeze()`
(BN folding must be frozen before conversion) →
`InferProcessor.convert_to_hardware_model(backend=...)` → `RunManager.validate()`.
The hardware model takes float32 images and returns float32 logits; everything
in between is int8 with TileCNN's exact two-step rounding.

## Exporting a graph bundle

For a validated model-zoo release, the exporter resolves the model,
checkpoint, and CLE setting from the release ID:

```bash
python tools/export_tilecnn_graph.py \
    --zoo-model resnet50/imagenet1k/int8-tqt@v1.0.0 \
    --out_dir outputs/resnet50_int8_tilecnn
```

Use the explicit options when exporting an unreleased checkpoint:

```bash
python tools/export_tilecnn_graph.py --model resnet50 \
    --checkpoint qat_models/resnet50/checkpoint/model_best.pth.tar \
    --out_dir outputs/resnet50_int8_tilecnn
```

Produces `manifest.json`, `graph.json`, `inputs/`, `params/`, and `refs/`.
The manifest records release identity, source checksums, the FixQuant revision,
reference preprocessing, and validation-artifact checksums. References are
recomputed using the fused bit-exact TileCNN arithmetic, with export-time
legality checks on all derived shifts. Missing quantization parameters raise
errors; nothing falls back to silent defaults.

## Verifying deployment correctness

* `python -m pytest tests/` — kernel bit-exactness, golden regression, export
  structure (fast, CPU-only).
* `fixquant.diagnostics.parity_sweep(qat_model, hw_model, input)` — per-layer
  int8 comparison between the QAT model and the hardware model; only
  ±1 LSB rounding differences are expected on the first layer, small
  propagated diffs downstream.
* Track accuracy in [docs/baselines.md](docs/baselines.md).
