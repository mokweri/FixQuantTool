# Reproducibility Baselines

Single source of truth for accuracy numbers. Every entry must state the exact
command, seed, dataset, and commit. Re-run and update after any change to the
quantization flow; a phase is only "done" when its row does not regress.

All commands use the `Obed_Cuda` conda env and `FIXQUANT_DATA_DIR` pointing to
the dataset root (default: imagenet-mini).

## Commands

```bash
# QAT training (per model)
python tools/qat_train.py --model <name> --n_epochs 10 --manual_seed 0 \
    --calib_batches 20                       # add --cle for mobilenet_v2

# QAT checkpoint evaluation (fake-quant model)
python tools/qat_test.py --model <name>
# ^ add --cle if the checkpoint was trained with --cle (mobilenet_v2); the eval
#   model must be equalized the same way or the state dict will not load.

# Hardware-model evaluation (digital twin, bit-exact integer kernels)
python tools/deploy_eval.py --model <name> --model_type tilecnn

# Per-layer sensitivity (diagnostic, no training)
python tools/layer_sensitivity.py --model <name> --eval_batches 4
```

`<name>` ∈ `resnet18 | resnet50 | vgg16 | mobilenet_v2`.

## Baseline table

Accuracy is top-1 / top-5 on the imagenet-mini validation set (full set unless
stated). "PTQ" = after calibration, before any QAT epoch.

| Model | Float | PTQ (calibrated) | QAT | Digital twin | Commit / date | Notes |
|---|---|---|---|---|---|---|
| resnet50 | | | 72.6 / 91.7 | 72.5 / 91.6 | 2026-07-13, post rounding-fix | pre-rework checkpoint (`qat_models/checkpoint/resnet50_best.pth.tar`); twin via `deploy_eval.py --model resnet50 --model_type tilecnn`. Twin now matches QAT (was 69.5/89.8 before the rounding-bias fix). |
| resnet18 | | | 69.5 / 88.7 | 68.9 / 88.5 | 2026-07-13 regression | pre-rework checkpoint (`qat_models/checkpoint/resnet18_best.pth.tar`), unfrozen/bias-free; loads after the `conv_mod.bias` loader fix. Twin tracks QAT (−0.6 top-1). |
| vgg16 | | | 69.6 / 90.3 | 69.9 / 90.3 | 2026-07-13 regression | pre-rework checkpoint (`qat_models/checkpoint/vgg16_best.pth.tar`), no BatchNorm. Twin matches QAT. |
| resnet50 | *TBD* | *TBD* | *TBD* | *TBD* | | (retrain post-rework) |
| mobilenet_v2 | 77.3 (subset) | 54.3 (subset, no CLE) | 69.9 / 89.2 | 69.7 / 89.2 | 2026-07-10 train, 2026-07-13 eval | trained with `--cle`; QAT via `qat_test.py --cle`, twin via `deploy_eval.py --cle --model_type tilecnn`. Twin now matches QAT (was 56.0/79.7 before the rounding-bias fix). Float/PTQ are quick non-CLE subset probes, not final. |

> The 2026-07 rework (see `docs/improvements_2026-07.md`) fixed a bug that froze
> all conv weights on the first QAT batch, and a digital-twin rounding
> regression from commit `a30fc02`. Numbers recorded before it (including the
> 69.5% ResNet-50 twin figure) are not comparable and must be re-measured.

## Rules

- Fixed seed (`--manual_seed 0`); note torch/torchvision versions on major re-runs.
- Record PTQ accuracy *before* training — it is the floor QAT must beat.
- After each QAT run, keep `qat_models/<model>/logs/quant_thresholds.csv` and
  `qat_models/<model>/<model>_calib_report.csv` next to the checkpoint.
