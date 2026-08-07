# Reproducibility Baselines

Single source of truth for accuracy numbers and completed experiment runs.
Every entry must state the exact command, seed, dataset, and commit. Re-run and
update after any change to the quantization flow; a phase is only "done" when
its row does not regress.

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

Accuracy is top-1 / top-5. Legacy rows use the imagenet-mini validation set;
the dataset and scope for newer runs are stated explicitly in their notes and
experiment record. "PTQ" = after calibration, before any QAT epoch.

| Model | Float | PTQ (calibrated) | QAT | Digital twin | Commit / date | Notes |
|---|---|---|---|---|---|---|
| resnet50 | | | 72.6 / 91.7 | 72.5 / 91.6 | 2026-07-13, post rounding-fix | pre-rework checkpoint (`qat_models/checkpoint/resnet50_best.pth.tar`); twin via `deploy_eval.py --model resnet50 --model_type tilecnn`. Twin now matches QAT (was 69.5/89.8 before the rounding-bias fix). |
| resnet18 | | | 69.5 / 88.7 | 68.9 / 88.5 | 2026-07-13 regression | pre-rework checkpoint (`qat_models/checkpoint/resnet18_best.pth.tar`), unfrozen/bias-free; loads after the `conv_mod.bias` loader fix. Twin tracks QAT (−0.6 top-1). |
| vgg16 | | | 69.6 / 90.3 | 69.9 / 90.3 | 2026-07-13 regression | pre-rework checkpoint (`qat_models/checkpoint/vgg16_best.pth.tar`), no BatchNorm. Twin matches QAT. |
| resnet50 | *TBD* | *TBD* | *TBD* | *TBD* | | (retrain post-rework) |
| mobilenet_v2 | 77.3 (subset) | 54.3 (subset, no CLE) | 69.9 / 89.2 | 69.7 / 89.2 | 2026-07-10 train, 2026-07-13 eval | trained with `--cle`; QAT via `qat_test.py --cle`, twin via `deploy_eval.py --cle --model_type tilecnn`. Twin now matches QAT (was 56.0/79.7 before the rounding-bias fix). Float/PTQ are quick non-CLE subset probes, not final. |
| mobilenet_v2 | *TBD* | *TBD* | **71.568 / 90.352** | **70.954 / 90.002** | `cdaffe8`, 2026-08-05 train / 2026-08-07 eval | Full ImageNet-1k validation set; Arrhenius GH200; CLE; final epoch-10 checkpoint. Twin delta: −0.614 top-1 / −0.350 top-5 percentage points. |

> The 2026-07 rework (see `docs/improvements_2026-07.md`) fixed a bug that froze
> all conv weights on the first QAT batch, and a digital-twin rounding
> regression from commit `a30fc02`. Numbers recorded before it (including the
> 69.5% ResNet-50 twin figure) are not comparable and must be re-measured.

## Rules

- Fixed seed (`--manual_seed 0`); note torch/torchvision versions on major re-runs.
- Record PTQ accuracy *before* training — it is the floor QAT must beat.
- After each QAT run, keep `qat_models/<model>/logs/quant_thresholds.csv` and
  `qat_models/<model>/<model>_calib_report.csv` next to the checkpoint.
- Training writes `latest.pth.tar` after every epoch and updates
  `model_best.pth.tar` only when validation top-1 improves. Both files contain
  the full training state and are written atomically.

## Experiment record: MobileNetV2 on full ImageNet-1k

### Training

| Field | Value |
|---|---|
| Training date | 2026-08-05 |
| Slurm job | `904116` (`COMPLETED`, 09:19:47) |
| Git commit | `cdaffe86565391e46982aec1f2d3f9dcff91c885` |
| Hardware | 1× NVIDIA GH200 120GB, 16 CPUs, 32 GiB host memory |
| Dataset | Arrhenius ImageNet-1k Arrow: 1,281,167 train / 50,000 validation |
| Model | torchvision MobileNetV2 pretrained weights |
| Quantization | TQT, INT8, 20 calibration batches, MSE scope 5 |
| Preparation | BN fold, cross-layer equalization, ReLU6→ReLU (`--cle`) |
| Training | 10 epochs, batch 64, initial weight LR `1e-5`, quantizer LR `1e-2` |
| Threshold freeze | 70% of training |
| Seed | 0 |

Training artifacts:

```text
results/qat/mobilenet_v2/904116/mobilenet_v2/
├── checkpoint/model_best.pth.tar
├── logs/quant_thresholds.csv
└── mobilenet_v2_calib_report.csv
```

### Evaluation

Evaluation job `949845` ran the full 50,000-image validation split and
completed in 1 minute 43 seconds.

| Representation | Loss | Top-1 | Top-5 |
|---|---:|---:|---:|
| QAT fake-quant model | 1.150242 | 71.568% | 90.352% |
| TileCNN hardware-exact digital twin | 1.175469 | 70.954% | 90.002% |
| Digital-twin delta | +0.025227 | −0.614 pp | −0.350 pp |

The evaluation used:

```bash
sbatch scripts/jobs/eval_mobilenet_imagenet.sbatch
```

The historical `model_best.pth.tar` from job `904116` contains the final
epoch-10 state (`epoch=9`, zero-based), not the peak epoch-4 state. The old
checkpoint routine overwrote the best file every epoch, so the peak weights
cannot be recovered. Checkpoint handling was corrected after this run; future
experiments preserve independent `latest.pth.tar` and `model_best.pth.tar`
files.

## New experiment template

For each run, record:

- date, Slurm job ID, Git commit, hardware, and wall time;
- dataset name, version/path, and train/validation sample counts;
- model initialization, preprocessing, quantization, and seed;
- epochs, batch sizes, learning rates, calibration, and freeze settings;
- float/PTQ/QAT/digital-twin loss, top-1, and top-5;
- accuracy deltas between QAT and deployment representations;
- checkpoint, calibration report, threshold log, and Slurm log paths;
- known caveats or deviations from the standard workflow.
