# Reproducible Arrhenius environment

Arrhenius login nodes are x86_64, while GPU nodes are ARM (`aarch64`). A Conda
environment copied from a laptop or created on the login node therefore cannot
run on a GH200 GPU node.

FixQuantTool uses a pinned NVIDIA PyTorch Apptainer image and a project-specific
virtual environment layered on top of it:

- container: `nvcr.io/nvidia/pytorch:25.09-py3`
- compute allocation: `naiss2025-22-1409-gpu`
- project storage: `/nobackup/proj/disk/naiss2024-22-1352/personal/mogaka`
- environment: `software/venvs/fixquant-ngc-25.09-py3`

## One-time setup

From the Arrhenius login node:

```bash
cd ~/work/projects/FixQuantTool
bash scripts/setup_arrhenius_env.sh
```

The script submits a one-GPU setup job. The submitted job downloads the ARM
container, creates the environment, installs the package in editable mode,
checks imports and CUDA, and runs the fast tests. The command prints the Slurm
job ID and log path.

The script is idempotent: rerunning it reuses the container and environment,
then refreshes the editable install and validates the result. To replace the
container image with a clean copy:

```bash
bash scripts/setup_arrhenius_env.sh --force-container
```

## Routine interactive use

Request a GPU shell:

```bash
interactive \
    -A naiss2025-22-1409-gpu \
    -p gpu \
    --gpus 1 \
    -n 1 \
    -c 8 \
    -t 01:00:00
```

Then run commands through the launcher:

```bash
cd ~/work/projects/FixQuantTool
scripts/run_arrhenius.sh python -m pytest -q -m "not slow"
scripts/run_arrhenius.sh python -c \
    'import torch; print(torch.cuda.get_device_name(0))'
```

The same launcher can be called from Slurm batch scripts. It binds project
storage and `/dataset`, activates the external environment, changes to the
repository root, and executes the requested command.

## Updating code and dependencies

After pulling repository changes, rerun the setup script if `pyproject.toml` or
the environment setup changed:

```bash
git pull --ff-only
bash scripts/setup_arrhenius_env.sh
```

Each successful setup records `pip freeze` output under
`~/work/software/locks/`. This is an audit snapshot of the resolved packages;
the pinned container tag supplies the reproducible CUDA and PyTorch base.

## ImageNet on Arrhenius

FixQuantTool reads Arrhenius's Hugging Face Arrow copy directly. It is
memory-mapped from `/dataset`; do not extract or copy the ZIP archive into the
project directory. The loader also remains compatible with a conventional
`train/<class>/...` and `val/<class>/...` ImageFolder tree.

The setup script installs the pinned Arrow reader. If the environment was
created before this support was added, rerun the idempotent setup once:

```bash
bash scripts/setup_arrhenius_env.sh
```

Submit the one-batch integration check from the login node:

```bash
cd ~/work/projects/FixQuantTool
sbatch scripts/jobs/imagenet_smoke.sbatch
```

The job requests one GPU, reads both ImageNet splits, and performs one
MobileNetV2 forward/backward optimization step. Its log is written under
`~/work/results/` as `fixquant-imagenet-smoke-<job-id>.out`.

For training commands, use:

```bash
--dataroot /dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b
```

## MobileNetV2 QAT job

Submit the production single-GPU job from the login node:

```bash
cd ~/work/projects/FixQuantTool
sbatch scripts/jobs/qat_mobilenet_imagenet.sbatch
```

The default run uses one GH200, 16 CPU cores, 32 GiB host memory, batch size
64, 10 QAT epochs, and a 24-hour limit. Each run writes checkpoints, threshold
logs, and the calibration report to
`~/work/results/qat/mobilenet_v2/<job-id>/mobilenet_v2/`. Slurm output is
written to `~/work/results/fixquant-qat-mobilenet-<job-id>.out`.

Training settings can be overridden without editing the script. For example:

```bash
sbatch --export=ALL,FIXQUANT_EPOCHS=5,FIXQUANT_TRAIN_BATCH_SIZE=32 \
    scripts/jobs/qat_mobilenet_imagenet.sbatch
```

Slurm resources are overridden with normal `sbatch` options. For example:

```bash
sbatch --time=12:00:00 --cpus-per-task=8 \
    --export=ALL,FIXQUANT_WORKERS=6 \
    scripts/jobs/qat_mobilenet_imagenet.sbatch
```

## VGG16 and ResNet QAT sweep

The model-sweep job trains VGG16, ResNet-18, and ResNet-50 as three Slurm
array tasks. Each task uses one GH200 and writes to an independent result
directory. Defaults are five epochs with model-specific batch sizes:

| Array task | Model | Train batch | Validation batch |
|---:|---|---:|---:|
| 0 | VGG16 | 32 | 64 |
| 1 | ResNet-18 | 128 | 256 |
| 2 | ResNet-50 | 64 | 128 |

Submit all three concurrently:

```bash
sbatch scripts/jobs/qat_imagenet_model_sweep.sbatch
```

To limit the sweep to one running GPU job at a time:

```bash
sbatch --array=0-2%1 scripts/jobs/qat_imagenet_model_sweep.sbatch
```

Individual models can be selected by array index:

```bash
sbatch --array=0 scripts/jobs/qat_imagenet_model_sweep.sbatch  # VGG16
sbatch --array=1 scripts/jobs/qat_imagenet_model_sweep.sbatch  # ResNet-18
sbatch --array=2 scripts/jobs/qat_imagenet_model_sweep.sbatch  # ResNet-50
```

Each model receives separate `latest.pth.tar` and `model_best.pth.tar`
checkpoints. If validation is still improving at epoch five, increase the
total with `--export=ALL,FIXQUANT_EPOCHS=<epochs>` for a subsequent run.
Successful future jobs also register candidates under the repository-local
`model_zoo/` registry.
See [model_zoo.md](model_zoo.md) for automated validation and manual promotion.
