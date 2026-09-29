# FP32 reference accuracy

Floating-point top-1/top-5 accuracy for the pretrained model each release
starts from, measured on the same full ImageNet-1k validation set (50,000
images) and data pipeline as the released QAT and integer-twin metrics.

| Model | Top-1 | Top-5 |
|---|---:|---:|
| resnet18 | 69.754 | 89.074 |
| resnet50 | 80.336 | 95.122 |
| vgg16 | 71.580 | 90.394 |
| mobilenet_v2 | 72.014 | 90.614 |
| efficientnet_lite0 | 75.396 | 92.500 |
| vgg16_tilecnn | 71.978 | 90.662 |

Produced on Arrhenius by `scripts/jobs/eval_fp32_imagenet.sbatch`, which runs
[`tools/fp32_eval.py`](../../tools/fp32_eval.py) for every released model:
on 2026-09-22 for the first four, and on 2026-09-29 (job 3122655, task 4) for
efficientnet_lite0, from the converted timm weights. vgg16_tilecnn has no
torchvision checkpoint; its figure is the 2026-09-23 evaluation of the
fine-tuned float checkpoint from job 2859325, the model its QAT started from.
Each JSON file records its dataset path, sample count and weights.

These files are reference measurements, not part of any release: release
directories are immutable and checksum-inventoried, so FP32 results are kept
here rather than added to them.
