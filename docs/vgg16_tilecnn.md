# vgg16_tilecnn: a VGG-16 variant the TileCNN fabric can run

Run log and results for the TileCNN-runnable VGG-16 variant. Every job ID,
hyperparameter, checkpoint path and accuracy number in this document was
produced on Arrhenius from the `vgg16-tilecnn-variant` branch.

## Why the architecture had to change

Stock torchvision VGG-16 is rejected by the TileCNN compiler for two
independent reasons, both fabric constraints rather than preferences:

1. **Pooling geometry.** The accelerator has exactly one pooling post-op:
   3x3, stride 2, padding 1, max. VGG's five `MaxPool2d(2, 2)` layers do not
   match it.
2. **Classifier weight footprint.** The weight buffer requires
   `ceil(Cin / 16) * K^2 <= 512` for every convolution and linear layer.
   `classifier.0` is 25088 -> 4096, needing `ceil(25088/16) * 1 = 1568`, three
   times the budget.

## The variant

Defined in [`src/fixquant/models/vgg_tilecnn.py`](../src/fixquant/models/vgg_tilecnn.py),
registered as `vgg16_tilecnn` in `get_model()`.

- **Feature stack:** `vgg16_bn`'s 13 conv 3x3 pad 1 layers with BN + ReLU,
  unchanged, except all five `MaxPool2d(2, 2)` become
  `MaxPool2d(kernel_size=3, stride=2, padding=1)`. Shape-identical at every
  stage because all inputs are even: 224 -> 112 -> 56 -> 28 -> 14 -> 7.
- **`AdaptiveAvgPool2d((7, 7))` and the pre-classifier flatten are removed.**
- **Head**, replacing `classifier.0` entirely:

  | layer | op | shape |
  |---|---|---|
  | `fc1` | `Conv2d(512, 4096, k=3, s=2, p=0)` | 512x7x7 -> 4096x3x3 |
  | | ReLU | |
  | `gap` | `AdaptiveAvgPool2d(1)` | -> 4096x1x1 |
  | | flatten, dropout | |
  | `fc2` | `Linear(4096, 4096)` + ReLU | |
  | | dropout | |
  | `fc3` | `Linear(4096, 1000)` | |

  `fc1` costs `ceil(512/16) * 9 = 288`, inside the budget. The head is new
  weights; only the 14.7M-parameter convolution stack carries over from
  `vgg16_bn`.

Batch norm is folded into the convolutions at export (`FusedConvBN`), so the
exported INT8 graph is structurally identical to the non-BN version.

`vgg16_bn_pool3` is an evaluation probe, not an export target: it swaps only
the pooling on the stock model so the pooling cost can be measured separately
from the head cost.

## Accuracy

ImageNet-1k validation, 50000 images,
`/dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b`, 224x224 centre
crop, the same pipeline `tools/qat_test.py` and `tools/deploy_eval.py` use.

| Model | Representation | top-1 | top-5 | Source |
|---|---|---|---|---|
| stock `vgg16_bn` | FP32 | **73.378** | 91.500 | job 2858888_0 |
| stock + pooling swap, no retraining | FP32 | **48.534** | 73.508 | job 2858888_1 |
| `vgg16_tilecnn` after fine-tuning | FP32 | _not run_ | | blocked, see below |
| `vgg16_tilecnn` | INT8 QAT | _not run_ | | |
| `vgg16_tilecnn` | INT8 TileCNN deploy | _not run_ | | |

73.378 for stock `vgg16_bn` matches torchvision's published 73.360, so the
evaluation pipeline is sound.

### Step 1 tripped a stop condition

The free pooling ablation costs **24.84 top-1 points** (73.378 -> 48.534) with
no retraining. The plan's stop condition is a drop of more than ~15 points, so
work paused here rather than spending GPU hours on the fine-tune.

Pooling is parameter-free: every weight in the ablation model is the
pretrained weight, so nothing was lost, something was *shifted*. A 3x3/s2/p1
max pools over nine elements where 2x2/s2 pooled over four, so each pooled
activation is systematically larger, and the shift compounds across five
stages. Downstream of each pool the convolutions' BatchNorm layers are
normalizing against running statistics gathered under the old distribution.

`tools/bn_recalibrate.py` measures how much of the drop that accounts for: it
resets every BatchNorm's running statistics and re-estimates them over 400
training batches without taking a single gradient step. Nothing learns, so the
recovered accuracy is a lower bound on what fine-tuning would recover.

| `vgg16_bn_pool3`, no retraining | top-1 | top-5 | vs stock |
|---|---|---|---|
| as evaluated (stale BN statistics) | 48.534 | 73.508 | -24.844 |
| after BN re-estimation, 25600 images, zero gradient steps | **69.170** | 89.204 | **-4.208** |

Re-estimating the statistics recovers **+20.64 points**. No weight changed, so
the 24.8-point figure was mostly an artifact of measuring a batch-normalized
network against statistics gathered under a different pooling operator; the
architectural cost of the pooling swap is about **4.2 points**, which the
fine-tune is expected to close further. Job 2859117.

## Run log

| # | Job ID | What | Status |
|---|---|---|---|
| 0 | 2858887 | `vgg16_tilecnn` smoke test, first attempt | failed on two test bugs (64x64 input too small for the head's 3x3 conv; repo root behind the environment on `sys.path`) |
| 1 | 2859061 | `vgg16_tilecnn` smoke test: structural legality + full QAT -> export -> acceptance check on random weights, plus the `not slow` regression suite | 10 passed; 74 existing tests still green |
| 2 | 2858888 | Pooling ablation array: task 0 stock `vgg16_bn` FP32, task 1 `vgg16_bn_pool3` FP32 | done |
| 3 | 2859117 | BatchNorm re-estimation on `vgg16_bn_pool3`, 400 batches x 64 images, no gradient steps | 48.534 -> 69.170 top-1 |

## Acceptance check

The TileCNN compiler is a separate repository and is not present on this
cluster, so [`tools/check_tilecnn_legality.py`](../tools/check_tilecnn_legality.py)
is the acceptance gate. It reads an exported package's `graph.json` and
`manifest.json` and verifies:

- every `maxpool2d` node has kernel `[3,3]`, stride `[2,2]`, padding `[1,1,1,1]`;
- `ceil(Cin/16) * Kh * Kw <= 512` for every `conv2d` and `linear`;
- the `gap2d` node's producer is a `conv2d`;
- every `linear` input tensor has shape `[C,1,1]`;
- kernels square and in {1,3,7}, strides equal and in {1,2}, symmetric padding,
  no dilation, groups either 1 or depthwise;
- the manifest lists and checksums every parameter, input and reference.

```bash
python tools/check_tilecnn_legality.py outputs/vgg16_tilecnn_int8_tilecnn
```
