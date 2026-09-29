# efficientnet_lite0: EfficientNet-Lite0 for DeepTile

EfficientNet-Lite is the edge variant of EfficientNet: no squeeze-and-excitation,
and ReLU6 in place of swish. Both changes suit fixed-point accelerators, which
is why it was chosen as the first contemporary edge CNN for DeepTile.

## The model

`src/fixquant/models/efficientnet_lite.py` defines the network from plain torch
modules. It is the topology of timm's `efficientnet_lite0`:

- a 3x3 stride-2 stem, then 16 blocks in seven stages;
- depthwise kernels of 3x3 (seven layers) and 5x5 (nine layers);
- nine identity residuals;
- a 1x1 head to 1280 channels, global average pooling, and a 1000-way classifier;
- 4,652,008 parameters, and a ReLU6 after every activated batch norm.

timm's TensorFlow port, `tf_efficientnet_lite0`, pads asymmetrically and is not
DeepTile-legal. The timm model used here pads symmetrically.

### Why not use timm's model class

timm fuses each batch norm and its activation into one `BatchNormAct2d` module,
a subclass of `nn.BatchNorm2d` whose `forward` also applies the ReLU6. FixQuant
folds every `(Conv2d, BatchNorm2d)` pair into the convolution, so it would fold
that module away and silently drop the ReLU6. The preflight check would not
catch it, because the module passes the `BatchNorm2d` type test. Here every
batch norm is a plain `nn.BatchNorm2d` with its own `nn.ReLU6`, and a test
asserts it.

## Pretrained weights

The weights are timm's `efficientnet_lite0.ra_in1k`, converted once:

```bash
python tools/convert_timm_efficientnet_lite.py
```

The tool needs timm and network access; training does not. Attribute names
match timm's, so the state dict loads with strict key checking. The converted
model must reproduce timm's logits on seeded inputs; the recorded difference is
exactly zero. The checkpoint is written to
`checkpoints/efficientnet_lite0_ra_in1k.pth` (sha256 `5b0e7a075e51...`, not
version-controlled) together with its source and the verification result.
Copy it to other machines rather than converting again.

## Quantization recipe

Per-tensor power-of-two scales struggle with depthwise layers whose weight
ranges differ strongly between channels, which is why MobileNetV2 is trained
with cross-layer equalization (CLE). FixQuant's CLE normally replaces ReLU6 with
ReLU, because CLE rescaling is exact only through unbounded activations.
DeepTile implements ReLU6 in hardware, so for this network the recipe keeps
ReLU6 through CLE: `qat_train.py --cle --cle_keep_relu6`.

Accuracy after calibration alone, before any QAT, on the 3,923-image
ImageNet-mini validation split with FixQuant's preprocessing:

| Recipe | FP32 top-1 | Calibrated top-1 |
|---|---:|---:|
| No CLE, ReLU6 kept | 74.66 | 61.74 |
| CLE, ReLU6 replaced by ReLU | 52.59 | 44.43 |
| CLE, ReLU6 kept | 74.64 | 64.75 |

Replacing ReLU6 costs 22 points before quantization starts; keeping it through
CLE costs almost nothing. `cle_keep_relu6` builds a different network, so it is
recorded in the run manifest, carried into candidates and releases under the
profile `int8-tqt-cle-relu6`, and honoured by `qat_test.py`, `deploy_eval.py`
and `export_deeptile_graph.py` when they rebuild the model.

The FP32 model scores 74.66% top-1 and 91.94% top-5 on that split. timm reports
about 75.5% on the full validation set with bicubic resizing; FixQuant's
evaluation uses its own preprocessing, as for every other network.

## DeepTile support

The network needs two things earlier DeepTile targets lacked, both added with
kernel ABI version 2: a hardware ReLU6 clamp, and 5x5 kernels. The nine 5x5
depthwise layers run as diagonal-dense convolutions, which is correct but slower
than the fused 3x3 path. `tests/test_efficientnet_lite.py` exports the network
with and without CLE and runs the acceptance checker: 49 convolutions, 33
ReLU6 post-operations, 9 residual adds, and kernels of 1, 3 and 5.

## Development QAT run

A local run validated the pipeline on ImageNet-mini: 34,745 training images,
about 35 per class, with the 3,923-image validation split. Recipe:
`--cle --cle_keep_relu6`, 30 epochs, batch 64, learning rate 1e-5, quantizer
learning rate 1e-2, 20 calibration batches.

| Point | Top-1 | Top-5 |
|---|---:|---:|
| FP32 | 74.66 | 91.94 |
| After calibration | 64.75 | 85.98 |
| Best epoch (4) | 73.97 | 91.92 |
| Last epoch (30) | 72.01 | 91.10 |

QAT recovers to 0.7 points below FP32 within four epochs. After that the
validation loss rises steadily: with 35 images per class the model overfits.
The best epoch was also selected on the same split it is reported on, so its
figure is optimistic. Neither limit applies to a full-ImageNet run, which should
use fewer epochs than the 30 here.

The best checkpoint exports to a ModelPackage of 51 nodes that DeepTile compiles
into 45 launches with fusion. DeepTile's software reference reproduces the
exported integer reference exactly, fused and unfused.

## Status

A release needs QAT on full ImageNet and 50,000-sample validation, as for the
other zoo models.
