# Fixed-Point Quantization Tool: Analysis and Roadmap

*Analysis date: 2026-07-07, at commit `a30fc02` ("fxp mobilenet").*
*Status update: 2026-07-08 — roadmap **Phases 0–5 are implemented** (see
`improvements_2026-07.md` for the change log). This document now serves two
purposes: the original analysis (kept as the record of what was wrong and why)
and a live status tracker. Findings are annotated **[FIXED]** / **[OPEN]** /
**[PENDING MEASUREMENT]**; line numbers refer to commit `a30fc02`.*

**What is left to do (short version):**

1. **Re-measure all baselines** — float / PTQ / QAT / twin numbers in
   `baselines.md` are TBD; every pre-rework figure is invalid (a training bug
   and a twin rounding regression were both fixed). Needs GPU runs
   (`scripts/jobscript_arrhenius.sh` for the new NAISS system).
2. **Hardware-side (HLS) work** — grouped/depthwise conv and the `relu6`
   post-op exist in the exporter, twin and spec (v1.1) but not yet in the
   TileCNN kernels; MobileNet cannot run on the FPGA until they do.
3. **Owner/hardware decisions** — per-channel `shift_out`, int16 bias lane,
   thesis dataset, legacy HWCM export path (§13).
4. **Phase 6 optional items** — AdaQuant-style fast finetune, two-step-rounding-
   aware fake quant, MobileNetV3 ops, per-channel weight scales.

---

## 1. Executive Summary

The tool implements a coherent and well-thought-out pipeline: TQT-based
(trained-log2-threshold) fixed-point QAT with power-of-2 per-tensor scales, FX-graph
Conv–BN fusion, conversion to an INT8 hardware-emulation model, a TileCNN digital
twin, and a graph/binary exporter matching the TileCNN Graph Handoff Specification.
For ResNet/VGG this works and produces near-hardware-accurate results.

**The MobileNet QAT accuracy decay has one confirmed primary cause and two
structural amplifiers** (all three addressed in the 2026-07 rework — status
noted per item):

1. **[FIXED]** **(Confirmed bug) BN folding freezes on the very first training batch, and the
   freeze silently detaches all conv weights from the optimizer.**
   `FusedConvBN.get_batch_stats()` compares `bn_mod.num_batches_tracked` against
   `FREEZE_BN_DELAY_DEFAULT = 6000`. Pretrained torchvision checkpoints ship with
   `num_batches_tracked = 736,839` (verified for `mobilenet_v2`), so the freeze
   condition is true on training step 1. Worse, `freeze()` replaces
   `conv_mod.weight`/`conv_mod.bias` with **new** `nn.Parameter` objects
   (`fused_conv_bn.py:215,222`), but the Adam optimizer was built earlier in
   `RunManager.__init__` and still references the old tensors. Result: **every
   Conv weight in the backbone stops training after step 1**. The only parameters
   that keep training are the TQT `log_threshold`s (lr `1e-2`) and the classifier.
   The thresholds are pulled toward 1.0 by *two* simultaneous regularizers
   (Adam `weight_decay=1e-4` on thresholds plus an explicit
   `1e-4·sqrt(Σ log_t²)` loss term in `run_manager.py:201-209`), so quantization
   ranges shrink epoch after epoch with no weight adaptation to compensate —
   accuracy decays over a few epochs. ResNet/VGG suffered the same bug but are
   robust enough per-tensor that threshold-only training still looked acceptable.
   *Fix: counter reset at fusion, in-place fold, zero-bias Parameter at fusion,
   `weight_decay=0` on the threshold group; tested in
   `tests/test_fused_conv_bn.py`.*

2. **[MITIGATED]** **(Structural) Per-tensor power-of-2 weight quantization is severely lossy on
   BN-folded depthwise weights.** Measured on pretrained MobileNetV2: median
   per-output-channel |w|max spread across a depthwise layer is **42×** (worst
   layers contain near-dead channels, ratios in the 10⁶–10¹² range from tiny BN
   gammas). Quantization SQNR of depthwise folded weights: median **31 dB
   per-tensor vs 43 dB per-channel** (worst layer 23.7 dB → 42.6 dB). Several
   depthwise layers effectively lose most of their channels to quantization noise
   before QAT even starts.
   *Mitigation: cross-layer equalization (`equalization.py`, `qat_train.py
   --cle`) cuts the measured spread from 42× to 4.5× on pretrained MobileNetV2.
   The full fix — per-output-channel `shift_out` in hardware — remains an open
   decision (§13 Q1).*

3. **[FIXED in software / OPEN in HLS]** **(Structural) The TileCNN export path does not support MobileNet yet.**
   The bit-exact reference generator raises `ValueError("TileCNN reference
   supports groups=1 only")` (`tilecnn_exporter.py:69`), and the exporter fuses
   `HardwareRelu6` into predecessor nodes as a plain `relu` post-op
   (`tilecnn_exporter.py:580-591`), silently dropping the 6-clamp. The handoff
   spec (`graph_handoff_spec.md`) has no `relu6` post-op or `groups` support on
   the hardware side.
   *Fix: grouped conv and true `relu6`/`post_add_relu6` post-ops implemented in
   the exporter reference, digital twin, and spec (v1.1 section), tested in
   `tests/test_export.py`/`test_kernels.py`. Still **open**: the HLS kernels
   themselves must implement the v1.1 extensions.*

**Highest-impact fixes, in order** — all five landed in the 2026-07 rework:

1. ✅ BN-freeze/optimizer bug fixed (counter reset at fusion, in-place fold).
2. ✅ Vitis-style schedule: multi-batch calibrate → configurable BN-fold freeze
   → QAT with thresholds frozen after `threshold_freeze_frac` of epochs.
3. ✅ Multi-batch calibration with the MSE fix-pos search
   (`QatProcessor.calibrate`, reservoir + `find_fix_pos(scope=5)`).
4. ✅ CLE implemented as the software path (`--cle`); per-channel shifts remain
   a hardware decision (§13 Q1).
5. ✅ Diagnostics: `fixquant/diagnostics.py` (per-layer frac/SQNR/clip reports,
   per-epoch threshold CSV, QAT-vs-hardware parity sweep) +
   `tools/layer_sensitivity.py`.

**Bonus finding**: the new kernel tests caught a regression introduced *by*
commit `a30fc02` itself — the digital twin had dropped the first `+1` of the
two-step EMIT_LOOP rounding, so twin and exporter reference disagreed by
±1 LSB. Restored and pinned by `tests/test_golden.py`.

---

## 2. Repository Structure

*(Updated to the post-rework tree; deletions/additions vs `a30fc02` noted.)*

```
src/fixquant/
  quantization/
    fix_ops.py          Fixed-point primitives: rounding modes, fake-quant autograd
                        (FakeQuantize), to_int_tensor, find_fix_pos (MSE fix-pos
                        search, ported from Vitis "diffs" method), FixedPointQuantizer.
    tqt_quantizer.py    TQTQuantizer + TQTQuantize autograd. Learnable log2
                        threshold, warmup init, multi-batch calibration
                        (reservoir + MSE fix-pos), bounded_range, enable_quant.
    equalization.py     NEW: BN fold + cross-layer equalization + bias correction.
    qat_modules.py      QAT wrappers: QuantizedConv2d, QuantizedLinear, QMaxPool2D,
                        QAdaptiveAvgPool2d, QElementwiseAdd (hardware-aligned
                        residual add), QuantStubC.  (QAvgPool2d deleted.)
    fused_conv_bn.py    FusedConvBN with the freeze fixes: counter reset at fusion,
                        in-place fold, configurable freeze_bn_delay.
                        (tqt_ops.py deleted — duplicate threshold init.)
  graph/
    qat_processor.py    QatProcessor passes: ConvBnFusion, ModuleReplacement,
                        ReplaceAdd, ReplaceFunctionalPool, ActRange (ReLU/ReLU6
                        bounds), InputQuantStub; multi-batch calibrate();
                        freeze_thresholds(); preflight_check().
    inference_processor.py  InferProcessor: QAT → StandardModel → hardware model
                        (strict qconfig, input frac from QuantStub); legacy
                        HWCM weight-export helpers still present (§13 Q7).
  diagnostics.py        NEW: quantizer_report, threshold logging, parity_sweep.
  emulation/
    fxp_emu_modules.py  Hardware* modules — bit-exact TileCNN integer arithmetic
                        (two-step rounding restored; add uses _signed_shift).
    model_introspector.py  StdModelInspector (unchanged).
  export/
    tilecnn_exporter.py Exporter + _tilecnn_* reference kernels: grouped conv,
                        relu6 post-ops, maxpool attrs, shift-legality checks,
                        no silent frac defaults.
  models/               get_model() factory (torchvision-backed) + resnet.py,
                        cifar models.
  data/, training/      Providers; RunConfig/RunManager (no weight-decay on
                        thresholds, threshold-freeze schedule, per-epoch logs).
tools/
  qat_train.py / qat_test.py / deploy_eval.py    All take --model/--checkpoint;
                        qat_train adds --cle/--bias_corr/--calib_batches/
                        --threshold_freeze_frac; seeded.
  layer_sensitivity.py  NEW: one-layer-at-a-time quantization probes.
  export_tilecnn_graph.py, export_hw_testcases.py, export_fixA_refactor_testcases.py,
  print_model_graph.py, train.py (working float baseline trainer)
  archive/              Retired: hw_layer_test_gen, gen_resnet18_fc_testdata,
                        train_cifar, ddp_train_hvd (see archive/README.md).
tests/                  NEW: 53 pytest tests — kernel bit-exactness, golden
                        regression (tests/golden/), FusedConvBN training
                        correctness, calibration, QAT flow (incl. MobileNetV2),
                        export structure, CLE numerics.
configs/quant_config.yaml       Layer map + freeze_bn_delay. (Note: bitwidths are
                        still hardcoded at 8 in module constructors — open.)
docs/                   This report, improvements_2026-07.md, baselines.md,
                        arrhenius_gpu_guide.md, tqt.md, conv_fused.md, qmodules.md,
                        tilecnn_exporter_and_digital_twin.md, mobilenet roadmap.
graph_handoff_spec.md   Spec incl. v1.1 extensions (groups, relu6 post-ops).
scripts/jobscript_arrhenius.sh  NEW: GH200 job template (old Alvis one deprecated).
```

~~No automated test suite exists~~ **[FIXED]** — `tests/` holds 53 pytest tests
(run `python -m pytest tests/`); the root smoke scripts were absorbed into it.
The stale README references are gone.

## 3. Current Quantization Flow

**Scheme.** Symmetric, signed, per-tensor, power-of-2-scale ("fixed-point")
quantization for everything: weights, biases, activations, all at 8 bits.
`q = clamp(round(x / s), -128, 127)`, `s = 2^ceil(log2 t) / 2^(b-1)`, i.e. each
tensor is described by a single fractional-bit count `frac = 7 - ceil(log2 t)`.
Zero-points are always 0; activations are signed even after ReLU (one sign bit is
always spent).

**TQT.** Thresholds `t` are `nn.Parameter`s (`log_threshold`) trained with the
TQT gradient (Jain et al. 2019, same formulation as Vitis AI's nndct TQT). On the
first forward pass ("warmup"), the threshold is initialized: weights → |mean|+3σ,
activations → KL-divergence histogram method. After that single batch the
initializer never runs again.

**Calibration** — **[FIXED]**. At `a30fc02`, "calibration" was only a one-shot
warmup init on the first batch (24 images); `find_fix_pos(scope>1)` existed but
was unused. Now `QatProcessor.calibrate(loader, device, max_batches, scope)` has
every quantizer collect a value reservoir across batches and set its threshold
by the MSE fix-position search (default 20 batches, `scope=5`); ReLU/ReLU6-fed
activations are calibrated on their bounded range (`ActRangePass`). A
quantized-accuracy (PTQ) check after calibration is still a manual step —
record it per `baselines.md`.

**Per-layer wiring.** Conv/Linear QAT modules quantize weight and bias (8-bit TQT)
and the layer *output* (pre-activation). ReLU/ReLU6 stay as float modules after the
quantizer; residual adds are replaced by `QElementwiseAdd` (float add + TQT
re-quantize); maxpool and GAP have their own output quantizers; an input
`QuantStubC` quantizes the network input. Each layer's input frac is implicitly the
predecessor's output frac — this matches the hardware's chained-frac model.

**BN folding** (`FusedConvBN`): during training, the standard two-forward-pass
scheme (batch stats correction, weights scaled by γ/σ_running before weight
quantization). Auto-freeze after `freeze_bn_delay` steps *counted from fusion*
(configurable, `null` = manual) folds BN into `conv_mod.weight/bias` in place.
The two bugs described in §5.1 are **[FIXED]**. The `--cle` path folds BN before
QAT instead (no `FusedConvBN` at all).

**QAT training** (`RunManager` + `RunConfig`): Adam, two param groups —
`log_threshold`s at `quantizer_lr=1e-2` (staircase-decayed continuously, and
**`weight_decay=0`** — the double regularization is **[FIXED]**, only the
explicit TQT norm penalty remains), everything else at `init_lr=1e-5`.
Thresholds are **frozen after `threshold_freeze_frac`** (default 0.7) of the
epochs; per-epoch threshold/frac state is logged to
`logs/quant_thresholds.csv`. Still absent (by choice, revisit if needed):
warmup epochs, gradient clipping.

**Bias & accumulator.** Bias is quantized to 8 bits with its own TQT threshold —
both in QAT and in the export (the handoff spec mandates `bias_dtype: int8`; the
hardware widens to int32 internally and pre-shifts by `fout - fb + 1`). The
accumulator is int32 (emulated in int64/float64); QAT does not model accumulator
width at all (float accumulate).

**Rounding/saturation.** QAT fake-quant rounds HALF_UP ("round -1.5 → -1")
and saturates to [-128,127]. The hardware (and digital twin) does a **two-step**
emit: `s1 = (acc >> (shift_out-1)) + 1`, then `out = (s1 + bias_adj + 1) >> 1`,
then saturate. Training math and hardware math still differ by up to ±1 LSB per
layer — modeling the two-step rounding inside QAT fake-quant remains a Phase 6
**[OPEN]** item. (Note: the twin had *lost* the first `+1` in the `a30fc02`
refactor; restored 2026-07 and pinned by the golden test. The residual-add
inputs are now aligned to the output grid during QAT, closing risk §8/#5.)

**Inference conversion & matching.** `convert_to_std_model()` rebuilds a plain
module-graph with fake-quantized weights and `frac_*` buffers;
`convert_to_hardware_model()` swaps in the `Hardware*` INT8 modules using
`generate_qconfig()` fracs — now **strict** (missing params raise; input frac
comes from the learned QuantStub). Bit-exactness is asserted automatically:
`tests/test_kernels.py` (twin ↔ exporter reference), `tests/test_golden.py`
(committed integer golden), and `diagnostics.parity_sweep` (QAT ↔ hardware,
first-layer diff ≤ 1 LSB enforced in `tests/test_qat_flow.py`).

## 4. Current TileCNN Export Flow

Pipeline: QAT checkpoint → `QatProcessor.quantize()+load_qat_weights()+freeze()` →
`InferProcessor.convert_to_std_model()` / `convert_to_hardware_model()` →
`StdModelInspector` (hooks capture int8 activations) → `TileCNNGraphExporter.export()`
→ `graph.json` + `inputs/ params/ refs/` int8 binaries → the exporter then
**recomputes** all reference outputs with the bit-exact `_tilecnn_*` kernels so the
C-simulation testbench compares against true hardware arithmetic.

Bit-exactness-critical facts (status vs `a30fc02`):

- Layouts: activations CHW, conv weights OIHW. **[OPEN]** The legacy HWCM path
  in `InferProcessor.export_weights_to_file` still coexists (§13 Q7).
- All tensors int8, including bias; per-tensor `frac` in `graph.json`.
- Conv/linear emit: `(acc >> (shift_out-1)) + 1` → bias add at `fout+1` → +1,
  `>>1` → saturate → optional `relu`/`relu6` post-op.
- Residual add: align residual by rounding shift to main-branch `fout`, add,
  saturate, optional post-add relu/relu6.
- **[FIXED]** MaxPool reference now reads kernel/stride/padding from node attrs
  (was hardcoded 3×3/s2/p1); the converter wires `post_pool_shift`.
- GAP: fixed-point reciprocal with 16 fractional bits; export-time legality
  check on the total shift.
- **[FIXED in software]** grouped/depthwise conv supported; `relu6` is a real
  post-op (spec v1.1). **[OPEN]** HLS kernels must implement both.
- **[FIXED]** No silent defaults: missing qconfig entries raise in both the
  converter and (via required params) the exporter; `_check_shift_legality`
  rejects out-of-range shifts.

The strongest asset of the repo is this exporter + digital-twin pair: the same
integer kernels exist in three places (exporter reference, Hardware* modules, HLS)
and the ResNet-50 twin has been validated to FPGA-level accuracy.

## 5. MobileNet Quantization Problem

Ranked by likely contribution to the observed "accuracy drops after a few epochs
of QAT". Status: 5.1 fixed; 5.2 mitigated (CLE); 5.3–5.6 fixed in software —
the decisive confirmation (a stable, accurate MobileNetV2 QAT run) is
**pending measurement** (`baselines.md`).

**5.1 [FIXED] BN freeze fires on batch 1 and detaches conv weights from the optimizer
(primary, confirmed).** As detailed in §1: pretrained `mobilenet_v2` has
`num_batches_tracked = 736,839 > 6000`, so every `FusedConvBN` freezes on its
first training call, and `freeze()` re-creates `conv_mod.weight`/`bias` as new
`nn.Parameter`s that the already-constructed Adam optimizer does not own. From
step 1, the backbone weights are constants. What *does* keep evolving:

- `log_threshold`s at lr 1e-2 (100,000× the weight lr), doubly regularized toward
  `log t = 0` (Adam weight_decay + the explicit `1e-4·sqrt(Σ log t²)` loss term).
  With weights frozen, threshold shrinkage monotonically increases clipping and
  nothing can adapt → the characteristic slow decay over epochs.
- The classifier (`QuantizedLinear`, no BN, so unaffected by the freeze bug).

Also note the `4b58f5a`/`30477b1` "freeze inconsistency" commits guarded
*double*-folding but not this. And even with the intended delay, freeze at step
6000 mid-training would still silently detach weights — the reparameterization is
the deeper bug, `num_batches_tracked` initialization is the trigger.

**5.2 [MITIGATED] Depthwise convolutions under per-tensor power-of-2 weight quantization
(primary structural).** *CLE (`--cle`) cuts the spread 42×→4.5× measured; full fix needs per-channel shifts in HW (§13 Q1).* Measured on BN-folded pretrained MobileNetV2 weights:

| Metric (17 depthwise layers) | per-tensor pow2 | per-channel pow2 |
|---|---|---|
| Median weight SQNR | 31.2 dB | 43.0 dB |
| Worst layer (features.13.conv.1.0) | 23.7 dB | 42.6 dB |
| Median per-channel range spread | 42× | — |

Depthwise channels don't share an accumulator with other channels, so one outlier
channel (|w|max ≈ 8 in `features.13`/`features.1`) forces `frac_w = 3` for the
whole tensor and quantizes the many small-range channels to near-zero. TQT can
trade clipping vs resolution but cannot fix a 42× spread with one scale. QAT
*could* partially re-train around this — but not with the weights frozen (5.1).

**5.3 [FIXED] ReLU6 / linear-bottleneck specifics (moderate).** *`ActRangePass` bounds ReLU6-fed quantizers to [0,6]; export emits true `relu6`; linear-bottleneck sensitivity is measurable with `tools/layer_sensitivity.py`.*
- The activation quantizer sits *before* ReLU6 and is signed, so for every
  ReLU6-bounded tensor (range [0,6]) at least the sign bit plus any headroom above
  6 is wasted. With the KL init computed on the *pre*-activation distribution
  (which has large negative mass), the initial threshold can be far from the
  useful [0,6] range.
- The linear bottleneck outputs (pointwise convs without ReLU, feeding residual
  adds) are the known accuracy-sensitive tensors of MobileNetV2; they get the same
  8-bit treatment with no special handling or sensitivity check.
- In export, ReLU6 degrades to plain ReLU (§4) — for inference this is a hard
  correctness bug for any activation frac where 6·2^frac < 127.

**5.4 [FIXED] Residual adds.** *`QElementwiseAdd` now rounds both inputs onto the output grid (hardware alignment) during QAT.* `QElementwiseAdd` learns an output threshold on the float
sum; the hardware aligns each input to `fout` with rounding shifts *before* the
add (double rounding, saturation before relu). Scales of the two branches are
unconstrained relative to each other; large frac differences cost precision on the
shifted branch. Not MobileNet-specific but inverted residuals have many adds on
sensitive linear-bottleneck tensors.

**5.5 [FIXED] Optimizer/hyperparameter risks (secondary).** *Threshold weight-decay removed, `--quantizer_lr_decay` type fixed, LR decay continuous, threshold freeze schedule added. Weight-decay exclusion for depthwise weights remains a tuning option, not implemented.*
- Adam `weight_decay=1e-4` on all params: for MobileNet, decay on BN-folded conv
  weights (had they been training) and on thresholds is aggressive; standard
  MobileNet recipes use ~4e-5 and exclude BN/depthwise from decay.
- `--quantizer_lr_decay` is declared `type=int` in `qat_train.py:35` — passing
  any value on the CLI turns 0.5 into an int (0 → thresholds stop moving, or 1 →
  never decays).
- No threshold freeze near the end of training: quantization params can keep
  moving until the last step; the exported frac is whatever `ceil(log2 t)` landed
  on at the final step (frac is quantized with `ceil`, so a hair-trigger threshold
  crossing a power of 2 late in training changes the entire layer grid).
- 24-image warmup init for KL thresholds is very small for a 53-conv network.

**5.6 [FIXED in software] Export/eval parity for MobileNet.** *`--model` CLI everywhere; `HardwareRelu6` gets the producer's output frac; depthwise supported in the reference; MobileNetV2 golden-reference export is tested (`tests/test_export.py`). Bit-exact C-sim validation awaits the HLS v1.1 kernels.* `deploy_eval.py`/`qat_test.py` select
models by editing hardcoded lines; the emu path runs (HardwareConv2d passes
`groups` through) but `HardwareRelu6` receives `frac_in` while clamping should use
the *output* frac of the producing conv (they're equal in the current wiring, but
the constructor semantics are fragile), and the TileCNN reference export rejects
depthwise entirely. So there is currently **no way to produce a MobileNet
bit-exact golden reference**, meaning QAT results can't be validated against
hardware semantics end to end.

## 6. Comparison with Vitis AI Quantization Ideas

The TQT quantizer, threshold init schemes, rounding conventions, and the
`find_fix_pos` "diffs" search are already ports of vai_q_pytorch (nndct) code, so
the repo shares Vitis AI's DNA. What Vitis AI adds *around* that core, and what is
worth borrowing in simplified form. *Status 2026-07: every row marked "Yes" was
adopted — multi-batch calibration, threshold freezing, bias correction
(simplified fast-finetune), CLE, relu6 post-op, pre-flight inspector, layer
sensitivity, legality checks. Remaining: full AdaQuant fast finetune (Phase 6)
and the per-channel-weights hardware decision.*

| Vitis AI concept | What it does | Worth adopting for TileCNN? |
|---|---|---|
| **Multi-batch calibration** | Forward 100–1000 images in "calib" mode; fix positions chosen from accumulated statistics via MSE ("diffs") search, then written to a config | **Yes — cheap.** Replace one-shot warmup with N-batch stat accumulation + `find_fix_pos(scope≈5)`; keep KL as an option. The search code already exists in `fix_ops.py`. |
| **Freeze after calibration** | Calibrated fix positions are fixed for evaluation/deployment; QAT is a separate deliberate phase | **Yes.** Add explicit `freeze_thresholds()` (already exists per-quantizer as `freeze_quant`) called for the final K epochs and always before export. |
| **Fast finetune (AdaQuant-style)** | Per-layer optimization of weights/bias to minimize layer-output MSE on ~1000 images, no labels/optimizer needed | **Yes, simplified.** Even a bias-correction-only pass (match per-channel output means float-vs-quant) is known to recover several % on MobileNet and is ~50 lines. Full AdaQuant later. |
| **QAT with trained thresholds** | Same TQT; run *after* calibration init, few epochs, small lr | Already present — needs the schedule fixes from §5, not new machinery. |
| **BN folding into QAT modules** | Fuse then train; BN stats frozen deliberately | Present, but Vitis controls freeze explicitly rather than via a step counter — adopt explicit control. |
| **Per-channel weights** | Available in the general quantizer config (not for pow-2 DPU targets) | **Decide with hardware.** Per-output-channel *shift* (still pow-2) is cheap in an FPGA emit stage. If TileCNN won't add it, use CLE instead. |
| **Cross-layer equalization (CLE)** | Rescale adjacent layers (Nagel et al.) to equalize per-channel weight ranges; standard fix for depthwise per-tensor quantization; requires ReLU6→ReLU conversion | **Yes if staying per-tensor.** Pure software pass before quantization; no hardware change. Pair with the `convert_relu6_to_relu` idea below. |
| **ReLU6→ReLU conversion option** | DPU flow converts ReLU6 to ReLU when ranges allow (QAT clamps ranges anyway) | Partially — safer for TileCNN is a real `relu6` post-op in the spec **or** proving `6·2^frac ≥ 127` per layer and only then exporting `relu`. Don't silently degrade (current behavior). |
| **Inspector** | Checks each op against target constraints before quantizing; produces a report | **Yes, tiny version:** a pre-flight pass that walks the FX graph and errors on unsupported ops (groups≠1, relu6-without-post-op, non-GAP avgpool…) instead of failing mid-export or silently defaulting. |
| **Layer sensitivity / QuantAnalyzer** | Per-layer quantize-one-layer-at-a-time accuracy probes | **Yes, simplified:** one script, N≈200 images, report top-k most sensitive layers; drives "which layers get relaxed treatment". |
| **Hardware legality checks** | DPU checks `shift_cut`/`shift_bias` legal ranges and adjusts fix pos | **Yes.** TileCNN has the same implicit constraints (`shift_out ≥ 0`? `fout - fb + 1` range, GAP `total_shift ≥ 0`) — validate at export, adjust frac or fail loudly. |
| Full nndct graph IR, ONNX/XIR export, hard/soft fusion engine | Heavy infrastructure | **No** — unnecessary for this prototype. |

Caveat: Vitis AI behavior is not automatically correct for TileCNN — e.g. DPU's
bias handling, leaky-relu approximations, and avg-pool scale tricks are
DPU-specific. Borrow the *process* (calibrate → check → finetune → freeze →
export → verify), not the constants.

## 7. Inefficiencies and Technical Debt

Concrete issues found (file references at analyzed commit `a30fc02`).
*Status 2026-07: 1, 2, 4, 5, 6, 8, 9 fixed; 7 mostly fixed; 3, 10, 11
partially open — inline notes below.*

1. **[FIXED] Silent training bugs**
   - `fused_conv_bn.py:183` freeze trigger uses pretrained `num_batches_tracked` (§5.1).
   - `fused_conv_bn.py:213-222` `freeze()` re-creates Parameters → optimizer detachment (§5.1).
   - `run_manager.py:201-209` + `run_config.py:133`: double threshold regularization; Adam `weight_decay` also decays `log_threshold`s.
   - `qat_train.py:35` `--quantizer_lr_decay` has `type=int`.
   - `qat_modules.py:89-95` `QuantizedConv2d._conv_forward` computes the padded conv for non-zero `padding_mode` and then unconditionally overwrites it with the default conv.
   - `run_config.py:76-84` LR "decay" only fires when `step % decay_steps == 0` exactly; with float decay_steps this is fragile.
2. **[FIXED] Calibration is a no-op beyond batch 1** (§3) — now real multi-batch MSE calibration.
3. **[PARTIALLY OPEN] Config file is decorative.** `configs/quant_config.yaml` maps string class names, but `QatProcessor` hardcodes its pass list and replacement maps; `QuantizeLayerPass` (which would read the config) is dead code, as is `OutputQuantStubPass`. Bitwidths (8) are hardcoded in every module constructor. *Dead passes deleted; `freeze_bn_delay` now read from the yaml. Still open: bitwidths and pass selection are not config-driven.*
4. **[FIXED] Model selection by editing source** *(all tools take `--model`/`--checkpoint`; per-model checkpoint dirs)* — in `qat_train.py`/`qat_test.py`/`deploy_eval.py` (commented-out lines); checkpoint paths and dataset paths hardcoded; `deploy_eval.py:97-102` builds the *same* model for both `--model_type` branches.
5. **[FIXED] Silent fallback fracs** — `inference_processor.py:413-417` and the exporter default to `frac=5`/`frac_w=7` when a layer is missing from qconfig; a wiring bug becomes an accuracy bug instead of an error.
6. **[FIXED] `generate_qconfig` hardcodes the first-layer input frac to 5 “for imagenet”** *(now captured from the input QuantStub)* — (`inference_processor.py:605-616`), and reconstructs `frac_in` by an ad-hoc graph walk duplicated in `model_introspector.py`.
7. **[MOSTLY FIXED] Duplicated / dead code** *(all named dead code deleted; state_dict-boilerplate mixin and `models/resnet.py` duplication deliberately deferred)*: threshold-init logic exists twice (`tqt_quantizer.py`, `tqt_ops.py`); ~120 lines of state_dict boilerplate copy-pasted across every QAT module; `models/resnet.py` duplicates torchvision; `FXPConv2dTorch`, `HLSMaxPool2D`, `QAvgPool2d` (with a broken `from_float` reading `output_size` off an `AvgPool2d`) appear unused; README references deleted `model_transforms.py`; debug `print`s left in library code (`tqt_quantizer.py:151` "pybassed", `inference_processor.py:309`).
8. **[FIXED] No tests** *(53 pytest tests incl. golden regression)*: no unit tests for rounding/shift kernels, no golden-file regression tests wired into CI-style scripts, no QAT-vs-emu-vs-twin consistency assertion. The exported testcase dirs under `outputs/` are one-off artifacts.
9. **[FIXED] No diagnostics** *(`fixquant/diagnostics.py`, `tools/layer_sensitivity.py`, per-epoch threshold CSV)*: nothing logs per-layer frac choices, threshold trajectories, clip rates, or per-layer float-vs-quant error; QAT failures are only visible as end-to-end accuracy.
10. **[OPEN] Layout ambiguity** *(legacy HWCM path still present — §13 Q7)*: exporter writes OIHW; legacy `export_weights_to_file` writes HWCM; both live in the codebase with no single source of truth.
11. **[PARTIALLY OPEN] Research vs deployment code intermixed** *(smoke scripts → tests/, legacy tools → tools/archive/; `outputs/` + `hw_data_files/` ~40 MB still git-tracked — untrack manually)*: root-level smoke scripts, `outputs/` artifacts and `hw_data_files/` checked into git (~40 MB), tools with hardcoded user paths.

## 8. Bit-Exactness Risk Register

| # | Risk | Where it appears | Why it matters | How to test | Recommended fix | Status (2026-07) |
|---|---|---|---|---|---|---|
| 1 | ReLU6 exported as plain `relu` | `tilecnn_exporter.py:580-591`; spec has no `relu6` post-op | Wrong results whenever `6·2^frac < 127` for that tensor | Layer testcase with inputs > 6.0 equivalent | Add `relu6`/`clip_max` post-op to spec + exporter + kernels, or export-time proof that clamp is unreachable | **Fixed (SW)**: relu6 post-op in exporter/twin/spec v1.1; HLS open |
| 2 | Depthwise (`groups≠1`) unsupported in reference | `tilecnn_exporter.py:69` | MobileNet cannot get golden refs; HW behavior undefined | Attempt export of one dw block (currently raises) | Implement grouped conv in `_tilecnn_conv2d` + HW; add pre-flight inspector check | **Fixed (SW)**: grouped conv in reference/twin, tested; HLS open |
| 3 | QAT rounding ≠ HW two-step emit rounding | `fix_ops.py` HALF_UP vs `_tilecnn_conv2d` truncate-then-round | ±1 LSB/layer systematic drift; compounds over 50+ layers | Compare std-model vs twin per-layer outputs, count mismatched elements | Either model two-step rounding in QAT fake-quant (exact), or accept and *measure* per layer (report) | **Open** (Phase 6): parity sweep now *measures* it per layer |
| 4 | 8-bit bias resolution | spec `bias_dtype: int8`; TQT bias quantizer | Folded BN biases are large-range; error is a per-channel DC offset | Per-channel output-mean comparison float vs quant | Bias-correction pass post-fold; longer term consider int16 bias lane in HW | **Mitigated**: bias-correction pass available (`--bias_corr`); int16 lane = §13 Q4 |
| 5 | Residual-add double rounding & pre-relu saturation | `HardwareElementwiseAdd`, `_tilecnn_residual_add` vs float QAT add | QAT never sees align-shift error or saturation-before-relu | Adversarial add testcase with frac mismatch of 2+ | Model align-shifts in `QElementwiseAdd` during QAT; constrain branch fracs (e.g. force equal frac on both add inputs) | **Fixed**: QAT add aligns inputs to output grid; kernels use `_signed_shift`, tested |
| 6 | Silent default fracs (5/7) on qconfig miss | `inference_processor.py:413-417`, exporter fallbacks | A name-mapping bug produces plausible-but-wrong binaries | Export with an intentionally renamed layer; must fail | Replace defaults with hard errors | **Fixed**: hard errors, tested |
| 7 | First-layer input frac hardcoded to 5 | `inference_processor.py:605-616`, `default_input_frac=5` everywhere | Wrong for CIFAR/other normalizations; input clipping (ImageNet-normalized range ±~2.64 fits, but only by luck) | Feed constant extreme images; compare quantized input | Derive from the input QuantStub's learned frac; single source of truth | **Fixed**: frac captured from input QuantStub |
| 8 | MaxPool kernel hardcoded 3×3/s2/p1 in reference | `_tilecnn_maxpool:121-124` | Any other pool config silently produces wrong refs | Export a 2×2 pool testcase | Read attrs from node | **Fixed**: attrs read from node, tested |
| 9 | Maxpool re-quantization to a *different* frac | `QMaxPool2D` learns its own threshold; HW applies `post_pool_shift` | Max is grid-preserving; a learned frac change adds a pointless rounding step and a shift the HW must honor | Check exported `frac_in==frac_out` for pools | Remove pool output quantizers; inherit input frac | **Mitigated**: `post_pool_shift` wired from qconfig; pool quantizer kept for checkpoint compat |
| 10 | Unfrozen model export | `convert_to_std_model` copies `conv_mod.weight` (unfolded) if `freeze()` wasn't called | Silently exports non-BN-folded weights | Export without freeze; outputs diverge grossly | Assert `frozen` in conversion | **Open** (minor): no `frozen` assert in conversion yet |
| 11 | Weight layout OIHW vs HWCM | exporter vs `export_weights_to_file` | Wrong layout = garbage inference | Round-trip load test per artifact | Kill or clearly deprecate the legacy exporter | **Open**: legacy HWCM exporter still present (§13 Q7) |
| 12 | `frac = 7 - ceil(log2 t)` off-by-one at pow-2 boundaries | `tqt_quantizer.export_quant_info` | Threshold drifting across a power of 2 at the last training step flips the whole layer grid | Log `log2 t` distance-to-boundary at export | Freeze thresholds before final epochs; warn when `log2 t` within ε of an integer | **Mitigated**: threshold freeze schedule (default last 30% of epochs) |
| 13 | GAP reciprocal & shift constraints | `_tilecnn_gap` requires `total_shift ≥ 0` | Certain frac combinations crash or wrap | Sweep frac_in/out combos in a unit test | Export-time legality check (see Vitis `shift_bias`-style checks) | **Fixed**: `_check_shift_legality` at export |
| 14 | Accumulator width unmodeled in QAT | float accumulation in QAT vs int32 HW | Large layers could overflow int32 in principle (unlikely at 8×8×k²·C) | Worst-case bound check per layer at export | Add static bound check `log2(C_in·k²·127·127) < 31` | **Open** (low risk): static bound check not implemented |
| 15 | Padding count/stride conventions | QAT `F.conv2d` vs HW tiling | Classic source of off-by-one at borders | Boundary-activation testcases (already exported — keep) | Keep golden border tests in the standard suite | **Kept**: kernel tests + exported boundary testcases remain the guard |
| 16 | Signed activation after ReLU wastes a bit; HW assumes signed | whole pipeline | Not a mismatch (consistent), but 1 bit of accuracy left on the table | — | Optional: unsigned activation mode for post-ReLU tensors (HW change — probably not worth it now) | **Open by design**: unsigned mode deferred (HW change) |

**Validation strategy** (practical, ordered) — *items 1–3 implemented in `tests/`; item 4 open pending HLS v1.1*:
1. **Kernel unit tests** (pure Python, seconds): exhaustive small-tensor tests of
   `_tilecnn_conv2d`/`HardwareConv2d`/HLS kernel triples over all shift sign
   combinations, including relu6, grouped conv, GAP, add alignment. These three
   implementations must agree bit-exactly with each other before touching the FPGA.
2. **Golden model test** (minutes): for each supported architecture, one fixed
   input batch + committed reference logits (int8) produced by the digital twin;
   any refactor must reproduce them exactly. Store under `tests/golden/`.
3. **Per-layer parity sweep**: run std-model and twin side by side with hooks;
   report per-layer count of mismatched int8 elements and max |Δ|. Gate: only
   rounding-explainable ±1 diffs, at expected rates.
4. **C-sim/FPGA loop**: keep the existing exported testcase mechanism, but
   generate it from the test suite (not one-off scripts) and include MobileNet
   blocks once #1/#2 in the register are fixed.

## 9. Recommended Improved Quantization Flow

**[IMPLEMENTED]** — this flow is now what `tools/qat_train.py` + the exporter
execute (step numbers → implementation: 2 = fixed `FusedConvBN`, 3 =
`preflight_check`, 4 = `--cle`, 5 = multi-batch `calibrate()`, 6 =
`--bias_corr`, 7 = threshold-freeze schedule, 8 = strict conversion, 9 =
`tests/` + `parity_sweep`, 10 = exporter legality checks). Step 1 (recorded
float baselines) and step 11 (C-sim/FPGA on v1.1 kernels) are the open ends.

Target flow (each step explicit, scriptable, and logged):

1. **Float baseline** — train/verify float model; record top-1 (reproducible seed).
2. **Fuse & fold** — FX fusion of Conv+BN (as today) but: reset
   `num_batches_tracked=0` at fusion, `freeze_bn_delay` set explicitly from config
   (or `None` = manual), and BN freeze implemented as a *flag flip only* (no
   Parameter re-creation).
3. **Pre-flight inspection** — walk the graph; verify every op/attr is exportable
   (groups, relu6, pool configs, avgpool type); fail with a report otherwise.
4. **(MobileNet-class models) equalize** — optional CLE pass on
   conv→dw→pw chains before calibration; optional per-channel-shift weight mode if
   TileCNN adopts it.
5. **Calibrate** — N batches (≥ 200–1000 images) in eval mode with BN folding in
   *frozen semantics*: accumulate stats, then set thresholds by MSE fix-pos search
   (weights: per-tensor minimum-MSE; activations: MSE or KL). Evaluate quantized
   accuracy right here — this is the PTQ number, the baseline QAT must beat.
6. **Bias correction / fast finetune (optional but recommended)** — per-layer
   output-mean bias correction; later, AdaQuant-style layerwise finetune.
7. **Short QAT** — 2–10 epochs, Adam or SGD, weight lr ~1e-5…1e-6, threshold lr
   ~1e-2 with decay, **no weight_decay on thresholds or BN/depthwise params**,
   BN frozen from the start (fold already trusted); freeze thresholds for the last
   ~30% of training so fracs are stable at export.
8. **Convert & assert** — `freeze()` asserted, no silent defaults, qconfig written
   to `configs/qconfig_files/` with the checkpoint.
9. **Golden tests** — std-model vs twin per-layer parity sweep + committed golden
   logits (§8) must pass.
10. **Export** — TileCNN graph + binaries + bit-exact refs; export-time legality
    checks (shift ranges, GAP shift, bias shift).
11. **Hardware verification** — C-sim testbench on exported refs; then FPGA.

This is deliberately Vitis-shaped (calibrate → finetune → QAT → inspect → export →
verify) but with only the pieces TileCNN needs.

## 10. MobileNet-Specific Roadmap

*Status: items 1–4 and 6 implemented (item 2 via `ActRangePass` + relu6
post-op rather than a separate QReLU6 module; item 3 short-term CLE done,
per-channel `shift_out` open; item 4 weight-decay exclusion for depthwise not
implemented). Item 5 defaults are wired into `qat_train.py` but the prescribed
runs have not been executed. Item 7 done in software (HLS pending). Item 8 =
acceptance, **pending measurement**.*

1. **Unblock training (bug fixes, do first)** ✅
   - Reset `num_batches_tracked` at fusion; make BN freeze explicit (config/epoch
     boundary), and make `freeze()` fold in-place (`param.data.copy_`) instead of
     re-creating Parameters — or rebuild the optimizer after freezing.
   - Remove threshold double-regularization: keep *either* the explicit
     `sqrt(Σlog t²)` penalty *or* Adam weight_decay on thresholds, not both; set
     `weight_decay=0` for the threshold param group in any case.
   - Fix `--quantizer_lr_decay` type.
   - Re-run the existing MobileNet QAT recipe; verify accuracy no longer decays.
     (Also re-run ResNet-18: it should now *improve* over the old numbers.)
2. **ReLU6 handling** ✅ *(implemented as bounded-range quantization + true relu6 export)*
   - QAT: for conv→ReLU6, use an unsigned/clipped activation quantizer whose
     threshold initializes at 6.0 (learnable below 6, hard-capped at 6) placed
     *after* the clamp semantics — i.e. quantize `min(max(y,0),6)`.
   - Export: add `relu6` post-op (spec + exporter + `_tilecnn_conv2d` + HLS), or
     an export-time check that emits `relu` only when provably equivalent.
3. **Depthwise weights** — CLE ✅ / per-channel shift ⬜ (§13 Q1)
   - Short term (no HW change): CLE across dw/pw pairs before calibration
     (requires the ReLU6 handling above or temporary ReLU6→ReLU during
     equalization); plus bias correction after folding.
   - Medium term (small HW change): per-output-channel `shift_out` (per-channel
     pow-2 weight frac). Measured headroom: +12 dB median weight SQNR (§5.2). This
     is the single biggest accuracy lever if TileCNN can take it.
4. **Depthwise/pointwise QAT hygiene** — sensitivity tool ✅ / decay exclusion ⬜
   - Exclude depthwise weights (and all BN-fold gammas/betas if kept) from weight
     decay.
   - Sensitivity analysis script: quantize one layer at a time (≈200 images);
     expect `features.1`, `features.13`, first conv, and linear-bottleneck outputs
     to top the list; consider first/last layer at higher effective precision only
     if the data says so.
5. **Calibration & schedule** — machinery ✅, runs ⬜
   - ≥ 500-image calibration with MSE fix-pos; PTQ eval before QAT.
   - QAT: 5–10 epochs, weight lr 1e-5 cosine, threshold lr 1e-2 halving per epoch,
     thresholds frozen for the final 2–3 epochs, batch ≥ 64 if memory allows.
6. **Residual adds** ✅: constrain both add inputs to the add's output frac during
   QAT (re-quantize the skip branch), matching hardware alignment exactly.
7. **Export validation** — software ✅ / C-sim ⬜: extend `_tilecnn_conv2d` + HW kernels to grouped conv;
   export one inverted-residual block (expand 1×1 → dw 3×3 → project 1×1 + add)
   as the canonical MobileNet testcase; per-layer parity sweep over the full net.
8. **Acceptance for this track** ⬜ (pending measurement): quantized MobileNetV2 within ~1–2% top-1 of
   float after ≤10 QAT epochs, stable (non-decaying) training curves, bit-exact
   block testcase passing in C-sim.

## 11. Implementation Roadmap

*All six phases below were executed on 2026-07-08 except where noted;
per-phase details in `improvements_2026-07.md`.*

### Phase 0 — Audit & reproducibility baseline (risk: low) — ✅ DONE (infrastructure) / ⬜ baseline numbers TBD
- **Goal**: trustworthy baselines before changing anything.
- **Files**: `tools/qat_train.py`, `tools/qat_test.py`, `tools/deploy_eval.py`, new `tools/run_baseline.py` or CLI flags.
- **Tasks**: add `--model` CLI arg (stop editing source); pin seeds; record float / PTQ-after-calibration / QAT / emu / twin top-1 for ResNet-18/50 and MobileNetV2 into a tracked `docs/baselines.md`.
- **Validation**: two consecutive runs reproduce within noise.
- **Benefit**: every later phase has a regression reference.

### Phase 1 — Quantization diagnostics & logging (risk: low) — ✅ DONE
- **Goal**: visibility.
- **Files**: new `src/fixquant/diagnostics.py`, hooks in `run_manager.py`, `tqt_quantizer.py`.
- **Tasks**: per-layer log of `log2 t`, exported frac, clip rate, weight SQNR at init and per epoch (CSV/JSON); std-vs-twin per-layer mismatch sweep tool; layer-sensitivity script.
- **Validation**: diagnostics reproduce §5.2 numbers; sensitivity ranking is stable across seeds.
- **Benefit**: turns "accuracy dropped" into "layer X clipped 30%".

### Phase 2 — Calibration & observer cleanup + training bug fixes (risk: medium) — ✅ DONE (accuracy validation pending)
- **Goal**: correct QAT dynamics.
- **Files**: `fused_conv_bn.py`, `run_manager.py`, `run_config.py`, `qat_processor.py`, `tqt_quantizer.py`, `qat_train.py`.
- **Tasks**: BN-freeze fixes (counter reset, in-place fold, explicit control); remove threshold double-decay; multi-batch calibration with MSE fix-pos (`find_fix_pos(scope>1)`); `freeze_thresholds()` schedule hook; fix argparse/LR-schedule/`_conv_forward` bugs.
- **Validation**: ResNet-18 QAT ≥ previous baseline; MobileNetV2 QAT curve no longer decays; unit test asserting optimizer still owns all trainable params after freeze.
- **Benefit**: unblocks MobileNet; likely improves all models.

### Phase 3 — MobileNet QAT stabilization (risk: medium) — ✅ DONE (code) / ⬜ accuracy target unverified
- **Goal**: MobileNetV2 within ~1–2% of float.
- **Files**: `qat_modules.py` (QReLU6 / clipped act quantizer), new `equalization.py` (CLE + bias correction), `qat_processor.py` (add-input frac constraint), config plumbing.
- **Tasks**: §10 items 2–6.
- **Validation**: PTQ-after-CLE ≥ PTQ-baseline + several %; QAT hits target; ResNet/VGG regression suite unchanged.
- **Benefit**: the headline deliverable.

### Phase 4 — TileCNN export correctness & golden tests (risk: medium/high) — ✅ DONE (software) / ⬜ HLS v1.1 kernels + C-sim
- **Goal**: MobileNet exportable; bit-exactness enforced by tests.
- **Files**: `tilecnn_exporter.py`, `fxp_emu_modules.py`, `graph_handoff_spec.md`, new `tests/`.
- **Tasks**: grouped conv + `relu6` post-op in reference/twin/spec (coordinate with HLS side); remove silent frac defaults → errors; pre-flight inspector; export-time shift-legality checks; kernel unit tests + golden logits tests (§8 strategy); fix maxpool attr hardcoding.
- **Validation**: `pytest tests/` green; inverted-residual C-sim testcase matches bit-exactly; ResNet-50 twin accuracy unchanged.
- **Benefit**: MobileNet on FPGA; refactors become safe.

### Phase 5 — Codebase cleanup & documentation (risk: low) — ✅ DONE (mixin refactor, config-driven bitwidths, outputs/ untracking deferred)
- **Goal**: maintainability for the PhD writeup and successors.
- **Files**: whole tree.
- **Tasks**: delete dead code (`tqt_ops.py` duplicate, `FXPConv2dTorch`, legacy HWCM exporter or mark deprecated, `QuantizeLayerPass`/`OutputQuantStubPass`); factor state_dict boilerplate into a mixin; make `quant_config.yaml` actually drive bitwidths/passes; update README; move smoke scripts into `tests/`; stop tracking `outputs/` artifacts.
- **Validation**: golden tests from Phase 4 still green.

### Phase 6 — Optional advanced features (risk: high, only if needed) — ⬜ OPEN
- Per-channel `shift_out` in hardware + exporter + QAT (biggest accuracy lever, needs HLS work).
- AdaQuant-style fast finetune.
- Two-step-rounding-aware fake quant (close the last ~0.3%).
- MobileNetV3 (hardswish/hardsigmoid/SE mul) per `docs/mobilenet_support_roadmap.md`.
- Higher-precision bias lane (int16) if bias correction proves insufficient.

## 12. Acceptance Criteria

*Status: the last three criteria (export, bit-exactness within software,
reproducibility, documentation) are met; the two accuracy criteria are
**pending measurement** — they require the baseline runs in `baselines.md`,
and final bit-exactness sign-off requires the HLS v1.1 kernels + C-sim.*

- ⬜ **MobileNetV2 (ImageNet-mini val)**: quantized (twin) top-1 within **2%** of the
  float baseline after ≤ 10 QAT epochs; training curve monotone-ish (no decay).
- ⬜ **No regression**: ResNet-18/50 and VGG-16 QAT + twin top-1 ≥ current recorded
  baselines (Phase 0 numbers).
- ✅ **Export**: `export_hw_testcases.py` and full-graph export succeed for ResNet-50
  *and* MobileNetV2; exporter fails loudly (never silently defaults) on unknown
  layers.
- ✅ (software) / ⬜ (C-sim) **Bit-exactness**: std-model vs digital twin per-layer parity report shows only
  explainable ±1-LSB rounding diffs; digital twin vs C-sim refs bit-exact on the
  committed testcase set; golden-logit tests pass in CI-style script.
- ✅ **Reproducibility**: one documented command each for float train, calibrate+PTQ
  eval, QAT, deploy eval, export; fixed seeds; baselines file in repo.
- ✅ **Documentation**: README matches the tree; quantization semantics (frac
  derivation, rounding, bias handling, layouts) documented in one place.

## 13. Open Questions (need project-owner / hardware decisions)

*Q6 was resolved by the rework (freeze is explicit and configurable, counted
from QAT start; fold-then-QAT available via `--cle`). Q1–Q5, Q7, Q8 remain
open and are now the main blockers outside GPU time.*

1. ⬜ **Can TileCNN support a per-output-channel `shift_out`** (per-channel pow-2
   weight scales)? This is the single biggest MobileNet accuracy lever; if no, CLE
   becomes mandatory rather than optional.
2. ⬜ **Will the hardware add a `relu6`/`clip_max` post-op** *(exporter/spec/twin ready — HLS decision pending; note `--cle` sidesteps it for MobileNetV2 by training with plain ReLU)*, or should the flow
   guarantee `6·2^frac ≥ 127` per layer (train with that constraint) and export
   plain `relu`?
3. ⬜ **Does TileCNN support grouped/depthwise convolution at all** in the current
   HLS kernels, or is depthwise planned as channel-looped standard conv? The
   exporter reference must mirror the real dataflow.
4. ⬜ **Is int8 bias a hard constraint** (URAM/format), or is an int16 bias lane
   feasible? Determines how far bias correction must carry.
5. ⬜ **Which dataset defines "accuracy"** for the thesis: full ImageNet val or
   imagenet-mini? Current numbers mix 40-batch mini-val subsets.
6. ✅ *(resolved)* **Intended BN-freeze policy**: fold-then-QAT from epoch 0 (recommended,
   Vitis-style) or trainable BN statistics for the first epochs? The 6000-step
   counter suggests the latter was intended — decide and make it explicit.
7. ⬜ **Is the legacy HWCM `export_weights_to_file` path still consumed by any HW
   flow**, or can it be deleted in favor of the graph exporter?
8. ⬜ **MobileNetV3 in scope for the thesis?** If yes, Phase 6 items (SE multiply,
   hardswish) need hardware answers too.

---

## Appendix A — Evidence collected during this analysis

*Original evidence (2026-07-07, pre-fix):*

- `torchvision` `mobilenet_v2` pretrained checkpoint: all BN
  `num_batches_tracked = 736,839` → `FusedConvBN` freeze condition true on first
  training batch (verified with torch 2.6.0 in the `Obed_Cuda` env).
- BN-folded depthwise weight statistics (17 dw layers): median per-channel
  |w|max spread 42×; worst layers `features.13.conv.1.0` (|w|max 8.17),
  `features.1.conv.0.0` (7.98) with spread ratios > 10⁶ (near-dead channels).
- Weight SQNR, best-case per-tensor pow-2 (frac = 7−ceil(log2 max|w|)) vs
  per-channel pow-2: median 31.2 dB vs 43.0 dB; worst layer 23.7 dB vs 42.6 dB.
- `test_mobilenet_qat.py` and `test_mobilenet_emu.py` both run successfully
  (structural smoke tests only).
- No experiments were run that train models or use the FPGA; PTQ/QAT accuracy
  impacts quoted from literature/measured SQNR are expectations, not measured
  end-to-end here.

*Post-rework evidence (2026-07-08):*

- All 53 tests in `tests/` pass (kernel bit-exactness across shift signs,
  groups, relu/relu6, residual adds; golden integer regression; FusedConvBN
  optimizer-ownership and fold-equivalence; calibration; end-to-end MobileNetV2
  quantize → calibrate → freeze → hardware model; export structure + legality;
  CLE exactness).
- CLE on pretrained MobileNetV2 (real weights): median depthwise per-channel
  spread 42× → **4.5×**; full `--cle` → QAT-graph → bias-correction → hardware
  model pipeline runs end to end.
- The kernel tests exposed and confirmed the `a30fc02` twin rounding
  regression (missing first `+1`), fixed against the pre-refactor
  `TileCNNConv2d` and the HLS EMIT_LOOP definition.
- Still not measured: any trained accuracy number (float / PTQ / QAT / twin) —
  see `baselines.md`.
