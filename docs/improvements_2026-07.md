# 2026-07 Rework: Phases 0–5

Implementation of the roadmap in `quantization_repo_analysis_and_roadmap.md`
(Phases 0–5). This doc records what changed, why, and how to use it.
Run `python -m pytest tests/` to verify everything (53 tests).

## Bug fixes (the important ones)

1. **BN freeze fired on the first QAT batch and silently froze all conv weights**
   (`fused_conv_bn.py`). Pretrained checkpoints carry
   `num_batches_tracked ≈ 737k`, which exceeded `FREEZE_BN_DELAY` immediately;
   worse, `freeze()` re-created the weight/bias `nn.Parameter`s, so the
   already-built optimizer stopped updating them. This is the primary cause of
   the MobileNet QAT accuracy decay.
   *Fix:* the counter is reset when `FusedConvBN` wraps a conv, the fold happens
   in place (Parameter identity preserved — the optimizer keeps working), a zero
   bias Parameter is created at fusion time so freeze never mints a new one, and
   the delay is configurable (`quant_config.yaml: freeze_bn_delay`, `null` = manual).
   Tests: `tests/test_fused_conv_bn.py`.

2. **Digital-twin rounding regression** (`fxp_emu_modules.py`). The `a30fc02`
   refactor dropped the first `+1` of the two-step EMIT_LOOP rounding in
   `HardwareConv2d`/`HardwareLinear`, so the twin no longer matched the exporter
   reference / HLS. Restored; `HardwareElementwiseAdd` now uses the same
   `_signed_shift` rounding as the residual path. The twin and the exporter
   reference kernels are now bit-identical (enforced by `tests/test_kernels.py`
   and the golden test `tests/test_golden.py`).

3. **TQT thresholds were regularized twice** (`run_manager.py`): Adam
   `weight_decay` *and* the explicit TQT norm penalty both pulled every
   `log_threshold` toward 0. The quantizer param group now has
   `weight_decay=0`; the TQT paper penalty stays.

4. **ReLU6 was exported as plain ReLU** and **depthwise conv was rejected** by
   the exporter reference. Both fixed; see "Export" below.

5. Smaller: `--quantizer_lr_decay` was `type=int` (0.5 became unusable from the
   CLI); the QAT LR decay only applied on exact step multiples; `QuantizedConv2d`
   ignored non-zero `padding_mode`; functional `adaptive_avg_pool2d` (used by
   torchvision MobileNetV2) was never quantized (new `ReplaceFunctionalPoolPass`).

## Calibration (Phase 2)

`QatProcessor.calibrate(loader, device, max_batches, scope)` is now real
multi-batch calibration: every `TQTQuantizer` collects a value reservoir over
the batches and picks its fractional position by the MSE fix-pos search
(`fix_ops.find_fix_pos`, the Vitis "diffs" method, `scope=5` by default).
Previously only the first batch mattered (one-shot warmup init).
Returns `{quantizer_name: frac}`.

## QAT schedule (Phase 2)

- `RunConfig.threshold_freeze_frac` (default 0.7, CLI
  `--threshold_freeze_frac`): TQT thresholds are frozen after that fraction of
  epochs so exported frac bits are stable. `RunManager.freeze_thresholds()` for
  manual control.
- Per-epoch threshold/frac state is appended to
  `<save_dir>/logs/quant_thresholds.csv`.

## ReLU/ReLU6-aware activation ranges (Phase 3)

New `ActRangePass`: a conv whose only consumer is ReLU6 gets
`act_quantizer.bounded_range = (0, 6)` (ReLU: `(0, ∞)`). Warmup initializes the
threshold at the bound and calibration samples are clipped to it, so the
quantizer spends its range on values that survive the activation.

## Cross-layer equalization + bias correction (Phase 3)

`fixquant/quantization/equalization.py`:

- `equalize_model(model)` — BN-fold (`torch.fx` fuse) → ReLU6→ReLU → CLE over
  conv→relu→conv pairs (depthwise-aware). Use for MobileNet-class models to fix
  the ~40× per-channel depthwise range spread under per-tensor scales.
  QAT then runs on the BN-free model (`QuantizedConv2d` path, no `FusedConvBN`).
- `apply_bias_correction(float_ref, qat_model, loader, device)` — one-shot
  per-channel output-mean correction after calibration.
- CLI: `python tools/qat_train.py --model mobilenet_v2 --cle [--bias_corr]`.
- Note: after ReLU6→ReLU replacement the deployed graph uses plain `relu` +
  int8 saturation, which is self-consistent (QAT trains the same function).
  Without `--cle`, ReLU6 is kept and exported as a true `relu6` post-op.

`QElementwiseAdd` now rounds both residual inputs onto the output frac grid
before the add (matching the hardware alignment shift).

## Export correctness (Phase 4)

- `_tilecnn_conv2d` supports `groups` (depthwise); `relu6` /
  `post_add_relu6` post-ops clamp at `min(127, round(6·2^frac_out))`;
  maxpool reference reads kernel/stride/padding from node attrs (was hardcoded
  3×3/s2/p1). Spec: `graph_handoff_spec.md` §"Schema v1.1 extensions" —
  the hardware importer must adopt these for MobileNet.
- `convert_to_hardware_model` no longer falls back to `frac=5`/`frac_w=7` on a
  missing qconfig entry — it raises. The network input frac is captured from
  the input QuantStub instead of being hardcoded to 5.
- `HardwareMaxPool2d` gets its `post_pool_shift` from the qconfig
  (was always 0 — mismatched the exporter refs whenever frac_in ≠ frac_out).
- Export-time shift-legality checks (`_check_shift_legality`) reject insane
  `shift_out` / `bias_shift` / GAP shifts before artifacts reach the FPGA.
- `preflight_check(model)` (in `qat_processor.py`) validates a float model
  against the supported-op set before quantization.

## Tests (Phase 4)

`tests/` (pytest; `pip install pytest` in the env):

- `test_kernels.py` — twin vs exporter-reference bit-exactness (all shift
  signs, groups, relu/relu6, residual add, linear, GAP, maxpool).
- `test_golden.py` — committed integer golden for a kernel chain
  (`tests/golden/`); regenerate only on deliberate semantics changes with
  `FIXQUANT_REGEN_GOLDEN=1`.
- `test_fused_conv_bn.py`, `test_calibration.py` — the Phase 2 fixes.
- `test_qat_flow.py` — end-to-end structure for a depthwise/ReLU6 block net and
  torchvision MobileNetV2 (marked `slow`), plus a QAT-vs-hardware parity sweep.
- `test_export.py` — full export of a depthwise/ReLU6 model, relu6/groups in
  graph.json, legality, CLE numerics.

## Diagnostics (Phase 1)

- `fixquant/diagnostics.py`: `quantizer_report` (log2 t, frac, weight
  SQNR/clip-rate per quantizer), `write_report_csv`, `log_quantizer_state`,
  `parity_sweep(qat_model, hw_model, input)`.
- `tools/layer_sensitivity.py`: quantize-one-layer-at-a-time top-1 probes.
- `qat_train.py` writes `<save_dir>/<model>_calib_report.csv` after calibration.

## Reproducibility & cleanup (Phases 0, 5)

- All tools take `--model` (resnet18|resnet50|vgg16|mobilenet_v2) and seed via
  `--manual_seed`; checkpoints live under `qat_models/<model>/`.
  `qat_test.py`/`deploy_eval.py` take `--checkpoint`.
- Baselines protocol + table: `docs/baselines.md`.
- Deleted dead/broken code: `tqt_ops.py` (duplicate threshold init),
  `QAvgPool2d` (broken `from_float`), `QuantizeLayerPass`/`OutputQuantStubPass`
  (never wired), `FXPConv2dTorch`/`HLSMaxPool2D`/`HLSAdaptiveAvgPool2d`
  (unused), `tools/hw_fxp_test.py` (imported modules deleted in `a30fc02`),
  root smoke scripts (superseded by `tests/`).
- Retired to `tools/archive/` (see its README): `hw_layer_test_gen.py` +
  `gen_resnet18_fc_testdata.py` (pre-spec per-layer handoff format),
  `train_cifar.py` (CIFAR not in the TileCNN flow), `ddp_train_hvd.py`
  (Horovod trainer; the HPC jobscript entry point is stale).
- Remaining tools updated to current conventions: `print_model_graph.py` and
  the two testcase exporters take `--model`/new checkpoint paths, load weights
  *before* freezing, and derive the input frac from the QuantStub; `train.py`
  is now a working float baseline trainer (`--model`, `--pretrained`,
  `--eval_only`; its `train()` call had been commented out).
- Legacy docs brought in line with the code: `conv_fused.md` rewritten (it
  documented a long-deleted module), `DEPLOY.md` rewritten (deploy_eval is an
  evaluation script, not a parameter extractor), `tilecnn_exporter_and_digital_twin.md`
  updated to the `convert_to_hardware_model` / `Hardware*` APIs, `tqt.md` /
  `qmodules.md` / `QAT.md` updated for calibration, bounded ranges, add
  alignment and the current CLI; `mobilenet_support_roadmap.md` carries a
  status banner; stale `layers.txt` removed.

## Follow-up fixes (2026-07-13)

- **`--cle` now propagates to the eval/export tools.** A checkpoint trained with
  `qat_train.py --cle` is BN-free (`QuantizedConv2d`, no `conv_mod`/`bn_mod`) and
  would not load into the default Conv-BN-fused model. `qat_test.py`,
  `deploy_eval.py`, `export_tilecnn_graph.py`, `print_model_graph.py`, and the two
  hw-testcase exporters all take `--cle`, applying `equalize_model` before
  `quantize()` so the architecture matches. `load_qat_weights` detects the
  BN-free vs BN-fused mismatch and raises a one-line hint instead of a wall of
  key errors. Verified: `qat_test.py --model mobilenet_v2 --cle` on the
  `--cle`-trained checkpoint gives 69.9/89.2 (matches its recorded best_acc).
- **`diagnostics.quantizer_report` no longer crashes on all-zero bias tensors.**
  `FusedConvBN` creates a zero bias Parameter for bias-free convs; weight SQNR of
  an all-zero tensor did `log10(0)`. Now returns `-inf` / writes `zero-tensor`.
  Regression test: `tests/test_diagnostics.py`.
- **`FusedConvBN.to_qconv` fixed** (`NameError: QuantizedConv2d`) — it now imports
  the class lazily, so the compact-model path (`ModuleReplacementPass("compact")`)
  works.
- **Requant rounding-bias fix (hardware + twin + exporter).** The TileCNN two-step
  requant applied *two* `+1`s (`s1 = (acc>>(shift-1))+1` and `out=(s1+bias+1)>>1`)
  where correct round-half-up needs only the second one. The extra `+1` added a
  constant **+0.5/layer** output bias that compounded with depth — negligible for
  ResNet (~3 pts) but ~14 pts for MobileNet-V2. Confirmed against the HLS source
  (`output_postproc.cpp`), which had the same two `+1`s. Removed the `s1` `+1` in
  lockstep across: the HLS kernel (`output_postproc.cpp` acc_quantize + emit_tile,
  `dw_stage.cpp`), the HLS runtime reference (`tilecnn_utils/runtime_reference.cpp`
  conv + dw), the digital twin (`fxp_emu_modules.py` HardwareConv2d/HardwareLinear),
  and the exporter reference (`tilecnn_exporter.py` conv + linear). Golden
  regenerated (`tests/test_golden.py`). Result: twin now matches QAT — MobileNet-V2
  56.0→**69.7** (QAT 69.9), ResNet-50 69.5→**72.5** (QAT 72.6). The HLS change needs
  a bitstream re-synthesis (trivial logic, removes an adder input). Residual-add /
  GAP / MaxPool use a different (round-half-away-from-zero) convention and were not
  affected.

## Pre-rework checkpoint loading (2026-07-13)

Unfrozen ResNet checkpoints saved before the "create a zero `conv_mod.bias` at
fusion" change omit `conv_mod.bias` for bias-free convs, so the current strict
load rejected them (e.g. resnet18). `FusedConvBN._load_from_state_dict` now
injects the model's own zero bias into the state dict when the checkpoint omits
it (the BN offset is folded into that bias at `freeze()`), so these load again
without loosening strictness. Regression (2026-07-13, pre-rework checkpoints
through the current pipeline): resnet18 QAT 69.5/88.7, twin 68.9/88.5; vgg16 QAT
69.6/90.3, twin 69.9/90.3 — twins track their QAT models.

## Not done here (needs decisions / hardware work)

- Hardware-side grouped-conv and relu6 support (spec v1.1) — HLS kernels.
- Per-output-channel `shift_out` (per-channel weight scales) — biggest accuracy
  lever if the hardware can take it.
- Re-measuring all baselines (`docs/baselines.md` is scaffolded with TBDs —
  requires GPU time + dataset).
- `outputs/` and `hw_data_files/` (~40 MB of artifacts) are still tracked in
  git; untrack manually if desired:
  `git rm -r --cached outputs hw_data_files && echo -e "outputs/\nhw_data_files/" >> .gitignore`.
