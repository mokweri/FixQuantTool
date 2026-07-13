# FusedConvBN — Conv + BatchNorm fusion for QAT

*Rewritten 2026-07: the previous version of this file documented classes
(`_ConvBnNd`, `QuantizedConvBatchNorm2d`) from a module that no longer exists.
The current implementation is `FusedConvBN` in
`src/fixquant/quantization/fused_conv_bn.py`.*

## What it is

`FusedConvBN` wraps a `nn.Conv2d` + `nn.BatchNorm2d` pair into one QAT module
so that the *folded* weights are what gets quantized — matching the hardware,
which has no BatchNorm. It owns three 8-bit `TQTQuantizer`s (weight, bias,
output activation). `ConvBnFusionPass` in `qat_processor.py` creates these
automatically for every Conv→BN pair during `QatProcessor.quantize()`.

## Forward semantics

Following Krishnamoorthi 2018 (arXiv:1806.08342 §3.2.2):

- **Training (unfrozen)** — two forward passes: batch statistics are computed
  from the unquantized conv output, the weight is scaled by
  `gamma / sigma_running` and quantized, and the output is corrected by
  `sigma_running / sigma_batch`. BN running stats keep updating.
- **Eval (unfrozen)** — weights folded with running stats, then quantized.
- **Frozen** — BN is folded permanently into `conv_mod.weight` / `conv_mod.bias`
  and the module behaves as a plain quantized conv.

## Freezing (fixed 2026-07 — the MobileNet decay bug)

Three guarantees introduced by the rework (`docs/improvements_2026-07.md` §1,
tested in `tests/test_fused_conv_bn.py`):

1. **`num_batches_tracked` is reset to 0 at fusion time.** Pretrained
   torchvision checkpoints carry values around 737k, which used to trip the
   freeze condition on the very first training batch.
2. **`freeze()` folds in place** (`weight.data.mul_`, `bias.data.copy_`), so
   the Parameter objects survive and an optimizer built before the freeze keeps
   updating them. The old implementation re-created the Parameters, silently
   detaching the whole backbone from the optimizer.
3. **A zero bias Parameter is created at fusion** if the conv has none, so the
   freeze never has to mint a new Parameter mid-training.

Auto-freeze triggers after `freeze_bn_delay` training steps *counted from the
start of QAT*. It is configured via `configs/quant_config.yaml`:

```yaml
freeze_bn_delay: 3000   # steps; null = never auto-freeze (use QatProcessor.freeze())
```

`QatProcessor.freeze()` freezes all `FusedConvBN` modules explicitly — required
before `InferProcessor` conversion/export (unfrozen modules would export
unfolded weights).

## Alternative: fold before QAT (CLE path)

With `tools/qat_train.py --cle`, BN is folded *before* quantization by
`equalization.equalize_model()` (torch.fx fuse + cross-layer equalization). In
that flow there are no `FusedConvBN` modules at all — convs carry the folded
bias and are wrapped as `QuantizedConv2d`. Recommended for MobileNet-class
models; see `docs/improvements_2026-07.md`.
