# N:M weight sparsity during QAT

`fixquant.sparsity.nm` prunes half of each selected layer's weights with a
fixed N:M mask and keeps the mask through quantization-aware training. It is
the training side of DeepTile's 2:4 sparsity study; the hardware does not use
it yet.

## Patterns

| Pattern | Shape of the choice |
|---|---|
| `filter24` | 2 of every 4 consecutive input channels, chosen per output row and tap |
| `pair24` | the same, one choice shared by two consecutive output rows |
| `oc_n16` | 8 of every 16 consecutive output channels, per input channel and tap |
| `filter24+perm`, `pair24+perm` | after a channel permutation that keeps more magnitude |

The pruned layers are every non-grouped convolution except one reading the
3-channel image, and every linear layer; `--nm_layers` restricts them. Masks
are chosen by magnitude on the float model before BatchNorm folding and
cross-layer equalization, which scale weights per channel and so keep the
zeros.

**Permutations are real reorders.** One permutation is searched per *channel
class*: the tensors that must share a channel order because they are joined
by an add or pass through BatchNorm, activations, pooling or a depthwise
convolution. It is searched against every pruned layer reading the class,
and it also orders the output rows of the layers writing it.
`nm.permute_channels` applies the permutations to a float model; the network
then computes the same function and the pruned weights show the plain
pattern in storage order (`tests/test_nm_sparsity.py`).

## Training

```bash
python tools/qat_train.py --model resnet18 --nm_pattern pair24+perm ...
```

The run directory gets `nm_masks.pt` (masks, permutations, channel classes)
and `nm_report.json` (kept magnitude per layer). The masks are re-applied
after every optimizer step; at the end the best checkpoint is checked against
them (`sparsity.verification` in `run_manifest.yaml`) and the run fails if
any masked weight is nonzero.

On Arrhenius, `scripts/nm_submit.sh MODEL PATTERN EPOCHS LR [SEED] [LAYERS.json]`
submits `scripts/jobs/qat_nm.sbatch`; `dense` as the pattern trains the
reference with the same schedule. Runs land in
`results/qat-nm/<model>/<pattern>-lr<lr>-e<epochs>-s<seed>/<job>/`.
