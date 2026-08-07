# FixQuant model zoo

The model zoo separates mutable experiments from immutable models approved for
hardware use.

```text
results/   -> training runs, logs, latest checkpoints, failed experiments
model_zoo/ -> candidates, validation reports, immutable versioned releases
```

The default registry is stored directly inside the FixQuant repository:

```text
<FixQuantTool>/model_zoo
```

The repository is located on Arrhenius `/nobackup`, so large model artifacts do
not consume home-directory quota. Override the location with
`FIXQUANT_ZOO_ROOT` when testing against a temporary registry or using external
storage.

Git tracks release manifests, model cards, metrics, qconfigs, and the catalog.
Checkpoint binaries and mutable `.candidates/` state are ignored. A clone can
therefore inspect the release history without downloading multi-gigabyte model
weights; the local checkpoint must be restored before evaluating that release.
Lightweight registry commands can run on the login node through
`scripts/model_zoo.sh`; they do not require a GPU allocation.

## Lifecycle

```text
QAT training
    -> candidate registration
    -> QAT evaluation on the full validation set
    -> TileCNN digital-twin evaluation
    -> checksum, sample-count, accuracy-delta, and artifact gates
    -> validated candidate
    -> explicit human promotion
    -> immutable semantic-version release
```

Automation stops at `validated`. Promotion remains a deliberate login-node
command so version assignment, unexpected tradeoffs, and the recommended model
selection remain reviewable.

## Release identity and layout

Release IDs have the form:

```text
model/dataset/quantization-profile@version
```

Example:

```text
mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0
```

The corresponding release is stored as:

```text
model_zoo/releases/mobilenet_v2/imagenet1k/int8-tqt-cle/v1.0.0/
├── manifest.yaml
├── model_card.md
├── metrics.json
├── checksums.sha256
├── run_manifest.yaml
├── qat/
│   ├── model_best.pth.tar
│   ├── calibration_report.csv
│   └── quant_thresholds.csv
├── evaluation/
│   ├── qat_metrics.json
│   ├── tilecnn_metrics.json
│   └── validation_report.json
└── deployment/
    └── qconfig.json
```

Released directories are never overwritten. Promotion stages and verifies all
files before atomically installing the final version.

## Training metadata and automatic registration

`tools/qat_train.py` writes `run_manifest.yaml` beside every future QAT run.
It records the model, dataset counts, hyperparameters, quantization settings,
Git commit, Slurm identity, best epoch, final metrics, and artifact paths.

The Arrhenius QAT job scripts automatically register a candidate after
successful training. Disable this for an exploratory run with:

```bash
sbatch --export=ALL,FIXQUANT_AUTO_REGISTER=0 <training-job.sbatch>
```

A historical or externally produced run can be registered manually:

```bash
scripts/model_zoo.sh register /path/to/run/model \
    --model resnet18 \
    --dataset-name imagenet1k \
    --dataset-path /dataset/easybuild/data/ImageNet-1k-data/20250917-hf-2025b
```

Candidate IDs default to `<model>-run-<slurm-job-id>`.

## Inspect candidates

```bash
scripts/model_zoo.sh list --candidates
scripts/model_zoo.sh show resnet18-run-1234 --candidate
```

## Automated validation

Submit validation from the login node:

```bash
sbatch scripts/jobs/validate_zoo_candidate.sbatch resnet18-run-1234
```

The job runs `qat_test.py` and `deploy_eval.py --model_type tilecnn` over the
complete validation set, writes machine-readable metrics and qconfig files,
then applies `configs/model_zoo_policy.yaml`.

Default gates require:

- the QAT and TileCNN results to reference the candidate checkpoint SHA-256;
- 50,000 validation samples;
- no more than 1.0 percentage point TileCNN top-1 loss;
- no more than 0.5 percentage point TileCNN top-5 loss;
- the genuine-best checkpoint, calibration report, threshold log, and qconfig.

A failed gate marks the candidate `rejected` and records every actual and
expected value. Corrected candidates may be evaluated again; rejected models
cannot be promoted.

## Manual promotion

After reviewing the validation report:

```bash
scripts/model_zoo.sh promote resnet18-run-1234 --version 1.0.0
```

Promotion copies artifacts into an immutable release, calculates checksums,
writes a release manifest and model card, and marks the candidate `released`.
A second attempt to create the same version fails rather than overwriting it.

Regenerate the Git-trackable catalog afterward:

```bash
scripts/model_zoo.sh catalog --output model_zoo/catalog.yaml
git add model_zoo/catalog.yaml
```

## List, resolve, and verify releases

```bash
scripts/model_zoo.sh list
scripts/model_zoo.sh show \
    mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0
scripts/model_zoo.sh resolve \
    mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0
scripts/model_zoo.sh verify \
    mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0
```

Verification recalculates every artifact checksum and reports missing or
modified files.

## Consume a released model

The evaluation tools resolve the model, checkpoint, dataset, and CLE setting
from a release ID:

```bash
python tools/qat_test.py \
    --zoo-model mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0

python tools/deploy_eval.py \
    --zoo-model mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0 \
    --model_type tilecnn
```

This prevents incompatible combinations such as loading CLE-trained weights
into a non-CLE graph.

## Current scope

Version 1 of the registry releases the genuine-best QAT checkpoint, structured
metrics, calibration and threshold diagnostics, and TileCNN qconfig. Full
TileCNN graph bundles and board-measured latency/power can be added as further
required promotion artifacts without changing existing releases.
