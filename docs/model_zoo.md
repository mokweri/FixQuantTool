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

Git tracks release manifests, model cards, metrics, qconfigs, download records,
and the catalog. Checkpoint binaries, fetched `.artifacts/`, and mutable
`.candidates/` state are ignored. A clone can therefore inspect the release
history without downloading multi-gigabyte model weights and explicitly fetch
only the model it needs. Lightweight registry commands can run on the login
node through `scripts/model_zoo.sh`; they do not require a GPU allocation.

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
    -> metadata commit
    -> draft GitHub Release upload
    -> explicit publication
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
writes a release manifest and model card, records the deterministic GitHub
Release location, and marks the candidate `released`. A second attempt to
create the same version fails rather than overwriting it.

Regenerate the Git-trackable catalog afterward:

```bash
scripts/model_zoo.sh catalog --output model_zoo/catalog.yaml
git add model_zoo/catalog.yaml
```

## Publish a checkpoint

Publishing is a deliberate login-node operation after promotion. Authenticate
the GitHub CLI once with `gh auth login`; never store a token in the repository
or a Slurm job file.

First commit and push the promoted metadata. Then bind the binary release to
that exact FixQuant commit and create it as a draft:

```bash
scripts/model_zoo.sh publish \
    resnet18/imagenet1k/int8-tqt@v1.0.0 \
    --target <full-40-character-FixQuant-commit>
```

`publish` verifies the local checkpoint against the manifest, checks the
canonical tag, asset, URL, and size, confirms that the target commit contains
the exact current manifest, and calls `gh release create --draft`.
Inspect the draft on GitHub and publish it explicitly:

```bash
gh release edit \
    model-zoo-resnet18-imagenet1k-int8-tqt-v1.0.0 \
    --repo mokweri/FixQuantTool \
    --draft=false
```

Use `publish ... --dry-run` to inspect the generated GitHub CLI command without
changing GitHub. Releases promoted before download metadata was introduced can
be prepared once before their metadata commit:

```bash
scripts/model_zoo.sh prepare-publish \
    resnet50/imagenet1k/int8-tqt@v1.0.0
```

Every different checkpoint receives a new semantic version. Never replace an
asset belonging to an existing published version.

## Fetch a checkpoint

Fetch is explicit and does not run silently during export or evaluation:

```bash
scripts/model_zoo.sh fetch \
    resnet50/imagenet1k/int8-tqt@v1.0.0
```

The command reuses a verified local release checkpoint or cached copy. A
missing payload is streamed to a temporary file, checked for exact size and
SHA-256, and atomically installed under `model_zoo/.artifacts/`. Override that
location with `--cache-dir` or `FIXQUANT_ZOO_CACHE`. Consumers resolve the
verified cache automatically.

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

Fetch the checkpoint, then let the evaluation tools resolve the model,
checkpoint, dataset, and CLE setting from the same release ID:

```bash
scripts/model_zoo.sh fetch \
    mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0

python tools/qat_test.py \
    --zoo-model mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0

python tools/deploy_eval.py \
    --zoo-model mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0 \
    --model_type tilecnn

python tools/export_tilecnn_graph.py \
    --zoo-model mobilenet_v2/imagenet1k/int8-tqt-cle@v1.0.0 \
    --out_dir outputs/mobilenet_v2_int8_tilecnn
```

This prevents incompatible combinations such as loading CLE-trained weights
into a non-CLE graph. The graph exporter resolves the model, checkpoint, and
CLE setting from the same release record used by the evaluation tools. It
verifies the released checkpoint checksum and writes a
`tilecnn.model-package.v1` `manifest.json` into the exported package with the
release, producer, preprocessing, graph, input, and reference provenance.

Dataset locations are machine-local. Selection uses the following precedence:

1. explicit `--dataroot`;
2. `FIXQUANT_DATA_DIR` on the current machine;
3. the original path recorded in the release;
4. the tool's local fallback.

For example, after pulling the repository onto an accelerator PC:

```bash
export FIXQUANT_DATA_DIR=/home/obed/datasets/imagenet
python tools/deploy_eval.py \
    --zoo-model resnet50/imagenet1k/int8-tqt@v1.0.0 \
    --model_type tilecnn
```

An explicit `--dataroot` overrides the environment for one invocation.

## Current scope

Version 1 of the registry releases the genuine-best QAT checkpoint, structured
metrics, calibration and threshold diagnostics, and TileCNN qconfig. Full
TileCNN graph bundles and board-measured latency/power can be added as further
required promotion artifacts without changing existing releases.
