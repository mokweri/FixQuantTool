# FixQuant model-zoo metadata

This directory is the default FixQuant model-zoo registry. It contains mutable
local candidates plus immutable model releases. Lightweight release metadata,
metrics, model cards, and deployment qconfigs are reviewable in Git. Large
`model_best.pth.tar` checkpoint files are excluded from Git and may be restored
from checksum-pinned GitHub Release assets.

```text
model_zoo/
├── README.md
├── catalog.yaml
├── .candidates/                    # local validation state; Git-ignored
├── .artifacts/                     # fetched checkpoints; Git-ignored
└── releases/
    └── <model>/<dataset>/<profile>/<version>/
```

Because the FixQuant repository itself lives on `/nobackup`, local release and
cache payloads do not consume home-directory quota. `FIXQUANT_ZOO_ROOT` changes
the registry location and `FIXQUANT_ZOO_CACHE` changes the fetched-artifact
cache.

Fetch a published checkpoint explicitly before using a clean checkout:

```bash
scripts/model_zoo.sh fetch \
    resnet50/imagenet1k/int8-tqt@v1.0.0
```

Regenerate the catalog after a promotion:

```bash
scripts/model_zoo.sh catalog --output model_zoo/catalog.yaml
```

Commit catalog changes together with the promoted release manifest, metrics,
qconfig, and model card. Publish checkpoint binaries as GitHub Release assets;
never force-add them to Git.
