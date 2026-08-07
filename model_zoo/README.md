# FixQuant model-zoo metadata

This directory is the default FixQuant model-zoo registry. It contains mutable
local candidates plus immutable model releases. Lightweight release metadata,
metrics, model cards, and deployment qconfigs are reviewable in Git. Large
`model_best.pth.tar` checkpoint files remain on Arrhenius project storage and
are excluded from Git.

```text
model_zoo/
├── README.md
├── catalog.yaml
├── .candidates/                    # local validation state; Git-ignored
└── releases/
    └── <model>/<dataset>/<profile>/<version>/
```

Because the FixQuant repository itself lives on `/nobackup`, checkpoint files
stored here do not consume home-directory quota. `FIXQUANT_ZOO_ROOT` can still
override the location for tests or another installation.

Regenerate the catalog after a promotion:

```bash
scripts/model_zoo.sh catalog --output model_zoo/catalog.yaml
```

Commit catalog changes together with the promoted release manifest, metrics,
qconfig, and model card. Never force-add checkpoint binaries to Git.
