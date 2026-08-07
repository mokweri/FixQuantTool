# Released models

Validated model releases are stored below this directory as:

```text
<model>/<dataset>/<quantization-profile>/<version>/
```

Git tracks release manifests, model cards, metrics, checksums, calibration
reports, threshold logs, and deployment qconfigs. The heavyweight
`qat/model_best.pth.tar` files stay on Arrhenius project storage and are ignored
by Git.
