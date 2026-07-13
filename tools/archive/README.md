# Archived tools

Not maintained; kept for reference. They predate the 2026-07 rework
(`docs/improvements_2026-07.md`) and were not updated for the current CLI
conventions (`--model`, per-model checkpoint paths) or verified against the
reworked APIs.

- `hw_layer_test_gen.py` + `gen_resnet18_fc_testdata.py` — generate the old
  per-layer handoff format (`weights.data` / `qparams.json` / `details.txt`
  under `outputs/hw_data_files`). Superseded by the graph.json bundles of
  `graph_handoff_spec.md` (`tools/export_hw_testcases.py`,
  `tools/export_tilecnn_graph.py`).
- `train_cifar.py` — standalone CIFAR float trainer; the TileCNN flow targets
  ImageNet, and CIFAR is not part of the baselines.
- `ddp_train_hvd.py` — standalone Horovod ImageNet trainer for the Alvis
  cluster. `scripts/jobscript.sh` references a `run_dist2.py` that no longer
  exists, so the HPC entry point needs rewiring anyway if cluster training
  resumes.
