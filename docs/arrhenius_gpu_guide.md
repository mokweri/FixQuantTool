# Running FixQuantTool on Arrhenius (NAISS) — GPU Reference Guide

*Compiled 2026-07-08 from the NAISS Arrhenius documentation and official NAISS/SUPR
pages. The user documentation itself
(`hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/`) sits behind
NAISS GitLab authentication; items I could not verify from public sources are
marked **[verify on system]**. Cross-check against the quickstart once you can
log in — the system only reached general availability in June 2026 and details
may still change.*

---

## 1. What Arrhenius is

Arrhenius is the new NAISS/EuroHPC flagship system hosted at NSC (Linköping
University), built by HPE, general availability June 2026. It replaces the
national systems you may know: **Alvis migrates to Arrhenius in Summer 2026,
Tetralith in Fall, Dardel at the end of the year** — so this is where the
FixQuantTool training moves.

| Partition | Nodes | Hardware |
|---|---|---|
| `gpu` | 382 | 4× NVIDIA **GH200** superchips per node. Each superchip = 72-core Grace **ARM** CPU + Hopper GPU with **96 GB HBM3** + 128 GB LPDDR5X, coherently connected. ~1.8 TB local NVMe per node. |
| `cpu` | 192 | 2× AMD EPYC 9755 "Turin" (128 cores each), 768 GB RAM, 1.8 TB NVMe |
| `fat` | 20 | 2× AMD EPYC 9745, **3 TB** RAM, 1.8 TB NVMe |

Interconnect: HPE Slingshot (4× 200 Gb/s per GPU node). Aggregate GPU
performance ~66.8 PFLOPS (~7× Dardel).

**Storage** (shared Lustre):

| Tier | Size | Performance | Project path |
|---|---|---|---|
| Disk | 25 PB | 260 GiB/s | `/nobackup/proj/disk/<project-dir>` |
| Flash | 2 PB | 15M IOPS (4K random read) | `/nobackup/proj/flash/<project-dir>` |

The project directory name is chosen per project (often the SUPR id, e.g.
`naiss2026-1-234`, or a group name). **When you SSH in, the login message lists
"Project storage directories available to you."** Note the `/nobackup` prefix —
these are not backed up; keep code in git and copy final checkpoints somewhere
safe. Home directory quota/backup policy: **[verify on system]**.

## 2. The one thing that changes everything: aarch64

**The GPU compute nodes are ARM (aarch64). The login nodes are x86_64.**
Consequences:

- Your local `Obed_Cuda` conda env, any x86 wheels, and any x86 containers
  **will not run** on the GPU nodes. Everything must be (re)installed for
  aarch64.
- `pip install torch` from default PyPI on aarch64 installs a **CPU-only**
  PyTorch. You must either use the CUDA wheel index
  (`pip install torch --index-url https://download.pytorch.org/whl/cu128` —
  pick the CUDA version matching the system driver) or, better, use an
  **NVIDIA NGC PyTorch container**, which ships aarch64+CUDA builds and is the
  recommended route on GH200 systems.
- Anything you compile (and any pip package that builds from source) must be
  built **on a GPU node**, not on the login node. Modules prefixed `GPU/` only
  work on GPU nodes, so software building for the GPU partition is done inside
  an interactive GPU allocation.

## 3. Getting access

1. Be on a NAISS project with an **Arrhenius GPU** allocation (SUPR resource
   "Arrhenius GPU @ NAISS"). Rounds: NAISS Small (PhD student + named
   supervisor is enough), Medium, Large.
2. In [SUPR](https://supr.naiss.se) → Accounts → **"Request Account on
   Arrhenius HPC @ NAISS"**. Existing NAISS usernames are reused. You choose a
   password and set up **two-factor authentication**.
3. SSH to the login node. Exact hostname: shown in SUPR / the quickstart
   **[verify on system]** (expect something like `arrhenius.nsc.liu.se`).

## 4. SLURM on Arrhenius

### Accounts and partitions

- Every job needs a project account: `-A <project>` where the account name
  **ends in `-gpu` for the `gpu` partition** and `-cpu` for the CPU/fat
  partitions (e.g. `-A naiss2026-22-123-gpu`). Check yours with
  `projinfo` / SUPR **[verify command on system]**.
- Partitions: `-p gpu`, `-p cpu`, `-p fat`.
- **Max walltime: 72 h (3 days)**; confirm current limits with `sinfo`.

### Resource allocation model

The documented pattern (do **not** request whole nodes unless you need them):

```
CPU jobs:  -n <num_tasks> -c <cores_per_task>
GPU jobs:  -n <num_tasks> -c <cores_per_task> --gpus <num_tasks>
```

Each GH200 pairs one GPU with a 72-core Grace CPU, so the natural "full
superchip" request is `-c 72` per GPU task. The documentation's own example of
a 2-node, 8-GPU job:

```bash
#SBATCH -n 8          # 8 tasks
#SBATCH -c 72         # 72 Grace cores per task (one full superchip each)
#SBATCH --gpus 8      # one GPU per task
#SBATCH -t 10         # walltime (minutes here; use D-HH:MM:SS in practice)
#SBATCH -A naiss<project>-gpu
#SBATCH -p gpu
```

For a **single-GPU job** (the FixQuantTool case): `-n 1 --gpus 1` and up to
`-c 72`. Billing granularity (whether you're charged per GPU-hour regardless of
`-c`): **[verify on system]**.

### Interactive work (and building software)

```bash
interactive -p gpu --gpus 1 -A <project>-gpu -t 01:00:00
```

`interactive` is a wrapper around `salloc`+`srun` that gives you a shell **on
the compute node** — this is where you build aarch64 software, create venvs,
and pull/test containers. Use it for anything that compiles; never build for
the GPU partition on the x86 login node.

### Monitoring

`squeue --me`, `scontrol show job <id>`, `sacct -j <id>`, and `sinfo` for
partition/walltime status. On-node: `nvidia-smi` inside the job. Job efficiency
tooling (`seff`/`jobstats`): **[verify on system]**.

## 5. Software strategy for FixQuantTool

Two workable routes, in order of preference:

### Route A — NGC PyTorch container via Apptainer (recommended)

Apptainer is available on Arrhenius. NGC PyTorch images ship aarch64 + CUDA and
are NVIDIA's supported path on GH200:

```bash
# on a GPU node (interactive allocation), once:
cd /nobackup/proj/disk/<project-dir>/containers
apptainer pull pytorch_25.06.sif docker://nvcr.io/nvidia/pytorch:25.06-py3
# then install fixquant into the container's python at runtime (bind-mounted):
apptainer exec --nv pytorch_25.06.sif pip install --user -e /path/to/FixQuantTool
```

Run with `apptainer exec --nv <sif> python ...`. `--nv` exposes the GPU.
`pip install --user` lands in `~/.local` — fine, or use
`--env PYTHONUSERBASE=<proj>/pyuser` to keep it in project storage.

### Route B — module + venv with aarch64 CUDA wheels

```bash
# on a GPU node:
module avail                    # look for GPU/ prefixed Python/PyTorch modules
module load GPU/<python-or-buildenv-module>     # [verify names on system]
python -m venv /nobackup/proj/disk/<project-dir>/venvs/fixquant
source .../venvs/fixquant/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e /path/to/FixQuantTool && pip install pytest pyyaml onnx tqdm
```

If the site provides a ready PyTorch module under `GPU/`, prefer that over
self-installed wheels. Either way, **verify the install on the node**:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: ... True  GH200 / 'NVIDIA GH200 ...'
python -m pytest /path/to/FixQuantTool/tests -q -m "not slow"   # 52 fast tests, CPU-only, ~2 s
```

The test suite is pure-integer/CPU and portable to aarch64 — it is the cheapest
way to confirm the environment is sane before burning GPU hours.

## 6. Data and I/O layout for our jobs

- **Code**: clone the repo into `/nobackup/proj/disk/<project-dir>/FixQuantTool`
  (not `$HOME` — keep home small).
- **Dataset** (ImageNet/imagenet-mini): store the tarball or extracted tree on
  project Disk. If your job hammers many small files, either use the **Flash**
  tier or (better) copy/extract to the **node-local NVMe (~1.8 TB)** at job
  start and read from there. Node-local scratch path/env var
  (`$SNIC_TMP`/`$TMPDIR`-style): **[verify on system]** — the template below
  uses `${TMPDIR}` with a fallback.
- **Checkpoints/outputs**: write to project Disk (`qat_models/`, `outputs/`).
  Remember: `/nobackup` = no backup.
- Set `FIXQUANT_DATA_DIR` to wherever the dataset ends up; every tool honors it.

## 7. Sizing FixQuantTool jobs

- `qat_train.py` is **single-process, single-GPU** (the Horovod path is
  archived). Request `-n 1 --gpus 1`. One GH200 (96 GB HBM) is far more than
  the ~8-16 GB these models need — batch size can go up (e.g. 128–256 for
  MobileNetV2 at 224²) and `--n_worker` up to ~16–32 given 72 Grace cores.
- A 10-epoch MobileNetV2 QAT run on imagenet-mini is a modest job — hours, not
  days; `-t 08:00:00` with checkpointing (RunManager saves every epoch) is a
  sensible starting point. Full-ImageNet runs: budget within the 72 h cap and
  rely on the per-epoch checkpoints for restarts.
- Because CPU-heavy work (calibration reservoir sampling, data loading) runs on
  the Grace cores, ask for a healthy `-c` (16–72). Unused cores on your
  superchip may be wasted for others, so don't ask for 72 unless dataloading
  actually needs it — start with `-c 16` and check utilization.

## 8. Known unknowns to resolve on first login

1. Login hostname; whether 2FA is per-session.
2. Exact module names for Python/PyTorch/CUDA (`module avail`, `GPU/` prefix).
3. Node-local scratch env var and path.
4. Billing granularity (GPU-hour vs core-hour) and `projinfo` equivalent.
5. Whether `interactive` requires explicit `-A`/`-t` defaults.
6. Home quota; whether `/nobackup/proj/...` quotas are per Disk/Flash grant
   (SUPR lists "Arrhenius Storage" as a separate resource).

## 9. Template job script

See [`scripts/jobscript_arrhenius.sh`](../scripts/jobscript_arrhenius.sh)
(replaces the old Alvis `jobscript.sh`, which targeted A100s + Horovod and a
deleted entry script). Submit with:

```bash
sbatch scripts/jobscript_arrhenius.sh                       # defaults: mobilenet_v2 + --cle
MODEL=resnet50 EXTRA_ARGS= sbatch scripts/jobscript_arrhenius.sh
```

---

### Sources

- [Arrhenius quickstart (NAISS user docs — auth-gated; content via search index)](https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/basics/quickstart/)
- [Arrhenius technical description (naiss.se)](https://www.naiss.se/resources/arrhenius-technical-description/)
- [Arrhenius resource page (naiss.se)](https://www.naiss.se/resource/arrhenius/)
- [Arrhenius GPU @ NAISS (SUPR)](https://supr.naiss.se/resource/arrhenius-gpu/)
- [NAISS Slurm training — Arrhenius](https://hpc.pages.naiss.se/training/NAISS_Slurm/arrhenius/)
- [PyTorch aarch64 GPU wheels discussion (pytorch/pytorch#160162)](https://github.com/pytorch/pytorch/issues/160162)
