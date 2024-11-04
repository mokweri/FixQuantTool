#!/bin/env bash

#SBATCH -A NAISS2024-22-1352   # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:20:00          # how long time it will take to run
#SBATCH --gpus-per-node=T4:1   # choosing no. GPUs and their type
#SBATCH -J FixQuantTool        # the jobname (not necessary)

# Set up Environment
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CMake/3.23.1-GCCcore-11.3.0
module load PyTorch-bundle/1.13.1-foss-2022a-CUDA-11.7.0
module load Horovod/0.28.1-foss-2022a-CUDA-11.7.0-PyTorch-1.13.1
module load SciPy-bundle/2022.05-foss-2022a
module load tqdm/4.64.0-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a

source /mimer/NOBACKUP/groups/naiss2024-22-1352/Obed_Work/obed_venv/bin/activate

# Set up for the different multiprocessing alternatives
ngpus=$SLURM_GPUS_ON_NODE
export WORLD_SIZE=$ngpus

# run my code
python resnetQat.py

#submit a job interactively using srun
srun -A NAISS2024-22-1352 -p alvis --gpus-per-node=A100:4 -t 01:30:00 --pty bash

srun -A NAISS2024-22-1352 -p alvis --gpus-per-node=A100:1 -t 00:20:00 --pty bash

/mimer/NOBACKUP/groups/naiss2024-22-1034/PipeCNN_Interface/dataset/imagenet 


