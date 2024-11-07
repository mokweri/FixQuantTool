#!/bin/env bash

#SBATCH -A NAISS2024-22-1352     # find your project with the "projinfo" command
#SBATCH -p alvis                 # what partition to use (usually not necessary)
#SBATCH -t 1-12:00:00            # how long time it will take to run
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=A100:4   # choosing no. GPUs and their type
#SBATCH -J FixQuantTool          # the jobname (not necessary)
#SBATCH -D /mimer/NOBACKUP/groups/naiss2024-22-1352/Obed_Work/FixQuantTool

# export PS1="non-interactive"
# export OMPI_MCA_plm=slurm
# export OMPI_MCA_btl_vader_single_copy_mechanism=none

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

ngpus=$SLURM_GPUS_ON_NODE
export WORLD_SIZE=$ngpus

# run my code
mpirun -np 4 python run_dist2.py