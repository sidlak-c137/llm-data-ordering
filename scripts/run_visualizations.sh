#!/bin/bash
#SBATCH --job-name=visualizations
#SBATCH --account=<account-name>
#SBATCH --partition=<partition>
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time 0:20:00
#SBATCH --chdir=/path/to/repository
#SBATCH --output=out.txt
#SBATCH --error=err.txt

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo Visualizing...

export PYTHONPATH="$PYTHONPATH:nlp" # path to python env
export HF_DATASETS_CACHE="/path/to/cache/"
export HF_HOME="/path/to/cache/"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

srun python scripts/visualizations.py