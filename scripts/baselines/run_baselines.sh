#!/bin/bash
#SBATCH --account=<acc>
#SBATCH --partition=<partition>
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --time 1:00:00

#SBATCH --chdir=/gscratch/ml4ml/sidlak/cse517-final-project
#SBATCH --output=finetune_out.txt
#SBATCH --error=finetune_err.txt

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

export PYTHONPATH="$PYTHONPATH:nlp" # path to python env
export HF_DATASETS_CACHE="/gscratch/ml4ml/sidlak/.my_cache"
export TRANSFORMERS_CACHE="/gscratch/ml4ml/sidlak/.my_cache"
export HF_HOME="/gscratch/ml4ml/sidlak/.my_cache"

# srun python tiny_llama_snli.py
srun torchrun tiny_llama_snli.py
