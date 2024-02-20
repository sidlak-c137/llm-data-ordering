#!/bin/bash
#SBATCH --account=efml
#SBATCH --partition=gpu-a40
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time 2:00:00

#SBATCH --chdir=/mmfs1/gscratch/efml/hannahyk/nlp/cse517-final-project/
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
export HF_DATASETS_CACHE="/mmfs1/gscratch/efml/hannahyk/.of_cache" # path to HF cache; should be under /gscratch
export TRANSFORMERS_CACHE="/mmfs1/gscratch/efml/hannahyk/.of_cache" # ""
export HF_HOME="/mmfs1/gscratch/efml/hannahyk/.of_cache" # ""

# srun python tiny_llama_train.py
srun torchrun ./scripts/training/train.py --config_path="./scripts/training/finetune_snli_config.json"
