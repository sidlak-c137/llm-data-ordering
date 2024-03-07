#!/bin/bash
#SBATCH --job-name=ordered_train
#SBATCH --account=efml
#SBATCH --partition=gpu-a40
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time 4:00:00
#SBATCH --chdir=/mmfs1/gscratch/efml/hannahyk/nlp/cse517-final-project/
#SBATCH --output=finetune_out.txt
#SBATCH --error=finetune_err.txt

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo Lets go!
echo Training with ordered data...

export PYTHONPATH="$PYTHONPATH:nlp" # path to python env
export HF_DATASETS_CACHE="/mmfs1/gscratch/efml/hannahyk/.of_cache"
export HF_HOME="/mmfs1/gscratch/efml/hannahyk/.of_cache"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

# srun python tiny_llama_snli.py
# source ~/.bashrc
# conda deactivate
# source ./nlp-project/bin/activate
srun accelerate launch --num_processes=1 --num_machines 1 --mixed_precision 'no' --dynamo_backend 'no' ./scripts/training/train_parallel.py --config_path="./scripts/training/finetune_snli_ordered_config.json"
