echo Lets go!

export PYTHONPATH="$PYTHONPATH:nlp" # path to python env
export HF_DATASETS_CACHE="/mmfs1/gscratch/ml4ml/sidlak/.my_cache"
export HF_HOME="/mmfs1/gscratch/ml4ml/sidlak/.my_cache"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1

# srun python tiny_llama_snli.py
srun accelerate launch --multi_gpu --num_processes=2 --num_machines 1 --mixed_precision 'no' --dynamo_backend 'no' ./scripts/training/train_parallel.py --config_path="./scripts/training/finetune_snli_config.json"
