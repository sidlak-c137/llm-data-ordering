echo Lets go!

export PYTHONPATH="$PYTHONPATH:nlp" # path to python env
export HF_DATASETS_CACHE="/mmfs1/gscratch/ml4ml/sidlak/.my_cache"
export HF_HOME="/mmfs1/gscratch/ml4ml/sidlak/.my_cache"
export TOKENIZERS_PARALLELISM=true

# srun python tiny_llama_snli.py
srun python tiny_llama_snli.py
