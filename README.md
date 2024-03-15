# cse517-final-project

In this project, we study whether ordering the data beased on various "hardness" metrics affects the accuracy and efficiency of training.

## Environment Setup

Finetuning and evaluating the TinyLlama Model on SNLI and GSM8k data requires certain packages. The ```requirements.txt``` file lists what is needed.

For conda environment setup:
```
conda env create -f environment.yml
```

For venv environment setup (on VM) (see [Google Cloud docs](https://cloud.google.com/python/docs/setup#linux)), or anywhere else Python and/or venv might not yet be available:
```
sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```
Then, create venv environment, activate, and install all requirements:
```
cd <project>
python3 -m venv nlp-project

source nlp-project/bin/activate
pip install -r requirements.txt
```

## Finetuning and Evaluating TinyLlama

Scripts for training and evaluating TinyLlama can be found under ```scripts/training```. To set up arguments, create a ```config.json``` file with all training/evaluation parameters. Here is an example of a valid config to train using S-Loss with the confidence metric:
```
{
    "repo_path": "/path/to/cse517-final-project",
    "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "hardness_calc": "confidence",
    "loss_calc": "triangle",
    "data_order": null,
    "architecture_args": {
        "unfrozen_layers": [
            "score"
        ],
        "quantized": null
    },
    "dataset": "snli",
    "dataset_coordinates": "data/snli_data_map_coordinates.pickle",
    "dataset_ngram_path": "data/interpolated_ngram_perplexity.jsonl",
    "num_train_samples": 50000,
    "num_val_samples": 5000,
    "num_test_samples": 5000,
    "trainer_args": {
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.01,
        "learning_rate": 4e-4,
        "lr_scheduler_type": "constant",
        "num_warmup_steps": 0,
        "num_train_epochs": 2,
        "batch_size": 8,
        "dataloader_num_workers": 8,
        "load_best_model_at_end": true,
        "strategy": "epoch",
        "output_dir": "scripts/experiment_data/",
        "eval_every": 200
    },
    "do_train": true
}
```

Then pass the path to this configuration file as the first argument to the ```train_parallel.py```:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
srun accelerate launch --multi_gpu --num_processes=4 --num_machines 1 --mixed_precision 'no' --dynamo_backend 'no' ./scripts/training/train_parallel.py --config_path="./scripts/training/config.json"

```
Setting ```do_train=False``` will skip training and will only evaluate metrics on the validation and test sets (sampled with ```num_val_samples``` and ```num_test_samples``` from the validation and test samples of the provided dataset name (dataset is loaded from HF)).
