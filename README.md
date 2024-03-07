# cse517-final-project

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

Scripts for training and evaluating TinyLlama can be found under ```scripts/training```. To set up arguments, create a ```config.json``` file with all training/evaluation parameters:
```
{
    "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "architecture_args": {
        "unfrozen_layers": ["score"],
        "quantized": null
    },
    "dataset": "snli",
    "num_train_samples": 10000,
    "num_val_samples": 1000,
    "num_test_samples": 1000,
    "trainer_args": {
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.01,
        "learning_rate": 4e-4,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 10,
        "batch_size": 8,
        "dataloader_num_workers": 8,
        "load_best_model_at_end": true,
        "strategy": "epoch",
        "output_dir": "experiment_data/"
    },
    "do_train": true
}
```

Then pass the path to this configuration file as the first argument to the ```train.py```:
```
torchrun train.py --config_path="./config.json"
```
Setting ```do_train=False``` will skip training and will only evaluate metrics on the validation and test sets (sampled with ```num_val_samples``` and ```num_test_samples``` from the validation and test samples of the provided dataset name (dataset is loaded from HF)). ```trainer_args``` generally correspond to 



## Getting started
A note a /data:
snli_data_map_coordinates exclude entries where label = -1
snli_ngram_results did not exclude entries where label = -1
Thus difference in length.
