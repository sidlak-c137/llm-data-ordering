import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import os
import json
import argparse


def setup_train_args(trainer_args):
    """
    Args:
        trainer_args: modifiable training arguments used to create HF TrainingArguments

    Sets up a HF TrainingArguments object for model finetuning.
    """
    train_args = TrainingArguments(
        # params from config
        optim=trainer_args["optim"],
        adam_beta1=trainer_args["adam_beta1"],
        adam_beta2=trainer_args["adam_beta2"],
        weight_decay=trainer_args["weight_decay"],
        learning_rate=trainer_args["learning_rate"],
        output_dir=trainer_args["output_dir"],
        num_train_epochs=trainer_args["num_train_epochs"],
        per_device_train_batch_size=trainer_args["batch_size"],
        per_device_eval_batch_size=trainer_args["batch_size"],
        dataloader_num_workers=trainer_args["dataloader_num_workers"],
        load_best_model_at_end=trainer_args["load_best_model_at_end"],
        evaluation_strategy=trainer_args["strategy"],
        save_strategy=trainer_args["strategy"],
        logging_strategy=trainer_args["strategy"],
        # set params
        report_to="none",
        lr_scheduler_type="cosine",
        # gradient_accumulation_steps=4,
    )
    return train_args


def create_model(model_name, dataset, quantized=None, unfrozen_layers=None):
    """
    Args:
        model_name: model ID for model + tokenizer
        dataset: dataset to finetune/evaluate model on
        unfrozen_layers: list of layers that should not be frozen

    Returns HF model and tokenizer for given model ID.
    """
    # set up model + tokenizer
    bnb_config = None
    if quantized is not None:
        bnb_config = BitsAndBytesConfig(
            # load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16
            load_in_8bit=True,
        )

    if dataset == "snli":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, quantization_config=bnb_config, num_labels=3, device_map="auto"
        )
    elif dataset == "gsm8k":
        model = AutoModelForCausalLM(
            model_name, quantization_config=bnb_config, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantized:
        config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS"
        )
        model = get_peft_model(model, config)

    # freeze layers
    if unfrozen_layers is not None:
        model = freeze_layers(model, unfrozen_layers)

    return model, tokenizer


def freeze_layers(model, unfrozen_layers):
    """
    Args:
        model: model for which layers should be frozen
        unfrozen_layers: list of layers that should be frozen; layer names should start with these strings

    Returns model with frozen layers.
    """
    # freeze all layers
    for name, param in model.named_parameters():
        param.requires_grad = False

    # unfreeze certain layers
    for layer in unfrozen_layers:
        for name, param in model.named_parameters():
            if name.startswith(layer):
                param.requires_grad = False

    return model


def load_data(dataset_name, tokenizer, num_train, num_val, num_test):
    """
    Args:
        dataset_name: which HF dataset to use for finetuning/evaluation.
        tokenizer: tokenizer to tokenize dataset into HF datasets

    Returns HF datasets of train, validation, and test sets.
    """

    def tokenize_function(example):
        return tokenize_function_with_tokenizer(dataset_name, example, tokenizer)

    tokenized_ds = load_dataset(dataset_name)
    tokenized_ds = tokenized_ds.map(tokenize_function, batched=True)

    if dataset_name == "snli":
        tokenized_ds = tokenized_ds.remove_columns(["premise", "hypothesis"])
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
    else:
        raise ValueError(f"Dataset {dataset_name} unsupported.")

    tokenized_ds.set_format("torch")

    # filter -1 samples
    train_dataset = tokenized_ds["train"].filter(lambda sample: sample["labels"] != -1)
    eval_dataset = tokenized_ds["validation"].filter(
        lambda sample: sample["labels"] != -1
    )
    test_dataset = tokenized_ds["test"].filter(lambda sample: sample["labels"] != -1)
    # sample subsets
    train_dataset = train_dataset.shuffle(seed=42).select(range(num_train))
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(num_val))
    test_dataset = test_dataset.shuffle(seed=42).select(range(num_test))

    return train_dataset, eval_dataset, test_dataset


def tokenize_function_with_tokenizer(dataset_name, examples, tokenizer):
    """
    Function for creating datasets.
    """
    if dataset_name == "snli":
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
    else:
        raise ValueError(f"Dataset {dataset_name} unsupported.")


def compute_metrics(example):
    """
    Overridden method
    """
    metric = evaluate.load("accuracy")
    logits, labels = example
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to finetune and evaluate TinyLlama model."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to model, training, evaluation config json file",
    )
    return parser.parse_args()


def main():
    # set up model, tokenizer, data
    args = parse_args()
    config_file = open(args.config_path)
    configs = json.load(config_file)
    model_name = configs["model_name"]
    dataset_name = configs["dataset"]
    quantized = configs["architecture_args"]["quantized"]
    unfrozen_layers = configs["architecture_args"]["unfrozen_layers"]
    trainer_args = configs["trainer_args"]
    num_train = configs["num_train_samples"]
    num_val = configs["num_val_samples"]
    num_test = configs["num_test_samples"]
    do_train = configs["do_train"]

    model, tokenizer = create_model(
        model_name, dataset_name, quantized, unfrozen_layers
    )
    train_dataset, val_dataset, test_dataset = load_data(
        dataset_name, tokenizer, num_train, num_val, num_test
    )
    # set output directory for saved output
    output_dir = os.path.join(
        train_args["repo_path"],
        trainer_args["output_dir"],
        f"{dataset_name}/{num_train}-{num_val}-{num_test}-{trainer_args['num_train_epochs']}",
    )
    trainer_args["output_dir"] = output_dir

    # set up HF trainer for finetuning + evaluation
    train_args = setup_train_args(trainer_args)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    if do_train:
        # finetune model
        trainer.train()
        trainer.save_model()

        # finetune metrics
        train_metrics = trainer.evaluate(eval_dataset=train_dataset)
        print(f"train metrics (finetuned): {train_metrics}")

    # test metrics
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(f"val metrics: {val_metrics}")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f"test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
