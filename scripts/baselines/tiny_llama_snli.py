import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np 
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import os

def setup_train_args(training_args_dict):
    """
    Args:
        training_args_dict: modifiable training arguments used to create HF TrainingArguments
    
    Sets up a HF TrainingArguments object for model finetuning. 
    """
    training_args = TrainingArguments(
        # set hyperparams (default from Tiny LLaMA)
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        learning_rate=4e-4,
        # warmup_steps=2000,
        lr_scheduler_type='cosine',
        # max_grad_norm=0.5,
        # changing hyperparams
        output_dir=training_args_dict["output_dir"],
        num_train_epochs=training_args_dict["epochs"],
        per_device_train_batch_size=training_args_dict["batch_size"],
        per_device_eval_batch_size=training_args_dict["batch_size"],
        dataloader_num_workers=8,

        load_best_model_at_end=True,

        # evaluates every 'eval_steps', which defaults to `logging_steps` 
        # evaluation_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        # gradient_accumulation_steps=4,
    )
    
    return training_args

def create_model(model_name, quantized):
    """
    Args:
        model_name: model ID to get HF model
        quantized: Boolean flag to indicate if QLoRA should be used
    
    Returns HF model and tokenizer for given model ID.
    """
    # set up model + tokenizer
    if quantized==True:
        bnb_config = BitsAndBytesConfig(
            # load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16
            load_in_8bit=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_name, quantization_config=bnb_config, num_labels=3, device_map="auto")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantized==True:
        config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="SEQ_CLS"
        )
        model = get_peft_model(model, config)

    return model, tokenizer

def load_data(dataset_name, tokenizer, num_train=500000):
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
    small_train_dataset = tokenized_ds["train"].filter(lambda sample: sample["labels"] != -1)
    small_eval_dataset = tokenized_ds["validation"].filter(lambda sample: sample["labels"] != -1)
    small_test_dataset = tokenized_ds["test"].filter(lambda sample: sample["labels"] != -1)

    # get subset for training
    small_train_dataset = small_train_dataset.shuffle(seed=42).select(range(num_train))
    # subset for val/test
    small_eval_dataset = small_eval_dataset.shuffle(seed=42).select(range(100))
    small_test_dataset = small_test_dataset.shuffle(seed=42).select(range(100))
    
    return small_train_dataset, small_eval_dataset, small_test_dataset

    # train_dataset = tokenized_ds["train"].filter(lambda sample: sample["labels"] != -1)
    # eval_dataset = tokenized_ds["validation"].filter(lambda sample: sample["labels"] != -1)
    # test_dataset = tokenized_ds["test"].filter(lambda sample: sample["labels"] != -1)
    
    # return train_dataset, eval_dataset, test_dataset

    # return small_train_dataset, eval_dataset, test_dataset

def tokenize_function_with_tokenizer(dataset_name, examples, tokenizer):
    """
    Function for creating datasets.
    """
    if dataset_name == "snli":
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
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

def main():
    print(f"cuda device count: {torch.cuda.device_count()}")
    # set up model, tokenizer, data
    # TODO: move these to args of script
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    dataset_name = "snli"
    quantized=False
    frozen=True
    batch_size=8
    num_train = 1000

    model, tokenizer = create_model(model_name, quantized)
    train_dataset, val_dataset, test_dataset = load_data(dataset_name, tokenizer, num_train)

    if frozen:
        # freeze weights except for last score
        for name, param in model.named_parameters():
            if name.startswith("model"):
                param.requires_grad = False

        # try also freezing last layer
        # for name, param in model.named_parameters():
        #     if name.startswith("model.layers.21") or name.startswith("model.norm"):
        #         param.requires_grad = True

    # set up trainer arguments
    training_args_dict = {}
    epochs = 10
    training_args_dict["batch_size"] = batch_size
    training_args_dict['epochs'] = int(epochs)
    training_args_dict['output_dir'] = f'experiment_data/baselines/finetune-{dataset_name}-{epochs}-{batch_size}-{num_train}'

    train_args = setup_train_args(training_args_dict)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # test baseline evaluation (no finetuning)
    # metrics = trainer.evaluate(eval_dataset=val_dataset)
    # test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    # print("Tiny LLaMA Baseline Evaluations (No fine tuning)")
    # print(f"validation accuracy: {metrics}")
    # print(f"Test accuracy: {test_metrics}")

    # finetune model
    trainer.train()
    trainer.save_model()

    # finetune metrics
    train_metrics = trainer.evaluate(eval_dataset=train_dataset)
    print(f"train metrics: {train_metrics}")
    val_metrics = trainer.evaluate()
    print(f"val metrics: {val_metrics}")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f"test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
