import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, get_scheduler, DataCollatorWithPadding
from datasets import load_dataset, Dataset
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import os
import pickle
import json
import argparse
from tqdm.auto import tqdm

class SNLICartographyDataset(Dataset):
    hardnesses = None
    def __init__(self, coordinates_path, dataset, limit, tokenizer, is_eval):
        self.dataset = dataset
        if SNLICartographyDataset.hardnesses is None:
            with open(coordinates_path, 'rb') as stream:
                SNLICartographyDataset.hardnesses = pickle.load(stream)
        # Add hardnesses to train dataset
        if not is_eval:
            new_cols = {"correctness": [], "confidence": [], "variability": []}
            for item in dataset:
                coords = SNLICartographyDataset.hardnesses[(item["premise"], item["hypothesis"])]
                new_cols["correctness"].append(coords["correctness"])
                new_cols["confidence"].append(coords["confidence"])
                new_cols["variability"].append(coords["variability"])
            self.dataset = self.dataset.add_column("correctness", new_cols["correctness"])
            self.dataset = self.dataset.add_column("confidence", new_cols["confidence"])
            self.dataset = self.dataset.add_column("variability", new_cols["variability"])
        # Grab "limit" samples
        if limit > 0:
            self.dataset = self.dataset.shuffle(seed=42).select(range(limit))
        else:
            self.dataset = self.dataset.shuffle(seed=42)
        # Tokenize
        def tokenize(examples):
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        self.dataset = self.dataset.map(tokenize, batched=True)
        self.dataset = self.dataset.remove_columns(["premise", "hypothesis"])
        self.dataset.set_format("torch")
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

        
class Model():
    def __init__(self, configs, name):
        self.configs = configs
        self.name = name
    
    def create_model_tokenizer(self):
        bnb_config = None
        if self.configs["architecture_args"]["quantized"] is not None:
            bnb_config = BitsAndBytesConfig(
                # load_in_4bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.float16
                load_in_8bit=True,
            )
        
        if self.configs["dataset"] == "snli":
            self.model = AutoModelForSequenceClassification.from_pretrained(self.configs["model_name"], quantization_config=bnb_config, num_labels=3)
        elif self.configs["dataset"] == "gsm8k":
            self.model = AutoModelForCausalLM(self.configs["model_name"], quantization_config=bnb_config, device_map="auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs["model_name"])
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.configs["architecture_args"]["quantized"]:
            config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.05, 
                bias="none", 
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, config)

        # freeze layers
        if self.configs["architecture_args"]["unfrozen_layers"] is not None:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            
            # unfreeze certain layers
            for layer in self.configs["architecture_args"]["unfrozen_layers"]:
                for name, param in self.model.named_parameters():
                    if name.startswith(layer):
                        param.requires_grad = True


    def initialize_data(self):
        if self.configs["dataset"] == "snli":
            train_snli = load_dataset(self.configs["dataset"], split="train")
            train_snli = train_snli.filter(lambda sample: sample["label"] != -1)
            val_snli = load_dataset(self.configs["dataset"], split="validation")
            val_snli = val_snli.filter(lambda sample: sample["label"] != -1)
            test_snli = load_dataset(self.configs["dataset"], split="test")
            test_snli = test_snli.filter(lambda sample: sample["label"] != -1)
            coordinates_path = os.path.join(self.configs["repo_path"], self.configs["dataset_coordinates"])
            self.train_dataset = SNLICartographyDataset(coordinates_path, train_snli, self.configs["num_train_samples"], self.tokenizer, False)
            self.val_dataset = SNLICartographyDataset(coordinates_path, val_snli, self.configs["num_val_samples"], self.tokenizer, True)
            self.test_dataset = SNLICartographyDataset(coordinates_path, test_snli, self.configs["num_test_samples"], self.tokenizer, True)
        else:
            raise ValueError(f"Dataset {self.configs['dataset']} unsupported.")
    
    def init_train_eval(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.accelerator = Accelerator()
        # Create Optimizer
        if self.configs["trainer_args"]["optim"] == "adamw_torch":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.configs["trainer_args"]["learning_rate"],
                betas=(self.configs["trainer_args"]["adam_beta1"], self.configs["trainer_args"]["adam_beta2"]),
                weight_decay=self.configs["trainer_args"]["weight_decay"]
            )
        else:
            raise ValueError(f"Dataset {self.configs['optim']} unsupported.")
        # TODO: Change these to use custom dataloaders
        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.configs["trainer_args"]["batch_size"], collate_fn=self.data_collator
        )
        eval_dataloader = DataLoader(
            self.val_dataset, batch_size=self.configs["trainer_args"]["batch_size"], collate_fn=self.data_collator
        )

        self.train_dl, self.eval_dl, self.model_parallel, self.optimizer_parallel = self.accelerator.prepare(
            train_dataloader, eval_dataloader, self.model, self.optimizer
        )
    
    def _compute_metrics(self, predictions, labels):
        metric = evaluate.load("accuracy")
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)
        return metric.compute(predictions=predictions, references=labels)["accuracy"]

    
    def train(self):
        if self.configs["do_train"]:
            num_epochs = self.configs["trainer_args"]["num_train_epochs"]
            num_training_steps = num_epochs * len(self.train_dl)
            lr_scheduler = get_scheduler(
                self.configs["trainer_args"]["lr_scheduler_type"],
                optimizer = self.optimizer_parallel,
                num_warmup_steps = self.configs["trainer_args"]["num_warmup_steps"],
                num_training_steps = num_training_steps
            )
            # Train and Evaluate
            progress_bar = tqdm(range(num_training_steps))
            i = 0
            best_acc = 0
            for epoch in range(num_epochs):
                for batch in self.train_dl:
                    self.model_parallel.train()
                    outputs = self.model_parallel(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    self.optimizer_parallel.step()
                    lr_scheduler.step()
                    self.optimizer_parallel.zero_grad()
                    progress_bar.set_description(f"Training Loss: {loss.item()}")
                    progress_bar.update(1)
                    i += 1
                    # Evaluate on validation set
                    if i % self.configs["trainer_args"]["eval_every"] == 0:
                        best_acc = self.eval(best_acc)

    def eval(self, best_acc):
        predictions = []
        labels = []
        eval_progress_bar = tqdm(range(len(self.eval_dl)), leave=False)
        self.model_parallel.eval()
        for eval_batch in self.eval_dl:
            with torch.no_grad():
                outputs = self.model_parallel(input_ids=eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], labels=eval_batch["labels"])
                eval_progress_bar.set_description(f"Eval Loss: {outputs.loss.item()}")
                eval_progress_bar.update(1)
                predictions.append(self.accelerator.gather(outputs.logits).cpu().numpy())
                labels.append(self.accelerator.gather(eval_batch["labels"]).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        predictions = predictions[:len(self.eval_dl.dataset)]
        labels = labels[:len(self.eval_dl.dataset)]
        # Compute metrics
        acc = self._compute_metrics(predictions, labels)
        if self.accelerator.is_main_process:
            print(f"Evaluation Accuracy: {acc}")
        # Save model based on validation acc
        if acc > best_acc:
            unwrapped_model = self.accelerator.unwrap_model(self.model_parallel)
            unwrapped_model.save_pretrained(
                os.path.join(self.configs["repo_path"], self.configs["trainer_args"]["output_dir"], f"{self.name}/model_{acc}"),
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )
            best_acc = acc
        return best_acc


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Script to finetune and evaluate TinyLlama model."
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to model, training, evaluation config json file"
    )
    args = parser.parse_args()
    config_file = open(args.config_path)
    configs = json.load(config_file)

    # Initialize model, dataset, and tokenizer using configs
    model = Model(configs, "snli_pretrained_all")
    model.create_model_tokenizer()
    model.initialize_data()
    print(f"Loaded {len(model.train_dataset)} train, {len(model.val_dataset)} val, and {len(model.test_dataset)} test data points")


    # Train model
    model.init_train_eval()
    model.train()

if __name__ == "__main__":
    main()


