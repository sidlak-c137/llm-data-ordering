import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, get_scheduler, DataCollatorWithPadding, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Dataset
import pandas as pd
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
    def __init__(self, coordinates_path, dataset, limit, tokenizer, is_eval, hardness_calc):
        self.dataset = dataset
        if SNLICartographyDataset.hardnesses is None:
            with open(coordinates_path, 'rb') as stream:
                SNLICartographyDataset.hardnesses = pickle.load(stream)
        # Add hardnesses to train dataset
        if not is_eval:
            hardness = []
            if hardness_calc == "baseline":
                for item in dataset:
                    coords = SNLICartographyDataset.hardnesses[(item["premise"], item["hypothesis"])]
                    hardness.append(coords["correctness"]) # Some arbitrary column for baseline
            elif hardness_calc == "variability":
                for item in dataset:
                    coords = SNLICartographyDataset.hardnesses[(item["premise"], item["hypothesis"])]
                    hardness.append(coords["variability"])
                # min_hardness = min(hardness)
                # max_hardenss = max(hardness)
                # hardness = [(val - min_hardness) / (max_hardenss - min_hardness) for val in hardness]
                # hardness = torch.div(torch.argsort(torch.tensor(hardness)), len(hardness)).tolist()
            else:
                raise ValueError(f"Hardness Calc {hardness_calc} unsupported.")
            self.dataset = self.dataset.add_column("hardness", hardness)
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
        self.best_model_path = ""
        self.train_metrics = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
        self.accelerator = Accelerator()
    
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
            self.model_wrapper = AutoModelForSequenceClassificationWithLoss(self.configs)
            self.model = self.model_wrapper.model
        elif self.configs["dataset"] == "gsm8k":
            self.model = AutoModelForCausalLM(self.configs["model_name"], quantization_config=bnb_config, device_map="auto")
        else:
            raise ValueError(f"Dataset {self.configs['dataset']} unsupported.")
        
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
            self.train_dataset = SNLICartographyDataset(coordinates_path, train_snli, self.configs["num_train_samples"], self.tokenizer, False, self.configs["hardness_calc"])
            self.val_dataset = SNLICartographyDataset(coordinates_path, val_snli, self.configs["num_val_samples"], self.tokenizer, True, self.configs["hardness_calc"])
            self.test_dataset = SNLICartographyDataset(coordinates_path, test_snli, self.configs["num_test_samples"], self.tokenizer, True, self.configs["hardness_calc"])
            self.accelerator.print(f"Loaded {len(self.train_dataset)} train, {len(self.val_dataset)} val, and {len(self.test_dataset)} test data points")
        else:
            raise ValueError(f"Dataset {self.configs['dataset']} unsupported.")
    
    def init_train_eval(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
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
            self.train_dataset, shuffle=True, batch_size=self.configs["trainer_args"]["batch_size"], collate_fn=self.data_collator, drop_last=True
        )
        eval_dataloader = DataLoader(
            self.val_dataset, batch_size=self.configs["trainer_args"]["batch_size"], collate_fn=self.data_collator, drop_last=True
        )

        self.train_dl, self.eval_dl, self.model_parallel, self.optimizer_parallel = self.accelerator.prepare(
            train_dataloader, eval_dataloader, self.model_wrapper, self.optimizer
        )
    
    def _compute_metrics(self, predictions, labels):
        metric = evaluate.load("accuracy")
        predictions = torch.argmax(predictions, dim=-1)
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
            progress_bar = tqdm(range(num_training_steps), disable=not self.accelerator.is_local_main_process)
            i = 0
            best_acc = 0
            losses = []
            for epoch in range(num_epochs):
                for batch in self.train_dl:
                    self.model_parallel.train()
                    outputs = self.model_parallel(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], hardnesses=batch["hardness"], steps=i/num_training_steps)
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    self.optimizer_parallel.step()
                    lr_scheduler.step()
                    self.optimizer_parallel.zero_grad()
                    losses.append(self.accelerator.gather_for_metrics(loss).cpu().detach())
                    progress_bar.set_description(f"Training Loss: {loss.item()}")
                    progress_bar.update(1)
                    i += 1
                    # Evaluate on validation set
                    if i % self.configs["trainer_args"]["eval_every"] == 0:
                        self.accelerator.wait_for_everyone()
                        losses = torch.concatenate(losses)
                        losses = losses[:self.configs["trainer_args"]["eval_every"] * self.accelerator.state.num_processes * self.configs["trainer_args"]["batch_size"]]
                        self.train_metrics["train_loss"].append(losses.mean().item())
                        losses = []
                        best_acc = self._eval(best_acc, i)


    def _eval(self, best_acc, step):
        predictions = []
        labels = []
        losses = []
        eval_progress_bar = tqdm(range(len(self.eval_dl)), leave=False, disable=not self.accelerator.is_local_main_process)
        self.model_parallel.eval()
        for eval_batch in self.eval_dl:
            with torch.no_grad():
                outputs = self.model_parallel(input_ids=eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], labels=eval_batch["labels"])
                loss = outputs.loss
                eval_progress_bar.set_description(f"Eval Loss: {loss.item()}")
                eval_progress_bar.update(1)
                predictions.append(self.accelerator.gather_for_metrics(outputs.logits).cpu())
                labels.append(self.accelerator.gather_for_metrics(eval_batch["labels"]).cpu())
                losses.append(self.accelerator.gather_for_metrics(loss).cpu())
        
        predictions = torch.concatenate(predictions)
        labels = torch.concatenate(labels)
        losses = torch.concatenate(losses)
        predictions = predictions[:len(self.eval_dl) * self.configs["trainer_args"]["batch_size"]]
        labels = labels[:len(self.eval_dl) * self.configs["trainer_args"]["batch_size"]]
        losses = losses[:len(self.eval_dl) * self.configs["trainer_args"]["batch_size"]]
        # Compute metrics
        acc = self._compute_metrics(predictions, labels)
        loss = losses.mean().item()
        self.accelerator.print(f"Evaluation Accuracy: {acc}, Evaluation Loss: {loss}")
        if self.accelerator.is_main_process:
            self.train_metrics["step"].append(step * self.accelerator.state.num_processes)
            self.train_metrics["val_acc"].append(acc)
            self.train_metrics["val_loss"].append(loss)

        # Save model based on validation acc
        if acc > best_acc:
            self.best_model_path = os.path.join(self.configs["repo_path"], self.configs["trainer_args"]["output_dir"], f"{self.name}/model_{acc}_{step}")
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model_parallel).model
            unwrapped_model.save_pretrained(
                self.best_model_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )
            best_acc = acc

        if self.accelerator.is_main_process:
            df = pd.DataFrame.from_dict(self.train_metrics)
            df.to_csv(os.path.join(self.configs["repo_path"], self.configs["trainer_args"]["output_dir"], f"{self.name}_metrics.csv"), index=False)

        return best_acc
    
    def test(self):
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
            self.model = AutoModelForSequenceClassification.from_pretrained(self.best_model_path, quantization_config=bnb_config, num_labels=3)
        elif self.configs["dataset"] == "gsm8k":
            self.model = AutoModelForCausalLM(self.best_model_path, quantization_config=bnb_config, device_map="auto")

        test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.configs["trainer_args"]["batch_size"], collate_fn=self.data_collator, drop_last=True
        )

        self.test_dl, self.model_parallel = self.accelerator.prepare(
            test_dataloader, self.model
        )

        predictions = []
        labels = []
        losses = []
        test_progress_bar = tqdm(range(len(self.test_dl)), leave=False, disable=not self.accelerator.is_local_main_process)
        self.model_parallel.eval()
        for test_batch in self.test_dl:
            with torch.no_grad():
                outputs = self.model_parallel(input_ids=test_batch["input_ids"], attention_mask=test_batch["attention_mask"], labels=test_batch["labels"])
                loss = outputs.loss
                test_progress_bar.set_description(f"Test Loss: {loss.item()}")
                test_progress_bar.update(1)
                predictions.append(self.accelerator.gather_for_metrics(outputs.logits).cpu())
                labels.append(self.accelerator.gather_for_metrics(test_batch["labels"]).cpu())
                losses.append(self.accelerator.gather_for_metrics(loss).cpu())
        
        predictions = torch.concatenate(predictions)
        labels = torch.concatenate(labels)
        losses = torch.concatenate(losses)
        predictions = predictions[:len(self.test_dl) * self.configs["trainer_args"]["batch_size"]]
        labels = labels[:len(self.test_dl) * self.configs["trainer_args"]["batch_size"]]
        losses = losses[:len(self.test_dl) * self.configs["trainer_args"]["batch_size"]]
        # Compute metrics
        acc = self._compute_metrics(predictions, labels)
        self.accelerator.print(f"Test Accuracy: {acc}, Test Loss: {losses.mean().item()}")

class AutoModelForSequenceClassificationWithLoss(torch.nn.Module):
    def __init__(self, configs):
        super(AutoModelForSequenceClassificationWithLoss, self).__init__()
        self.configs = configs
        bnb_config = None
        if configs["architecture_args"]["quantized"] is not None:
            bnb_config = BitsAndBytesConfig(
                # load_in_4bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.float16
                load_in_8bit=True,
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(configs["model_name"], quantization_config=bnb_config, num_labels=3)

    def forward(self, input_ids, attention_mask, labels, hardnesses=None, steps=1):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if hardnesses is None:
            return outputs
        generated_labels = F.log_softmax(outputs.logits, dim=-1)
        return SequenceClassifierOutput(loss=self._compute_loss(generated_labels, labels, hardnesses, steps), logits = outputs.logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        
    def _compute_loss(self, generated_labels, predicted_labels, hardnesses, steps):
        loss_values = -generated_labels[range(predicted_labels.shape[0]), predicted_labels]
        if self.configs["loss_calc"] == "baseline":
            hardnesses = torch.ones_like(hardnesses)
        elif self.configs["loss_calc"] == "triangle":
            hardnesses = ((2 * hardnesses - 1) * (steps - 1)) + 1
        elif self.configs["loss_calc"] == "crazy":
            hardnesses = (1.5 - 5 * (0.5 - hardnesses)) / (steps + 0.2) + 1
            hardnesses[hardnesses <= 0.0] = 0.0
        else:
            raise ValueError(f"Loss Calc {self.configs['loss_calc']} unsupported.")
        
        loss = torch.multiply(loss_values, hardnesses).mean()
        # assert torch.isclose(loss, F.cross_entropy(generated_labels, predicted_labels)).item()
        return loss

def main():
    torch.manual_seed(42)
    set_seed(42)
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
    model = Model(configs, "snli_triangle_variability")
    model.create_model_tokenizer()
    model.initialize_data()

    # Train model
    model.init_train_eval()
    model.train()

    # Test model
    model.test()

if __name__ == "__main__":
    main()


