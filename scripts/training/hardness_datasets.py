from datasets import Dataset
import pandas as pd
import pickle

"""
Enriched HuggingFace Datasets with calculated hardness values.
"""


class SNLICartographyDataset(Dataset):
    hardnesses = None
    @staticmethod
    def _create_hardnesses_baseline(dataset):
        hardness = []
        for item in dataset:
            coords = SNLICartographyDataset.hardnesses[
                (item["premise"], item["hypothesis"])
            ]
            hardness.append(coords["correctness"])  # Some arbitrary column for baseline
        return hardness

    @staticmethod
    def _create_hardnesses_variability(dataset):
        hardness = []
        for item in dataset:
            coords = SNLICartographyDataset.hardnesses[
                (item["premise"], item["hypothesis"])
            ]
            hardness.append(coords["variability"])
        return hardness

    @staticmethod
    def _create_hardnesses_variability_normalized(dataset):
        hardness = []
        for item in dataset:
            coords = SNLICartographyDataset.hardnesses[
                (item["premise"], item["hypothesis"])
            ]
            hardness.append(coords["variability"])
        min_hardness = min(hardness)
        max_hardenss = max(hardness)
        hardness = [
            (val - min_hardness) / (max_hardenss - min_hardness) for val in hardness
        ]
        return hardness

    @staticmethod
    def _create_hardnesses_confidence(dataset):
        hardness = []
        for item in dataset:
            coords = SNLICartographyDataset.hardnesses[
                (item["premise"], item["hypothesis"])
            ]
            hardness.append(1 - coords["confidence"])
        return hardness

    @staticmethod
    def _create_hardnesses_confidence_normalized(dataset):
        hardness = []
        for item in dataset:
            coords = SNLICartographyDataset.hardnesses[
                (item["premise"], item["hypothesis"])
            ]
            hardness.append(1 - coords["confidence"])
        min_hardness = min(hardness)
        max_hardenss = max(hardness)
        hardness = [
            (val - min_hardness) / (max_hardenss - min_hardness) for val in hardness
        ]
        return hardness

    create_hardnesses = {
        "baseline": _create_hardnesses_baseline.__get__(object),
        "variability": _create_hardnesses_variability.__get__(object),
        "even-scaled-variability": _create_hardnesses_variability_normalized.__get__(object),
        "confidence": _create_hardnesses_confidence.__get__(object),
        "even-scaled-confidence": _create_hardnesses_confidence_normalized.__get__(object),
    }

    def __init__(
        self, coordinates_path, dataset, limit, tokenizer, is_eval, hardness_calc
    ):
        self.dataset = dataset
        if SNLICartographyDataset.hardnesses is None:
            with open(coordinates_path, "rb") as stream:
                SNLICartographyDataset.hardnesses = pickle.load(stream)
        # Add hardnesses to train dataset
        if not is_eval:
            hardness = []
            if hardness_calc in self.create_hardnesses:
                hardness = SNLICartographyDataset.create_hardnesses[hardness_calc](self.dataset)
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
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

        self.dataset = self.dataset.map(tokenize, batched=True)
        self.dataset = self.dataset.remove_columns(["premise", "hypothesis"])
        self.dataset.set_format("torch")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def sort_by_hardness(self, order="increasing"):
        if order == "increasing":
            self.dataset = self.dataset.sort("hardness")
        elif order == "decreasing":
            self.dataset = self.dataset.sort("hardness", reverse=True)


class SNLINgramPerplexityDataset(Dataset):
    hardnesses = None

    def __init__(self, coordinates_path, dataset, limit, tokenizer, is_eval):
        self.dataset = dataset
        if SNLINgramPerplexityDataset.hardnesses is None:
            SNLINgramPerplexityDataset.hardnesses = {}
            # read the file into a map of (premise, hypothesis) -> perplexity
            df = pd.read_json(path_or_buf=coordinates_path, lines=True)
            for _, line in df.iterrows():
                key = (line["premise"], line["hypothesis"])
                SNLINgramPerplexityDataset.hardnesses[key] = line["perplexity"]

        # hardness is only neeeded for train set
        if not is_eval:
            hardness = self._create_hardnesses()
            self.dataset = self.dataset.add_column("hardness", hardness)
        if limit > 0:
            self.dataset = self.dataset.shuffle(seed=42).select(range(limit))
        else:
            self.dataset = self.dataset.shuffle(seed=42)

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

        self.dataset = self.dataset.map(tokenize, batched=True)
        self.dataset = self.dataset.remove_columns(["premise", "hypothesis"])
        self.dataset.set_format("torch")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def sort_by_hardness(self, order="increasing"):
        if order == "increasing":
            self.dataset = self.dataset.sort("hardness")
        elif order == "decreasing":
            self.dataset = self.dataset.sort("hardness", reverse=True)

    def _create_hardnesses(self):
        hardness = []
        for item in self.dataset:
            key = (item["premise"], item["hypothesis"])
            val = SNLINgramPerplexityDataset.hardnesses.get(key, None)
            assert val is not None, f"couldn't find key: {key}"
            hardness.append(SNLINgramPerplexityDataset.hardnesses.get(key))
        min_hardness = min(hardness)
        max_hardness = max(hardness)
        hardness = [
            (val - min_hardness) / (max_hardness - min_hardness) for val in hardness
        ]
        return hardness
