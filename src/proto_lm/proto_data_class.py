from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import pytorch_lightning as pl
import torch
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os

class sst_datamodule(pl.LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
        "sentiment_features",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_pos",
        "sentiment_compound"
    ]

    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            dataset=None,  # Accept pre-loaded dataset
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset = dataset

        self.text_fields = ['review_clean']
        self.num_labels = 10  # Number of classes in drug review dataset, a user can rate from 1 to 10
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def load_dataset_locally(self):     

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
        data_files = {
            "train": os.path.join(base_dir, "drug_review_train_with_sentiment.csv"),
            "validation": os.path.join(base_dir, "drug_review_validation_with_sentiment.csv"),
            "test": os.path.join(base_dir, "drug_review_test_with_sentiment.csv")
        }
        self.dataset = load_dataset("csv", data_files=data_files)

    def setup(self, stage: str = None):
        if self.dataset is None:
            print("Loading dataset from local files...")
            self.load_dataset_locally()

        for split in self.dataset.keys():
            print(f'split is: {split}')
            

            if "rating" in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].rename_column("rating", "label")


            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                # remove_columns=["label", "Unnamed: 0"],
            )
            
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            print(f'self.columns: {self.columns}')
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        print(f'returned self batch size: {self.train_batch_size}')
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)

    def convert_to_features(self, example_batch, indices=None):
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length
        )

        # features["labels"] = [1 if int(label) >= 7 else 0 for label in example_batch["label"]]
        # Convert labels to zero-indexed (1-10 becomes 0-9)
        features["labels"] = [int(label) - 1 for label in example_batch["label"]]
        # # Add sentiment features
        # features["sentiment_neg"] = torch.tensor(example_batch["sentiment_neg"]).unsqueeze(1)
        # features["sentiment_neu"] = torch.tensor(example_batch["sentiment_neu"]).unsqueeze(1)
        # features["sentiment_pos"] = torch.tensor(example_batch["sentiment_pos"]).unsqueeze(1)
        # features["sentiment_compound"] = torch.tensor(example_batch["sentiment_compound"]).unsqueeze(1)
        # Combine sentiment features into a single tensor
        sentiment_features = torch.stack([
            torch.tensor(example_batch["sentiment_neg"]),
            torch.tensor(example_batch["sentiment_neu"]),
            torch.tensor(example_batch["sentiment_pos"]),
            torch.tensor(example_batch["sentiment_compound"])
        ], dim=1)  # Shape: (batch_size, 4)
        features["sentiment_features"] = sentiment_features
        return features