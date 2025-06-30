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

class DrugReviewDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Drug Review Dataset with Sentiment Features
    
    Supports both local development and Google Colab environments.
    Automatically detects environment and uses appropriate data paths.
    
    Features:
    - Regression setup (continuous rating values 1-10)
    - Sentiment analysis features (VADER scores)
    - Proper tensor handling to avoid PyTorch warnings
    - Cross-platform compatibility (local + Colab)
    """
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
        self.num_labels = 10  # Changed to 10 for classification (ratings 1-10 as classes 0-9)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def load_dataset_locally(self):
        """
        Load dataset from local files, automatically detecting Colab vs local environment
        """
        # Check if running on Google Colab
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
        
        if is_colab:
            # Google Colab paths
            base_dir = "/content/Data"
            print("🔍 Detected Google Colab environment")
        else:
            # Local development paths
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
            print("🔍 Detected local development environment")
        
        print(f"📁 Looking for data files in: {base_dir}")
        
        data_files = {
            "train": os.path.join(base_dir, "drug_review_train_with_sentiment.csv"),
            "validation": os.path.join(base_dir, "drug_review_validation_with_sentiment.csv"),
            "test": os.path.join(base_dir, "drug_review_test_with_sentiment.csv")
        }
        
        # Check if files exist
        missing_files = []
        for split, file_path in data_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{split}: {file_path}")
        
        if missing_files:
            print("❌ Missing data files:")
            for missing in missing_files:
                print(f"   - {missing}")
            
            if is_colab:
                print("\n💡 For Google Colab:")
                print("   1. Upload your data files to /content/Data/ directory")
                print("   2. Or mount Google Drive and update paths accordingly")
                print("   3. Make sure files are named correctly with '_with_sentiment.csv' suffix")
            else:
                print(f"\n💡 For local development:")
                print(f"   1. Make sure data files exist in: {base_dir}")
                print("   2. Check that file names match expected pattern")
            
            raise FileNotFoundError(f"Required data files not found in {base_dir}")
        
        print("✅ All data files found!")
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

        # For classification: convert ratings (1-10) to class indices (0-9)
        features["labels"] = [int(label) - 1 for label in example_batch["label"]]  # Convert 1-10 to 0-9
        
        # Add sentiment features - create tensors properly to avoid warnings
        # Use torch.tensor for numerical data (recommended for lists/arrays)
        sentiment_neg = torch.tensor(example_batch["sentiment_neg"], dtype=torch.float32)
        sentiment_neu = torch.tensor(example_batch["sentiment_neu"], dtype=torch.float32)
        sentiment_pos = torch.tensor(example_batch["sentiment_pos"], dtype=torch.float32)
        sentiment_compound = torch.tensor(example_batch["sentiment_compound"], dtype=torch.float32)
        
        # Stack them into a single tensor for easier processing
        sentiment_features = torch.stack([
            sentiment_neg,
            sentiment_neu, 
            sentiment_pos,
            sentiment_compound
        ], dim=1)  # Shape: (batch_size, 4)
        
        features["sentiment_features"] = sentiment_features
        return features

    @staticmethod
    def setup_colab_data():
        """
        Helper method to setup data in Google Colab environment.
        Call this before creating the data module in Colab.
        """
        try:
            import google.colab  # noqa: F401
            print("🔧 Setting up data for Google Colab...")
            
            # Create data directory
            import os
            os.makedirs("/content/Data", exist_ok=True)
            
            print("📁 Created /content/Data directory")
            print("\n📋 Next steps:")
            print("1. Upload your data files to /content/Data/ using:")
            print("   - File browser (left sidebar)")
            print("   - Or drag & drop files")
            print("   - Or use: from google.colab import files; files.upload()")
            print("\n📝 Required files:")
            print("   - drug_review_train_with_sentiment.csv")
            print("   - drug_review_validation_with_sentiment.csv") 
            print("   - drug_review_test_with_sentiment.csv")
            print("\n💡 Alternative: Mount Google Drive and update paths")
            print("   from google.colab import drive")
            print("   drive.mount('/content/drive')")
            
        except ImportError:
            print("ℹ️  Not running in Google Colab - no setup needed")

# Backward compatibility alias
sst_datamodule = DrugReviewDataModule

# Convenience function for quick setup
def create_drug_review_datamodule(model_name='bert-base-uncased', **kwargs):
    """
    Convenience function to create a DrugReviewDataModule with common defaults
    
    Args:
        model_name: HuggingFace model name for tokenizer
        **kwargs: Additional arguments passed to DrugReviewDataModule
    
    Returns:
        DrugReviewDataModule: Configured data module
    """
    return DrugReviewDataModule(model_name_or_path=model_name, **kwargs)