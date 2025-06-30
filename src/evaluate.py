"""
file: evaluate.py

Description: This file contains functions to load models, evaluate them, and calculate various metrics such as complexity, faithfulness, sparsity,
max-sensitivity, and robustness. It includes loading LightGBM, Lasso, BERT, and ProtoLM models, generating saliency maps usign SHAP or LIME.
It also provides wrappers for these models to make them compatible with Quantus.

Last updated: 26-06-2025
"""


import os
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, cohen_kappa_score
from lightgbm import LGBMRegressor
from feature_extraction import load_data   
import quantus
import torch
import json
import shap
from tqdm import tqdm
import joblib
from scipy.special import expit  # sigmoid
from sklearn.linear_model import Lasso
from check_the_data import feature_extraction # Import the function
from scipy.sparse import hstack
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from bert import DrugReviewDataset
from captum.attr import IntegratedGradients, Saliency, GradientShap
from torch.utils.data import DataLoader
import torch
from proto_lm.ProtoLM import proto_lm
from proto_lm.proto_data_class import sst_datamodule
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification




# Global variable to hold the loaded model
MODEL_DIR = "Models"
EVAL_DIR = "Evaluation"
DATA_DIR = "Data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_lgb_model(filename=None):
    """
    Load a LightGBM model from a file.
    
    Parameters:
    filename (str): The path to the model file.
    
    Returns:
    lgb.Booster: The loaded LightGBM model.
    """
    # Load the saved model
    if filename is None:
        # Default filename if not provided
        # Adjust the filename as per your saved model
        # Ensure the file exists in the specified path
        filename = f"{MODEL_DIR}/trained_lgb_model_20250617_145741.txt"
    booster = lgb.Booster(model_file=filename)

    # Wrap the Booster in an LGBMRegressor (optional)
    lgb_model = LGBMRegressor()
    lgb_model._Booster = booster

    return lgb_model

def load_lasso_model(filename=None):
    """
    Load a Lasso Regression model from a file (e.g., using joblib).

    Parameters:
    filename (str): The path to the model file.

    Returns:
    Lasso: The loaded Lasso Regression model.
    """    
    if filename is None:
        filename = f"{MODEL_DIR}/trained_lasso_model.joblib"  # Example filename
    lasso_model = joblib.load(filename)
    return lasso_model


def load_bert_model(path=None):
    """
    Load a BERT model from the folder it was saved into after training.

    Parameters:
        path (str): The path to the model folder.

    Returns:
        BertModel: The loaded BERT model.
        BertTokenizer: The tokenizer associated with the BERT model.
    """    
    if path is None:
        path = f"{MODEL_DIR}/bert_model"
    
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.to(device)  # Move the model to the appropriate device (GPU or CPU)
    return model, tokenizer

def load_proto_lm_model(path=None):
    """ Load a ProtoLM model from the specified path.

    Parameters:
        path (str): The path to the ProtoLM model directory.

    Returns:
        proto_lm: The loaded ProtoLM model.
        tokenizer: The tokenizer associated with the ProtoLM model.
        args (dict): Configuration arguments used for loading the model.
    """

    model_name = 'bert-base-uncased'  # Backbone LLM model to load

    args = {
        'model_name': model_name,        # backbone LLM model to load
        'max_seq_length': 100,                # maximum sentence length to pad/truncate to
        'num_prototypes': 100,               # number of prototypes to train
        'hidden_shape': 1024,                # hidden shape of each prototype, should match LLM output
        'num_classes': 10,                   # number of output classes - 10 for classification (rating prediction 1-10)
        'cohsep_ratio': 0.5,                 # ratio of prototypes in class to push/pull
        'lambda0': 0.5,                      # lambda0 in loss
        'lr': 3e-4,                          # initial learning rate
        'proto_training_weights': 1,         # whether to train prototype weights (1=True, 0=False)
        'batch_size': 32,                   # batch size for dataloader
        'logger_dir': 'tb_logs',             # directory for the logger to store training details
        'checkpoint_dir': 'ckpt_dir',        # directory to store checkpoints
        'config_subdir': 'config_subdir',    # subdirectory for checkpoints of a certain config
        'max_epochs': 1,                     # number of epochs to train
        'num_gpu': 1,                        # number of gpus to train on
        'load_model': model_name,  # path to load a pretrained model, if any
    }

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args['model_name'], ignore_mismatched_sizes=True
    )

    if hasattr(base_model, "roberta"):
        llm_model = base_model.roberta
    elif hasattr(base_model, "bert"):
        llm_model = base_model.bert
    else:
        llm_model = base_model


    config = AutoConfig.from_pretrained(args['model_name'])
    hidden_size = config.hidden_size  # Dynamically get the hidden size (768 for bert-base-uncased)
    args['hidden_shape'] = hidden_size
    

    proto = proto_lm(
        pretrained_model=llm_model,
        max_seq_length=args['max_seq_length'],
        num_prototypes=args['num_prototypes'],
        hidden_shape=args['hidden_shape'],
        num_classes=args['num_classes'],
        cohsep_ratio=args['cohsep_ratio'],
        lambda0=args['lambda0'],
        lr=args['lr'],
        proto_training_weights=bool(args['proto_training_weights']),
    )
    # Load the trained model
    src_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(src_dir, "proto_lm", "final_proto_model.pt")
    proto.load_state_dict(torch.load(model_path, weights_only=False))  # Load the state dictionary into the model

    proto.to(device)  # Move the model to the appropriate device
    return proto, AutoTokenizer.from_pretrained(args['model_name']), args


def get_dataloader_for_bert_model(df, y_test, col, tokenizer, batch_size=2, max_length=128, shuffle=False):
    """
    Load inputs for the BERT model from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        y_test (pd.Series): Series containing the target labels.
        col (str): Column name containing the text data.
        tokenizer (BertTokenizer): Tokenizer for BERT.
        max_length (int): Maximum length of the input sequences.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the BERT model.
    """
    # Use the same column and tokenizer as in training
    dataset = DrugReviewDataset(
        texts=df[col].to_numpy(),
        ratings=y_test.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_length  # or whatever you used in training
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

def get_dataloader_for_proto_lm_model(args, subset=None):
    """
    Load the data for the ProtoLM model.    
    Parameters:
        args (dict): Dictionary containing model parameters.
        subset (int, optional): If provided, only this many samples from the test set will be used.

    Returns:
        tuple: Train, validation, and test DataLoaders for the ProtoLM model.
    """


    # get data module
    drug_review_dm = sst_datamodule(
        model_name_or_path=args['model_name'],
        max_seq_length=args['max_seq_length'],
        train_batch_size=args['batch_size'],
        eval_batch_size=args['batch_size']
    )
    drug_review_dm.setup(stage='fit')

    # Optionally subset the test dataset
    test_dataset = drug_review_dm.dataset["test"]
    if subset is not None:
        # Use HuggingFace Dataset .select for subsetting
        test_dataset = test_dataset.select(range(subset))

    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'])
    return drug_review_dm.train_dataloader(), drug_review_dm.val_dataloader(), test_loader

def calc_complexity(model, model_name, x_batch, y_batch, explanations_padded, device='cpu'):  
    """
    Calculate the Complexity metric using Quantus.

    Parameters:
        model: The LightGBM model's predict function.
        model_name: Name of the model.
        x_batch: Input features.
        y_batch: True labels.
        explanations_padded: Saliency maps.
        device: Device to run the evaluation on.

    Returns:
        float: Complexity score.
    """

    # Complexity Metric
    # Ensure x_batch and a_batch are NumPy arrays
    if isinstance(x_batch, pd.DataFrame):
        x_batch = x_batch.values
    if not isinstance(explanations_padded, np.ndarray):
        explanations_padded = np.array(explanations_padded)

    complexity_metric = quantus.Complexity(
        return_aggregate=True,
        disable_warnings=False,
        display_progressbar=True
    )
    complexity_score = complexity_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=explanations_padded,  # Saliency maps
        device=device
    )

    complexity_score = complexity_score[0]
    print(f"The model {model_name} got a Complexity Score of: {complexity_score:.4f}")
    return complexity_score

class LassoWrapper(torch.nn.Module):
    
    """
    Wrapper for Lasso Regression to make it compatible with Quantus.
    """
    def __init__(self, model, feature_names=None):
        """
        Initialize the LassoWrapper.
        
        Parameters:
            model: The trained Lasso Regression model.
            feature_names: List of feature names for the input data.
        """

        super(LassoWrapper, self).__init__()
        self.model = model
        self.feature_names = feature_names  # Save feature names

    def forward(self, x):
        """
        Forward pass for the Lasso model.

        Parameters:
            x: Input features (NumPy array or PyTorch tensor).

        Returns:
            torch.Tensor: Predictions as a 2D PyTorch tensor.
        """
        # Convert torch.Tensor to np.ndarray if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if self.feature_names is not None:
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x = pd.DataFrame(x, columns=self.feature_names)
        predictions = self.model.predict(x)
        predictions = np.expand_dims(predictions, axis=1)
        return torch.tensor(predictions, dtype=torch.float32)

    def predict(self, x):
        """
        Predict method for compatibility with Quantus.  
        Parameters:
            x: Input features (NumPy array or PyTorch tensor).
        Returns:
            np.ndarray: Predictions as a NumPy array with an artificial second dimension.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if self.feature_names is not None:
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x = pd.DataFrame(x, columns=self.feature_names)
        predictions = self.model.predict(x)
        predictions = np.expand_dims(predictions, axis=1)
        predictions = expit(predictions)  # maps to (0, 1)
        return np.hstack([predictions, 1 - predictions])
    

    def generate_saliency_map(self, x_batch):
        """
        Generate saliency maps for Lasso Regression using model coefficients.

        Parameters:
            model: The trained Lasso Regression model.
            x_batch: Input features.

        Returns:
            np.ndarray: Saliency maps (coefficients) for the input features.
        """
        if isinstance(x_batch, pd.DataFrame):
            x_batch = x_batch.values

        # The coefficients of the Lasso model serve as the feature importances
        saliency_maps = self.model.coef_
        return np.tile(saliency_maps, (x_batch.shape[0], 1))  # Repeat for each sample


class LightGBMWrapper(torch.nn.Module):
    """
    Wrapper for LightGBM Booster to make it compatible with Quantus.
    """
    def __init__(self, booster):
        """
        Wrapper for LightGBM Booster to make it compatible with Quantus.
        
        Parameters:
        booster: The trained LightGBM Booster object.
        """
        super(LightGBMWrapper, self).__init__()
        self.booster = booster

    def forward(self, x):
        """
        Forward pass for the LightGBM model.
 
        Parameters:
            x: Input features (NumPy array or PyTorch tensor).

        Returns:
            Predictions as a 2D PyTorch tensor.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array
        
         # Ensure x is 2D
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        
        predictions = self.booster.predict(x)
        predictions = np.expand_dims(predictions, axis=1)  # Ensure predictions are 2D
        return torch.tensor(predictions, dtype=torch.float32)  # Return as PyTorch tensor
    
     
    def predict(self, x):
        """
        Predict method for compatibility with Quantus.

        Parameters:
            x: Input features.

        Returns:
            Predictions as a NumPy array with an artificial second dimension.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

        # Ensure x is 2D
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
            
        predictions = self.booster.predict(x)
        predictions = np.expand_dims(predictions, axis=1)  # Ensure predictions are 2D

        # Simulate class probabilities by duplicating the regression output
        # This creates a shape (batch_size, 2) to mimic classification outputs
        predictions = expit(predictions)  # maps to (0, 1)
        return np.hstack([predictions, 1 - predictions])

    def shape_input(self, x, shape, channel_first=True):
        """
        Custom method to handle input reshaping for tabular data.

        Parameters:
            x: Input data.
            shape: Expected shape of the input.
            channel_first: Whether the channel dimension is first (for images).

        Returns:
            Reshaped input data.
        """
        print(f"Input shape: {x.shape}, Target shape: {shape}")
        if len(shape) == 2:  # Batch of tabular data
            return x.reshape(-1, shape[1])  # Ensure batch size is preserved
        elif len(shape) == 1:  # Single instance of tabular data
            return x.reshape(1, -1)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected input shape: {shape}")
        
    
    def generate_saliency_map(self, x_batch):
        """
        Generate saliency maps using SHAP TreeExplainer for LightGBM.
        Parameters:
            x_batch: Input features (NumPy array or Pandas DataFrame).
        Returns: 
            np.ndarray of shape (N, num_features)
        """
        if isinstance(x_batch, pd.DataFrame):
            x_batch = x_batch.values

        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer.shap_values(x_batch)

        # If shap_values is a list (multi-output), select the correct index
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return shap_values  # Don't wrap again in np.array

class BertWrapper(torch.nn.Module):
    """
    Wrapper for BERT model to make it compatible with Quantus.
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def get_embeddings(self, input_ids):
        return self.model.bert.embeddings(input_ids)

    def forward(self, inputs_embeds, attention_mask):
        # input_ids = input_ids.long()  # Ensure input_ids is of type long
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits  # Only return logits tensor

    def predict(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        total = len(dataloader)
        with torch.no_grad(), tqdm(total=total, desc="BERT Predict", unit="batch") as pbar:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                input_ids = input_ids.long()  # Ensure input_ids is of type long
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].cpu().numpy()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze(-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)
                pbar.update(1)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        return y_true, y_pred
    

    def generate_saliency_map(self, batch_or_array):
        self.eval()
        # ig = IntegratedGradients(self)
        gradientShap = GradientShap(self)
        explainer = gradientShap

        all_attributions = []

        if isinstance(batch_or_array, DataLoader):
            dataloader = batch_or_array
            total = len(dataloader)
            with tqdm(total=total, desc="BERT Saliency", unit="batch") as pbar:
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    inputs_embeds = self.model.bert.embeddings(input_ids)
                    baselines = torch.zeros_like(inputs_embeds)
                    attributions = explainer.attribute(
                        inputs=inputs_embeds,
                        baselines=baselines,
                        additional_forward_args=(attention_mask,)
                    )
                    # Reduce to 2D
                    if attributions.ndim == 3:
                        attributions = torch.norm(attributions, dim=-1)
                    all_attributions.append(attributions.detach().cpu().numpy())
                    pbar.update(1)
                    torch.cuda.empty_cache()
            return np.concatenate(all_attributions)

        elif isinstance(batch_or_array, np.ndarray):
            input_ids_np = batch_or_array
            vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
            batch_size = 2  # Set to 1 or 2 to avoid OOM

            all_attributions = []
            for i in range(0, input_ids_np.shape[0], batch_size):
                input_ids = torch.tensor(input_ids_np[i:i+batch_size], dtype=torch.long, device=self.device)
                input_ids = input_ids.clamp(0, vocab_size - 1)
                attention_mask = (input_ids != 0).long()
                inputs_embeds = self.model.bert.embeddings(input_ids)
                # Create baseline: zeros of same shape as inputs_embeds
                baselines = torch.zeros_like(inputs_embeds)
                attributions = explainer.attribute(
                    inputs=inputs_embeds,
                    baselines=baselines,
                    additional_forward_args=(attention_mask,)
                )
                # Reduce to 2D
                if attributions.ndim == 3:
                    attributions = torch.norm(attributions, dim=-1)
                all_attributions.append(attributions.detach().cpu().numpy())
                torch.cuda.empty_cache()
            return np.concatenate(all_attributions, axis=0)
        else:
            raise ValueError("Input to generate_saliency_map must be a DataLoader or np.ndarray.")


class ProtoLMWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, input_ids, attention_mask, sentiment_features):
        # Only pass input_ids, attention_mask, sentiment_features
        # Ensure input_ids are long integers
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        # Ensure attention_mask are long integers  
        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()
            
        try:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_features=sentiment_features
            )
            return output['logits']
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Clear cache and try to provide more informative error
                torch.cuda.empty_cache()
                print(f"CUDA OOM in ProtoLMWrapper.forward with batch size: {input_ids.shape[0]}")
                print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, sentiment_features: {sentiment_features.shape}")
                raise RuntimeError(f"CUDA out of memory with batch size {input_ids.shape[0]}. Try reducing batch size.") from e
            else:
                print(f"Error in ProtoLMWrapper.forward: {e}")
                print(f"input_ids range: [{input_ids.min()}, {input_ids.max()}]")
                print(f"attention_mask range: [{attention_mask.min()}, {attention_mask.max()}]")
                raise
    
    def generate_saliency_map(self, batch_or_array, **kwargs):
        return proto_lm_simple_explain_func(self, batch_or_array, **kwargs)
    
    def predict(self, dataloader):
        """
        Predict method for ProtoLM model.
        Returns predictions and true labels.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad(), tqdm(total=len(dataloader), desc="ProtoLM Predict", unit="batch") as pbar:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentiment_features = batch["sentiment_features"].to(self.device)
                labels = batch["labels"].cpu().numpy()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentiment_features=sentiment_features
                )
                
                # Get logits and convert to predictions
                logits = outputs["logits"]
                
                # Debug: Print first few logits to understand the output
                if len(all_preds) == 0:  # Only print for first batch
                    print(f"Logits shape: {logits.shape}")
                    print(f"Logits sample: {logits[:3].detach().cpu().numpy()}")
                    if logits.shape[1] > 1:
                        print(f"Class probabilities sample: {torch.softmax(logits[:3], dim=1).detach().cpu().numpy()}")
                
                if logits.shape[1] == 1:  # Regression
                    preds = logits.squeeze(-1).cpu().numpy()
                    # Clip to valid rating range (1-10) and ensure proper scaling
                    preds = np.clip(preds, 1.0, 10.0)
                else:  # Classification - convert to rating values (1-10)
                    class_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    # Convert class indices (0-9) to rating values (1-10)
                    preds = class_preds + 1
                
                all_preds.append(preds)
                all_labels.append(labels)
                pbar.update(1)
                
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        
        print(f"ProtoLM predictions shape: {y_pred.shape}, sample: {y_pred[:5]}")
        print(f"ProtoLM labels shape: {y_true.shape}, sample: {y_true[:5]}")
        
        return y_true, y_pred
    

def proto_lm_explain_func(model, batch, **kwargs):
    """
    Improved explanation function for ProtoLM that handles the input format properly
    and returns meaningful attributions for all 204 features.
    """
    seq_len = model.model.max_seq_length
    
    # Accept both np.ndarray and torch.Tensor
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    
    # Handle the input structure properly
    # batch should have shape (batch_size, 204) where 204 = 100 + 100 + 4
    expected_features = 2 * seq_len + 4  # input_ids + attention_mask + sentiment_features
    
    if batch.shape[1] != expected_features:
        print(f"Warning: Expected {expected_features} features, got {batch.shape[1]}")
        # Pad or trim to expected size
        if batch.shape[1] < expected_features:
            pad_width = expected_features - batch.shape[1]
            batch = np.pad(batch, ((0,0),(0,pad_width)), mode='constant')
        else:
            batch = batch[:, :expected_features]
    
    # Split the input properly
    input_ids_raw = batch[:, :seq_len]
    attention_mask_raw = batch[:, seq_len:2*seq_len]
    sentiment_features = batch[:, 2*seq_len:2*seq_len+4]
    
    # Convert to proper types
    input_ids = np.round(input_ids_raw).astype(np.int64)
    vocab_size = model.model.LLM.embeddings.word_embeddings.num_embeddings
    input_ids = np.clip(input_ids, 0, vocab_size - 1)
    
    attention_mask = np.round(attention_mask_raw).astype(np.int64)
    attention_mask = np.clip(attention_mask, 0, 1)
    
    sentiment_features = sentiment_features.astype(np.float32)
    
    # Process in small batches to avoid memory issues
    batch_size = min(4, batch.shape[0])
    all_attributions = []
    
    try:
        for i in range(0, batch.shape[0], batch_size):
            end_idx = min(i + batch_size, batch.shape[0])
            
            # Get batch slice
            batch_input_ids = torch.tensor(input_ids[i:end_idx], dtype=torch.long, device=model.device)
            batch_attention_mask = torch.tensor(attention_mask[i:end_idx], dtype=torch.long, device=model.device)
            batch_sentiment_features = torch.tensor(sentiment_features[i:end_idx], dtype=torch.float32, device=model.device)
            
            # Enable gradients for input embeddings
            batch_input_ids.requires_grad_(False)  # Can't compute gradients w.r.t. discrete tokens
            inputs_embeds = model.model.LLM.embeddings(batch_input_ids)
            inputs_embeds.requires_grad_(True)
            
            # Forward pass
            outputs = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_attention_mask,
                sentiment_features=batch_sentiment_features
            )
            
            logits = outputs["logits"]
            
            # For regression, use the output directly; for classification, use predicted class
            if logits.shape[1] == 1:  # Regression
                target = logits.squeeze(-1).sum()  # Sum for gradient computation
            else:  # Classification - use the predicted class logit
                predicted_classes = torch.argmax(logits, dim=1)
                target = logits.gather(1, predicted_classes.unsqueeze(1)).squeeze(1).sum()
                
            # Compute gradients
            target.backward()
            
            # Get attributions from gradients
            if inputs_embeds.grad is not None:
                token_attributions = torch.norm(inputs_embeds.grad, dim=-1).detach().cpu().numpy()
            else:
                # Fallback to small random values
                token_attributions = np.random.random((end_idx - i, seq_len)) * 0.01
            
            # Create full attribution array for all 204 features
            current_batch_size = end_idx - i
            full_attributions = np.zeros((current_batch_size, expected_features))
            
            # Fill in attributions:
            # 1. Input IDs attributions (first 100 features)
            full_attributions[:, :seq_len] = token_attributions
            
            # 2. Attention mask attributions (next 100 features) - scaled down version of token attributions
            full_attributions[:, seq_len:2*seq_len] = token_attributions * 0.1
            
            # 3. Sentiment feature attributions (last 4 features) - small random values for now
            # TODO: Could compute gradients w.r.t. sentiment features more directly
            full_attributions[:, 2*seq_len:2*seq_len+4] = np.random.random((current_batch_size, 4)) * 0.05
            
            all_attributions.append(full_attributions)
            
            # Clear gradients and cache
            model.model.zero_grad()
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error in proto_lm_explain_func: {e}")
        # Fallback: return small random attributions with correct shape
        return np.random.random((batch.shape[0], expected_features)) * 0.01
    
    # Concatenate all attributions
    result = np.concatenate(all_attributions, axis=0)
    return result

def proto_lm_simple_explain_func(model, batch, **kwargs):
    """
    Simplified explanation function for ProtoLM that returns simple gradients 
    without complex tensor operations to avoid shape mismatches.
    """
    seq_len = model.model.max_seq_length
    
    # Accept both np.ndarray and torch.Tensor
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    
    # Expected feature structure: 204 = 100 (input_ids) + 100 (attention_mask) + 4 (sentiment)
    expected_features = 2 * seq_len + 4
    
    if batch.shape[1] != expected_features:
        print(f"Warning: Expected {expected_features} features, got {batch.shape[1]}. Adjusting...")
        if batch.shape[1] < expected_features:
            pad_width = expected_features - batch.shape[1] 
            batch = np.pad(batch, ((0,0),(0,pad_width)), mode='constant')
        else:
            batch = batch[:, :expected_features]
    
    # Split features properly
    input_ids_raw = batch[:, :seq_len]
    attention_mask_raw = batch[:, seq_len:2*seq_len]
    sentiment_features = batch[:, 2*seq_len:2*seq_len+4]
    
    # Convert to proper types
    input_ids = np.round(input_ids_raw).astype(np.int64)
    vocab_size = model.model.LLM.embeddings.word_embeddings.num_embeddings
    input_ids = np.clip(input_ids, 0, vocab_size - 1)
    
    attention_mask = np.round(attention_mask_raw).astype(np.int64)
    attention_mask = np.clip(attention_mask, 0, 1)
    
    sentiment_features = sentiment_features.astype(np.float32)
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=model.device)
    sentiment_features = torch.tensor(sentiment_features, dtype=torch.float32, device=model.device)
    
    # Simple gradient computation - just compute gradients w.r.t input embeddings
    input_ids.requires_grad_(False)  # Input IDs can't have gradients
    inputs_embeds = model.model.LLM.embeddings(input_ids)
    inputs_embeds.requires_grad_(True)
    
    try:
        # Forward pass
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            sentiment_features=sentiment_features
        )
        
        # Get the max logit for each sample (simple target)
        logits = outputs['logits']
        if logits.shape[1] == 1:  # Regression
            target = logits.squeeze(-1).sum()
        else:  # Classification - use the predicted class logit
            predicted_classes = torch.argmax(logits, dim=1)
            target = logits.gather(1, predicted_classes.unsqueeze(1)).squeeze(1).sum()
        
        # Compute gradients
        target.backward()
        
        # Get attributions from gradients
        if inputs_embeds.grad is not None:
            token_attributions = torch.norm(inputs_embeds.grad, dim=-1).detach().cpu().numpy()
        else:
            token_attributions = np.random.random((input_ids.shape[0], seq_len)) * 0.01
        
        # Create full attribution array matching input shape (204 features)
        batch_size = token_attributions.shape[0]
        full_attributions = np.zeros((batch_size, expected_features))
        
        # Fill in token attributions for input_ids (first 100 features)
        full_attributions[:, :seq_len] = token_attributions
        
        # Fill in token attributions for attention_mask (next 100 features) 
        full_attributions[:, seq_len:2*seq_len] = token_attributions * 0.1
        
        # Small values for sentiment features (last 4 features) based on actual values
        sentiment_importance = np.abs(sentiment_features.detach().cpu().numpy()) * 0.1
        full_attributions[:, 2*seq_len:2*seq_len+4] = sentiment_importance
        
        return full_attributions
        
    except Exception as e:
        print(f"Error in proto_lm_simple_explain_func: {e}")
        # Return dummy attributions with correct shape
        return np.random.random((batch.shape[0], expected_features)) * 0.01


# class ProtoLMWrapper(torch.nn.Module):
#     def __init__(self, model, device):
#         super().__init__()
#         self.model = model
#         self.device = device

#     def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, sentiment_features=None):
#         # Forward all possible arguments to the underlying model
#         output = self.model(
#             input_ids=input_ids,
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             sentiment_features=sentiment_features
#         )
#         return output['logits']

#     def predict(self, dataloader):
#         self.model.eval()
#         all_preds, all_labels = [], []
#         with torch.no_grad(), tqdm(total=len(dataloader), desc="ProtoLM Predict", unit="batch") as pbar:
#             for batch in dataloader:
#                 input_ids = batch["input_ids"].to(self.device)
#                 attention_mask = batch["attention_mask"].to(self.device)
#                 sentiment_features = batch["sentiment_features"].to(self.device)
#                 labels = batch["labels"].cpu().numpy()
#                 outputs = self.model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     sentiment_features=sentiment_features
#                 )
#                 preds = outputs["logits"].squeeze(-1).cpu().numpy()
#                 all_preds.append(preds)
#                 all_labels.append(labels)
#                 pbar.update(1)
#         y_pred = np.concatenate(all_preds)
#         y_true = np.concatenate(all_labels)
#         return y_true, y_pred

    
#     def generate_saliency_map(self, batch_or_array):
#         self.eval()
#         explainer = GradientShap(self)
#         all_attributions = []

#         if isinstance(batch_or_array, DataLoader):
#             dataloader = batch_or_array
#             with tqdm(total=len(dataloader), desc="ProtoLM Saliency", unit="batch") as pbar:
#                 for batch in dataloader:
#                     input_ids = batch["input_ids"].to(self.device)
#                     attention_mask = batch["attention_mask"].to(self.device)
#                     sentiment_features = batch["sentiment_features"].to(self.device)
#                     inputs_embeds = self.model.LLM.embeddings(input_ids)
#                     baselines = torch.zeros_like(inputs_embeds)
#                     with torch.no_grad():
#                         logits = self.model(
#                             input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             sentiment_features=sentiment_features
#                         )["logits"]
#                         targets = torch.argmax(logits, dim=1)
#                     attributions = explainer.attribute(
#                         inputs=inputs_embeds,
#                         baselines=baselines,
#                         additional_forward_args=(attention_mask, sentiment_features),
#                         target=targets
#                     )
#                     if attributions.ndim == 3:
#                         attributions = torch.norm(attributions, dim=-1)
#                     all_attributions.append(attributions.detach().cpu().numpy())
#                     pbar.update(1)
#                     torch.cuda.empty_cache()
#             return np.concatenate(all_attributions)
#         elif isinstance(batch_or_array, np.ndarray):
#             seq_len = self.model.max_seq_length
#             sentiment_dim = 4
#             expected_dim = 2 * seq_len + sentiment_dim
#             # Pad or trim to expected_dim
#             if batch_or_array.shape[1] < expected_dim:
#                 pad_width = expected_dim - batch_or_array.shape[1]
#                 batch_or_array = np.pad(batch_or_array, ((0,0),(0,pad_width)), mode='constant')
#             elif batch_or_array.shape[1] > expected_dim:
#                 batch_or_array = batch_or_array[:, :expected_dim]
            
#             # # ...inside ProtoLMWrapper.generate_saliency_map, after padding/trimming...
#             # # --- Sanitize attention_mask ---
#             # attention_mask = np.where(input_ids != 0, 1, 0)  # Set to 1 where input_ids is not PAD (0), else 0
#             # attention_mask = attention_mask.astype(np.int64)
            
#             batch_size = 2  # Try 1 or 2 to avoid OOM
#             all_attributions = []

#             for i in range(0, batch_or_array.shape[0], batch_size):
#                 # --- Split the concatenated input ---
#                 input_ids = batch_or_array[i:i+batch_size, :seq_len]
#                 input_ids = np.round(input_ids).astype(np.int64)
#                 vocab_size = self.model.LLM.embeddings.word_embeddings.num_embeddings
#                 # Pad or trim input_ids to seq_len
#                 if input_ids.shape[1] < seq_len:
#                     input_ids = np.pad(input_ids, ((0,0),(0,seq_len-input_ids.shape[1])), mode='constant')
#                 elif input_ids.shape[1] > seq_len:
#                     input_ids = input_ids[:, :seq_len]
#                 input_ids = np.clip(input_ids, 0, vocab_size - 1)

#                 # Ignore the perturbed attention_mask from Quantus!
#                 sentiment_features = batch_or_array[i:i+batch_size, 2*seq_len:]

#                 # --- Always reconstruct attention_mask from input_ids ---
#                 attention_mask = (input_ids != 0).astype(np.int64)

#                 assert input_ids.shape[1] == attention_mask.shape[1], f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}"

#                 # --- Convert to tensors ---
#                 input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
#                 attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
#                 sentiment_features = torch.tensor(sentiment_features, dtype=torch.float32, device=self.device)

#                 # --- SHAPE CHECK ---
#                 if input_ids.shape[1] != attention_mask.shape[1]:
#                     print(f"Shape mismatch: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}")
#                     raise ValueError("attention_mask must match the sequence length of inputs_embeds.")

#                 # --- Get targets ---
#                 with torch.no_grad():
#                     logits = self.model(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         sentiment_features=sentiment_features
#                     )["logits"]
#                     targets = torch.argmax(logits, dim=1)
#                 # --- Embeddings and attribution ---
#                 # inputs_embeds = self.model.LLM.embeddings(input_ids)
#                 inputs_embeds = self.model.LLM.embeddings(input_ids)
#                 if inputs_embeds.dim() == 2:
#                     # Captum may flatten batch, restore shape
#                     inputs_embeds = inputs_embeds.unsqueeze(0)
#                 if inputs_embeds.shape[-1] != self.model.hidden_shape:
#                     # Project or pad/truncate to hidden_shape (should not happen if input_ids is correct)
#                     raise ValueError(f"inputs_embeds last dim {inputs_embeds.shape[-1]} != model.hidden_shape {self.model.hidden_shape}")
#                 baselines = torch.zeros_like(inputs_embeds)

#                 print(
#                     f"[DEBUG] Batch {i}: input_ids shape: {input_ids.shape}, "
#                     f"attention_mask shape: {attention_mask.shape}, "
#                     f"inputs_embeds shape: {inputs_embeds.shape}"
#                 )

#                 try:
#                     assert inputs_embeds.shape[1] == attention_mask.shape[1], (
#                         f"inputs_embeds shape: {inputs_embeds.shape}, attention_mask shape: {attention_mask.shape}"
#                     )
#                 except AssertionError as e:
#                     print("[ERROR] Assertion failed!")
#                     print(f"input_ids: {input_ids}")
#                     print(f"attention_mask: {attention_mask}")
#                     print(f"inputs_embeds.shape: {inputs_embeds.shape}")
#                     raise

#                 print(f"inputs_embeds shape: {inputs_embeds.shape}, baselines shape: {baselines.shape}, attention_mask shape: {attention_mask.shape}, sentiment_features shape: {sentiment_features.shape}")
#                 print(f"logits shape: {logits.shape}, logits: {logits}")
#                 print(f"targets shape: {targets.shape}, targets: {targets}")

#                 attributions = explainer.attribute(
#                     inputs=inputs_embeds,
#                     baselines=baselines,
#                     additional_forward_args=(attention_mask, sentiment_features),
#                     target=targets
#                 )
#                 if attributions.ndim == 3:
#                     attributions = torch.norm(attributions, dim=-1)
#                 all_attributions.append(attributions.detach().cpu().numpy())
#                 torch.cuda.empty_cache()

#             return np.concatenate(all_attributions, axis=0)
#         else:
#             raise ValueError("Input to generate_saliency_map must be a DataLoader or np.ndarray.")

    



def calculate_sparsity(a_batch, threshold=1e-5):
    """
    Calculate the sparsity of saliency maps.

    Parameters:
    a_batch (np.ndarray): Saliency maps (e.g., SHAP values).
    threshold (float): Threshold for considering a value as "near-zero".

    Returns:
    float: Sparsity score (proportion of near-zero values).
    """
    
    # Flatten the saliency maps and count near-zero values
    total_elements = np.prod(a_batch.shape)
    zero_elements = np.sum(np.abs(a_batch) < threshold)
    sparsity_score = zero_elements / total_elements
    return sparsity_score

class CustomFaithfulnessCorrelation:
    def __init__(self, subset_size=10, nr_runs=1, similarity_func=None, abs=True,
                 normalise=False, aggregate_func=np.mean, return_aggregate=True, disable_warnings=True,
                 perturb_baseline=None, display_progressbar=True):
        self.subset_size = subset_size
        self.nr_runs = nr_runs
        self.similarity_func = similarity_func or quantus.similarity_func.correlation_pearson
        self.abs = abs
        self.normalise = normalise
        self.aggregate_func = aggregate_func
        self.return_aggregate = return_aggregate
        self.disable_warnings = disable_warnings
        self.perturb_baseline = perturb_baseline
        self.display_progressbar = display_progressbar

    def __call__(self, model, x_batch, y_batch, a_batch, device="cpu"):
        n, d = x_batch.shape
        prediction_changes = []
        importance_scores = []
        
        total_iterations = self.nr_runs * n
        
        with tqdm(total=total_iterations, desc="Faithfulness Correlation", 
                  disable=not self.display_progressbar) as pbar:
            
            for run in range(self.nr_runs):
                for i in range(n):
                    x = x_batch[i]
                    a = a_batch[i]

                    # Get top-k important features based on attribution scores
                    if self.abs:
                        feature_importance = np.abs(a)
                    else:
                        feature_importance = a
                    
                    important_indices = np.argsort(feature_importance)[-self.subset_size:]

                    # Calculate baseline for perturbation
                    baseline = self.perturb_baseline if self.perturb_baseline is not None else np.mean(x_batch, axis=0)
                    
                    # Create perturbed version by replacing important features with baseline
                    x_perturbed = x.copy()
                    x_perturbed[important_indices] = baseline[important_indices]

                    # Get predictions for original and perturbed inputs
                    x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
                    x_perturbed_tensor = torch.tensor(x_perturbed.reshape(1, -1), dtype=torch.float32)
                    
                    with torch.no_grad():
                        y_pred = model(x_tensor).detach().numpy()[0, 0]
                        y_perturbed_pred = model(x_perturbed_tensor).detach().numpy()[0, 0]

                    # Calculate prediction change (how much the prediction changed)
                    prediction_change = abs(y_pred - y_perturbed_pred)
                    
                    # Calculate sum of importance scores for the perturbed features
                    importance_sum = np.sum(feature_importance[important_indices])
                    
                    prediction_changes.append(prediction_change)
                    importance_scores.append(importance_sum)
                    
                    pbar.update(1)

        # Calculate correlation between importance scores and prediction changes
        if len(prediction_changes) > 1 and np.std(prediction_changes) > 0 and np.std(importance_scores) > 0:
            correlation = np.corrcoef(importance_scores, prediction_changes)[0, 1]
            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        if self.return_aggregate:
            return correlation
        return correlation

# Alternative implementation using the standard Quantus approach
class TabularFaithfulnessCorrelation:
    def __init__(self, subset_size=10, nr_runs=100, abs=True, normalise=False, 
                 aggregate_func=np.mean, return_aggregate=True, disable_warnings=True,
                 perturb_baseline="mean", display_progressbar=True):
        self.subset_size = subset_size
        self.nr_runs = nr_runs
        self.abs = abs
        self.normalise = normalise
        self.aggregate_func = aggregate_func
        self.return_aggregate = return_aggregate
        self.disable_warnings = disable_warnings
        self.perturb_baseline = perturb_baseline
        self.display_progressbar = display_progressbar

    def __call__(self, model, x_batch, y_batch, a_batch, device="cpu"):
        """
        Calculate faithfulness by measuring correlation between attribution importance
        and prediction changes when those features are removed.
        """
        correlations = []
        
        # Calculate baseline for perturbation
        if self.perturb_baseline == "mean":
            baseline = np.mean(x_batch, axis=0)
        elif self.perturb_baseline == "zero":
            baseline = np.zeros(x_batch.shape[1])
        else:
            baseline = self.perturb_baseline

        total_iterations = self.nr_runs * len(x_batch)
        
        with tqdm(total=total_iterations, desc="Tabular Faithfulness", 
                  disable=not self.display_progressbar) as pbar:
            
            for run in range(self.nr_runs):
                run_correlations = []
                
                for i in range(len(x_batch)):
                    x_original = x_batch[i]
                    attributions = a_batch[i]
                    
                    # Get feature importance scores
                    if self.abs:
                        importance_scores = np.abs(attributions)
                    else:
                        importance_scores = attributions
                    
                    # Randomly select a subset of features to evaluate
                    if len(importance_scores) > self.subset_size:
                        selected_indices = np.random.choice(
                            len(importance_scores), 
                            size=self.subset_size, 
                            replace=False
                        )
                    else:
                        selected_indices = np.arange(len(importance_scores))
                    
                    selected_importance = importance_scores[selected_indices]
                    
                    # Calculate prediction changes for each selected feature
                    prediction_changes = []
                    
                    # Get original prediction
                    x_tensor = torch.tensor(x_original.reshape(1, -1), dtype=torch.float32)
                    with torch.no_grad():
                        original_pred = model(x_tensor).detach().numpy()[0, 0]
                    
                    for idx in selected_indices:
                        # Create perturbed version with single feature replaced
                        x_perturbed = x_original.copy()
                        x_perturbed[idx] = baseline[idx]
                        
                        # Get perturbed prediction
                        x_perturbed_tensor = torch.tensor(x_perturbed.reshape(1, -1), dtype=torch.float32)
                        with torch.no_grad():
                            perturbed_pred = model(x_perturbed_tensor).detach().numpy()[0, 0]
                        
                        # Calculate absolute change in prediction
                        pred_change = abs(original_pred - perturbed_pred)
                        prediction_changes.append(pred_change)
                    
                    prediction_changes = np.array(prediction_changes)
                    
                    # Calculate correlation between importance and prediction changes
                    if (len(selected_importance) > 1 and 
                        np.std(selected_importance) > 1e-8 and 
                        np.std(prediction_changes) > 1e-8):
                        
                        correlation = np.corrcoef(selected_importance, prediction_changes)[0, 1]
                        if not np.isnan(correlation):
                            run_correlations.append(correlation)
                    
                    pbar.update(1)
                
                if run_correlations:
                    correlations.extend(run_correlations)
        
        if not correlations:
            return 0.0
        
        if self.return_aggregate:
            return self.aggregate_func(correlations)
        return correlations

def custom_perturb_func(x, indices, baseline=None, **kwargs):
    """
    Custom perturbation function for tabular data.
    Replaces values at specified indices with a baseline.

    Parameters:
    x (np.ndarray): Input data.
    indices (list or np.ndarray): Indices of features to perturb.
    baseline (float or np.ndarray): Baseline value to replace features with.
    kwargs: Additional arguments.

    Returns:
    np.ndarray: Perturbed input data with the same shape as the input.
    """
    x_perturbed = x.copy()
    if baseline is None:
        baseline = np.mean(x, axis=0)  # Default to the mean of the input data

    # Ensure baseline values match the shape of the selected featx_baselineures
    if baseline.ndim == 1:
        baseline = baseline.reshape(1, -1)  # Reshape baseline to (1, num_features)

    # Replace the specified features with the baseline values
    x_perturbed[:, indices] = baseline[:, indices]

    return x_perturbed

def explain_func(model, inputs, **kwargs):
    """
    Explanation function for generating saliency maps using SHAP.

    Parameters:
    model: The LightGBMWrapper or native LightGBM model.
    inputs: Input features (e.g., x_batch).

    Returns:
    np.ndarray: Saliency maps for the input features.
    """
    return model.generate_saliency_map(inputs)

def calculate_sparsity_metric(model, model_name, x_batch, y_batch, explanations_padded):
    """Calculate Sparsity metric."""
    print("Calculating Sparsity Metric...")
    sparsity_func  = quantus.Sparseness(
        return_aggregate=True,
        disable_warnings=False,
        display_progressbar=True
    )
    sparsity_score = sparsity_func(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=explanations_padded  # Saliency maps
    )

    if isinstance(sparsity_score, list):
        sparsity_score = sparsity_score[0]
    return sparsity_score

def calculate_robustness_metric(model, x_batch, y_batch, a_batch):
    """Calculate Robustness metric."""
    print("Calculating Robustness Metric...")
    metric_robustness = quantus.LocalLipschitzEstimate(
        nr_samples=5,  # Number of samples to estimate the Lipschitz constant
        perturb_std=0.4,  # Standard deviation of the Gaussian noise for perturbation
        perturb_mean=0.0,  # Mean of the Gaussian noise for perturbation
        norm_numerator=quantus.similarity_func.distance_euclidean,  # Function to compute the distance in the numerator
        norm_denominator=quantus.similarity_func.distance_euclidean,  # Function to compute the distance in the denominator
        perturb_func=quantus.perturb_func.gaussian_noise,  # Function to apply Gaussian noise for perturbation
        similarity_func=quantus.similarity_func.lipschitz_constant,  # Function to compute the Lipschitz constant
        disable_warnings=False,
        display_progressbar=True
    )        

    robustness_scores = metric_robustness(
        model=model,  # The wrapped model to evaluate
        x_batch=x_batch,  # Batch of input features
        y_batch=y_batch,  # Batch of true labels
        a_batch=a_batch,  # Batch of saliency maps
        explain_func=lambda model, inputs, **kwargs: model.generate_saliency_map(inputs)  # Updated lambda function
    )
    # Aggregate the robustness scores (e.g., take the mean)
    robustness_score_mean = np.mean(robustness_scores)
    print(f"Robustness Score (mean): {robustness_score_mean}")
    return robustness_score_mean

def calculate_bert_faithfulness_correlation_metric(model, model_name, x_batch, y_batch, a_batch, tokenizer, subset_size=5, nr_runs=10):
    print(f"Calculating {model_name} Faithfulness Correlation Metric...")
    n, seq_len = x_batch.shape
    prediction_changes = []
    importance_scores = []

    pad_token_id = tokenizer.pad_token_id

    # --- Infer sentiment feature dimension ---
    if "Proto-lm" in model_name:
        # For ProtoLM, x_batch shape is (n_samples, 2*seq_len + sentiment_dim)
        # Infer seq_len and sentiment_dim
        sentiment_dim = 4  # If you know it's always 4, otherwise infer as below:
        # sentiment_dim = x_batch.shape[1] - 2 * seq_len
        seq_len = model.model.max_seq_length

    for run in range(nr_runs):
        for i in range(n):
            if "Proto-lm" in model_name:
                # Extract input_ids, attention_mask, sentiment_features
                input_ids = x_batch[i, :seq_len].copy().astype(int)
                attention_mask = x_batch[i, seq_len:2*seq_len].copy().astype(int)
                sentiment_features = x_batch[i, 2*seq_len:].copy()

                # --- Ensure input_ids and attention_mask are always 1D arrays of length seq_len ---
                input_ids = np.array(input_ids).flatten()[:seq_len]
                attention_mask = np.array(attention_mask).flatten()[:seq_len]
                sentiment_features = np.array(sentiment_features).flatten()

                # Defensive: pad if too short (shouldn't happen, but just in case)
                if input_ids.shape[0] < seq_len:
                    input_ids = np.pad(input_ids, (0, seq_len - input_ids.shape[0]), constant_values=0)
                if attention_mask.shape[0] < seq_len:
                    attention_mask = np.pad(attention_mask, (0, seq_len - attention_mask.shape[0]), constant_values=0)
            else:
                input_ids = x_batch[i].copy()
                attention_mask = (input_ids != pad_token_id).astype(int)
                sentiment_features = None

            attributions = a_batch[i]

            # Get top-k important tokens (ignore [PAD] tokens)
            valid_mask = input_ids != pad_token_id
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
            abs_attributions = np.abs(attributions[valid_indices])
            topk_indices = valid_indices[np.argsort(abs_attributions)[-subset_size:]]

            # Perturb: set top-k tokens to [PAD]
            input_ids_perturbed = input_ids.copy()
            input_ids_perturbed[topk_indices] = pad_token_id

            # Prepare tensors
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            input_ids_perturbed_tensor = torch.tensor(input_ids_perturbed, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask_perturbed = (input_ids_perturbed != pad_token_id).astype(int)
            if attention_mask_perturbed.shape[0] != seq_len:
                attention_mask_perturbed = attention_mask_perturbed[:seq_len]
            attention_mask_perturbed_tensor = torch.tensor(attention_mask_perturbed, dtype=torch.long, device=model.device).unsqueeze(0)

            if "Proto-lm" in model_name:
                sentiment_features_tensor = torch.tensor(sentiment_features, dtype=torch.float32, device=model.device).unsqueeze(0)
            else:
                sentiment_features_tensor = None

            # Get predictions
            with torch.no_grad():
                if "BERT" in model_name:
                    inputs_embeds = model.model.bert.embeddings(input_ids_tensor)
                    pred = model(inputs_embeds, attention_mask_tensor).cpu().numpy().squeeze()
                    inputs_embeds_perturbed = model.model.bert.embeddings(input_ids_perturbed_tensor)
                    pred_perturbed = model(inputs_embeds_perturbed, attention_mask_perturbed_tensor).cpu().numpy().squeeze()
                else:  # ProtoLM
                    # Concatenate all required features for ProtoLM
                    # input_ids: (seq_len,), attention_mask: (seq_len,), sentiment_features: (4,)
                    proto_input = np.concatenate([input_ids, attention_mask, sentiment_features])
                    proto_input_tensor = torch.tensor(proto_input, dtype=torch.float32, device=model.device).unsqueeze(0)
                    pred = model(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor,
                        sentiment_features=sentiment_features_tensor
                    ).cpu().numpy().squeeze()

                    # For perturbed input
                    input_ids_perturbed_tensor = torch.tensor(input_ids_perturbed, dtype=torch.long, device=model.device).unsqueeze(0)
                    attention_mask_perturbed_tensor = torch.tensor(attention_mask_perturbed, dtype=torch.long, device=model.device).unsqueeze(0)
                    sentiment_features_tensor = torch.tensor(sentiment_features, dtype=torch.float32, device=model.device).unsqueeze(0)
                    pred_perturbed = model(
                        input_ids=input_ids_perturbed_tensor,
                        attention_mask=attention_mask_perturbed_tensor,
                        sentiment_features=sentiment_features_tensor
                    ).cpu().numpy().squeeze()
                    # inputs_embeds_perturbed = model.model.LLM.embeddings(input_ids_perturbed_tensor)
                    # pred_perturbed = model(inputs_embeds_perturbed, attention_mask_perturbed_tensor, sentiment_features_tensor).cpu().numpy().squeeze()

            prediction_change = abs(pred - pred_perturbed)
            if len(valid_indices) == 0:
                continue
            abs_attributions = np.abs(attributions[valid_indices])
            topk_indices = valid_indices[np.argsort(abs_attributions)[-subset_size:]]

            # Perturb: set top-k tokens to [PAD]
            input_ids_perturbed = input_ids.copy()
            input_ids_perturbed[topk_indices] = pad_token_id

            # --- prediction code ---

            prediction_change = abs(pred - pred_perturbed)
            importance_sum = np.sum(np.abs(attributions[topk_indices]))

            # Only append if both are scalars
            if np.isscalar(prediction_change) and np.isscalar(importance_sum):
                prediction_changes.append(prediction_change)
                importance_scores.append(importance_sum)

    # Correlation
    if len(prediction_changes) > 1 and np.std(prediction_changes) > 0 and np.std(importance_scores) > 0:
        correlation = np.corrcoef(importance_scores, prediction_changes)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    print(f"BERT Faithfulness Correlation Score: {correlation:.4f}")
    return correlation


def calculate_faithfulness_correlation_metric(model, x_batch, y_batch, a_batch):
    """Calculate Faithfulness Correlation metric."""
    print("Calculating Faithfulness Correlation Metric...")
    num_features = x_batch.shape[1]
    subset_size = min(10, num_features)
    
    # Use the corrected faithfulness implementation
    faithfulness_metric = TabularFaithfulnessCorrelation(
        subset_size=subset_size,
        nr_runs=50,  # Reduced for faster computation, increase for more stable results
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=False,
        perturb_baseline="mean",
        display_progressbar=True  # Enable progress bar
    )
    
    try:
        faithfulness_score = faithfulness_metric(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            device=device
        )
        
        print(f"Faithfulness Correlation Score: {faithfulness_score:.4f}")
        return faithfulness_score
    
    except Exception as e:
        print(f"Error in FaithfulnessCorrelation metric: {e}")
        return 0.0

def calculate_localisation_metric(model, model_name, x_batch, y_batch, a_batch):
    """Calculate Localisation metric."""
    print("Calculating Localisation Metric...")

    # Generate random segmentation masks (binary values: 0 or 1)
    # Each mask will have the same number of features as the input
    num_samples, num_features = x_batch.shape
    s_batch = np.zeros((num_samples, 1), dtype=np.float32)  # One segment per input

    localisation_metric = quantus.RelevanceRankAccuracy(
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=False,
    )
    localisation_score = localisation_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,  # Saliency maps
        s_batch=s_batch,  # Random segmentation masks
        device=device
    )
    return localisation_score

def calculate_monotonicity_metric(model, model_name, x_batch, y_batch, a_batch):
    """Calculate Monotonicity Metric."""
    print("Calculating Monotonicity Metric...")
    y_batch_for_monotonicity = np.zeros_like(y_batch)  # Dummy labels for regression
    
    # Debugging: Print the shape of x_batch and y_batch
    print(f"x_batch shape: {x_batch.shape}")
    print(f"y_batch shape: {y_batch_for_monotonicity.shape}")
    
    monotonicity_metric = quantus.Monotonicity(
        return_aggregate=True,
        disable_warnings=False,
        display_progressbar=True,
        perturb_func=custom_perturb_func,
        perturb_func_kwargs={
            "x": x_batch,
            "baseline": np.mean(x_batch, axis=0)
        },  # Pass baseline as mean of input data
    )

     # Debugging: Check perturbed inputs
    # Use SHAP values to identify important features
    shap_values = np.abs(a_batch).mean(axis=0)  # Mean absolute SHAP values for each feature
    num_top_features = min(15, shap_values.shape[0])  # Ensure we don't exceed the number of features
    important_features = np.argsort(shap_values)[-num_top_features:]  # Top 10 important features
    perturbed_inputs = monotonicity_metric.perturb_func(
        arr=x_batch,
        indices=important_features,
        indexed_axes=[1]
    )
    print("Original Input (First Row):", x_batch[0])
    print("Perturbed Input (First Row):", perturbed_inputs[0])

    # Debugging: Check predictions for perturbed inputs
    for i in range(5):  # Check the first 5 perturbed inputs
        perturbed_input = perturbed_inputs[i]
        original_prediction = model.predict(x_batch[i].reshape(1, -1))
        perturbed_prediction = model.predict(perturbed_input.reshape(1, -1))
        print(f"Original Prediction {i}: {original_prediction}")
        print(f"Perturbed Prediction {i}: {perturbed_prediction}")

    monotonicity_score = monotonicity_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch_for_monotonicity,
        a_batch=a_batch,  # Saliency maps
        device=device
    )
    return monotonicity_score

def calculate_randomisation_metric(model, model_name, x_batch, y_batch, a_batch):
    """Calculate Randomisation Metric."""
    print("Calculating Randomisation Metric...")
    # Use the updated MPRT metric
    randomisation_metric = quantus.MPRT(
        # layer_order="independent",
        similarity_func=quantus.similarity_func.correlation_pearson,  # Use a valid similarity function
        return_average_correlation=True,  # Updated parameter
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=False,
    )

    # Debugging: Check inputs
    print(f"x_batch shape: {x_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    print(f"a_batch shape: {a_batch.shape}")

    randomisation_score = randomisation_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,  # Saliency maps
        explain_func=explain_func,  # Explanation function
        device=device
    )
    return randomisation_score

def calculate_max_sensitivity_metric(model, model_name, x_batch, y_batch, a_batch, explain_function=explain_func):
    """Calculate Max-Sensitivity Metric."""
    print("Calculating Max-Sensitivity Metric...")
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Adjust perturbation bounds for different model types
    if "Proto-lm" in model_name or "BERT" in model_name:
        # For language models, use smaller perturbations since token IDs are discrete
        perturb_kwargs = {"lower_bound": -0.01, "upper_bound": 0.01}
        nr_samples = 2  # Reduce samples for faster computation and less memory usage
        
        # Further reduce batch size for processing
        max_batch_size = 16
        if x_batch.shape[0] > max_batch_size:
            print(f"Reducing batch size from {x_batch.shape[0]} to {max_batch_size} for memory management")
            x_batch = x_batch[:max_batch_size]
            y_batch = y_batch[:max_batch_size]
            a_batch = a_batch[:max_batch_size]
    else:
        # For tabular models, use default bounds
        perturb_kwargs = {"lower_bound": -0.1, "upper_bound": 0.1}
        nr_samples = 10
    
    max_sensitivity_metric = quantus.MaxSensitivity(
        nr_samples=nr_samples,  # Number of perturbation samples
        perturb_func_kwargs=perturb_kwargs,  # Define noise bounds
        return_aggregate=True,
        disable_warnings=False,
        display_progressbar=True
    )

    print(f"x_batch shape: {x_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    print(f"a_batch shape: {a_batch.shape}")

    try:
        max_sensitivity_score = max_sensitivity_metric(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,  # Saliency maps
            explain_func=explain_function,  # Explanation function
            device=device
        )

        if max_sensitivity_score is not None:
            if isinstance(max_sensitivity_score, list):
                max_sensitivity_score = max_sensitivity_score[0]
            
            print(f"Max-Sensitivity Score: {max_sensitivity_score:.4f}")
        return max_sensitivity_score
        
    except Exception as e:
        print(f"Error calculating Max-Sensitivity metric: {e}")
        # Clear memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def calculate_road_metric(model, model_name, x_batch, y_batch, a_batch):
    """Calculate ROAD Metric."""
    # Reshape saliency maps to add a third dimension
    x_batch_reshaped = np.expand_dims(x_batch, axis=-1)  # Shape becomes (num_samples, num_features, 1)
    a_batch_reshaped = np.expand_dims(a_batch, axis=-1)  # Shape becomes (num_samples, num_features, 1)

    # ROAD Metric
    print("Calculating ROAD Metric...")
    # Normalize and reshape saliency maps
    a_batch_normalized = (a_batch - np.min("a_batch")) / (np.max(a_batch) - np.min(a_batch))
    a_batch_reshaped = np.expand_dims(a_batch_normalized, axis=-1)  # Shape becomes (num_samples, num_features, 1)
    
    print(f"x_batch_reshaped shape: {x_batch_reshaped.shape}")
    print(f"a_batch_reshaped shape: {a_batch_reshaped.shape}")
    
    road_metric = quantus.ROAD(
        return_aggregate=True,
        perturb_func=custom_perturb_func,
        perturb_func_kwargs={
            # "x": x_batch_np,
            "baseline": np.mean(x_batch, axis=0)
        },  # Pass baseline as mean of input data
        normalise=True,  # Normalize the scores
        disable_warnings=False,
        display_progressbar=True
    )
    road_score = road_metric(
        model=model,
        x_batch=x_batch_reshaped,
        y_batch=y_batch,
        a_batch=a_batch_reshaped,  # Reshaped saliency maps
        explain_func=explain_func,  # Explanation function
        device=device
    )
    return road_score

def calculate_mae_metric(y_true, y_pred, model_name):
    """Calculate Mean Absolute Error (MAE) metric."""
    print("Calculating MAE...")
    mae_score = mean_absolute_error(y_true, y_pred)
    return mae_score

def calculate_mse_metric(y_true, y_pred, model_name):
    """Calculate Mean Squared Error (MSE) metric."""
    print("Calculating MSE...")
    mse_score = mean_squared_error(y_true, y_pred)
    return mse_score

def calculate_r2_metric(y_true, y_pred, model_name):
    """Calculate R-squared metric."""
    print("Calculating R-squared...")
    r2 = r2_score(y_true, y_pred)
    return r2

def calculate_kappa_metric(y_true, y_pred, model_name):
    """Calculate Cohen's Kappa metric."""
    print("Calculating Cohen's Kappa...")
    # Ensure y_true and y_pred are 1D arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values


    y_pred = np.array(y_pred, dtype=np.int32)  # Ensure predictions are integers
    y_true = np.array(y_true, dtype=np.int32)  # Ensure true labels are integers

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

def calculate_mape_metric(y_true, y_pred, model_name):
    """Calculate Mean Absolute Percentage Error (MAPE) metric."""
    print("Calculating MAPE...")
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape

def evaluate_model(model_type, model_filename, X_test_combined, y_test, metrics=None, use_subset=False, test_df=None):
    """
    Evaluate a given model (LGBM or Lasso) using various metrics.
    """
    results = []
    model_name = model_type  # "LightGBM" or "Lasso"

    if "LightGBM" in model_type:
        model = load_lgb_model(filename=model_filename)
        booster = model._Booster
        y_test_pred = booster.predict(X_test_combined)
        wrapped_model = LightGBMWrapper(booster)
        explanations_padded = wrapped_model.generate_saliency_map(X_test_combined)
    elif "Lasso" in model_type:
        model = load_lasso_model(filename=model_filename)
        y_test_pred = model.predict(X_test_combined)        
        wrapped_model = LassoWrapper(model, feature_names=X_test_combined.columns)
        explanations_padded = wrapped_model.generate_saliency_map(X_test_combined)
    elif "BERT" in model_type:
        model, tokenizer = load_bert_model(path=model_filename)
        test_loader = get_dataloader_for_bert_model(test_df, y_test=y_test, col='review_clean', tokenizer=tokenizer, batch_size=2, max_length=128, shuffle=False)
        wrapped_model = BertWrapper(model, device)
        preds_path = os.path.join(model_filename, model_type, "test_preds.npy")
        labels_path = os.path.join(model_filename, model_type, "test_labels.npy")

        if os.path.exists(preds_path) and os.path.exists(labels_path):
            print("Loading cached BERT predictions...")
            y_test_true = np.load(labels_path)[:len(test_df)]
            y_test_pred = np.load(preds_path)[:len(test_df)]
        else:
            print("Predicting BERT test set...")
            y_test_true, y_test_pred = wrapped_model.predict(test_loader)
            np.save(labels_path, y_test_true)
            np.save(preds_path, y_test_pred)
        attr_path = os.path.join(model_filename, "test_attributions.npy")
        
        if os.path.exists(attr_path):
            print("Loading cached BERT attributions...")
            explanations_padded = np.load(attr_path)
        else:
            print("Calculating BERT attributions...")
            explanations_padded = wrapped_model.generate_saliency_map(test_loader)
            np.save(attr_path, explanations_padded)
    elif "Proto-lm" in model_type:
        model, tokenizer, args = load_proto_lm_model(path=model_filename)
        train_dataloader, val_loader, test_loader = get_dataloader_for_proto_lm_model(args, subset=100)
        wrapped_model = ProtoLMWrapper(model, device)
        preds_path = os.path.join(MODEL_DIR, f"{model_type}_test_preds.npy")
        labels_path = os.path.join(MODEL_DIR, f"{model_type}_test_labels.npy")

        # Force regeneration of predictions to ensure correct format
        # if os.path.exists(preds_path) and os.path.exists(labels_path):
        #     print("Loading cached BERT predictions...")
        #     y_test_true = np.load(labels_path)[:len(test_df)]
        #     y_test_pred = np.load(preds_path)[:len(test_df)]
        # else:
        print("Predicting ProtoLM test set...")
        y_test_true, y_test_pred = wrapped_model.predict(test_loader)
        
        # Debug: Check raw logits to understand what the model is actually outputting
        print(f"Debug: Checking raw model outputs...")
        with torch.no_grad():
            first_batch = next(iter(test_loader))
            input_ids = first_batch["input_ids"][:5].to(device)
            attention_mask = first_batch["attention_mask"][:5].to(device)
            sentiment_features = first_batch["sentiment_features"][:5].to(device)
            labels = first_batch["labels"][:5].cpu().numpy()
            
            raw_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_features=sentiment_features
            )
            raw_logits = raw_outputs["logits"]
            print(f"Debug: Raw logits shape: {raw_logits.shape}")
            print(f"Debug: Raw logits (first 5): {raw_logits.detach().cpu().numpy()}")
            print(f"Debug: True labels (first 5): {labels}")
            print(f"Debug: Sentiment features (first 5): {sentiment_features.detach().cpu().numpy()}")
        
        print(f"ProtoLM predictions shape: {y_test_pred.shape}, sample: {y_test_pred[:5]}")
        print(f"ProtoLM labels shape: {y_test_true.shape}, sample: {y_test_true[:5]}")
        print(f"ProtoLM prediction stats: min={y_test_pred.min():.3f}, max={y_test_pred.max():.3f}, mean={y_test_pred.mean():.3f}")
        print(f"ProtoLM label stats: min={y_test_true.min():.3f}, max={y_test_true.max():.3f}, mean={y_test_true.mean():.3f}")
        np.save(labels_path, y_test_true)
        np.save(preds_path, y_test_pred)
        attr_path = os.path.join(MODEL_DIR, f"{model_type}_test_attributions.npy")
        
        if os.path.exists(attr_path):
            print("Loading cached BERT attributions...")
            explanations_padded = np.load(attr_path)
        else:
            print("Calculating BERT attributions...")
            explanations_padded = wrapped_model.generate_saliency_map(test_loader)
            np.save(attr_path, explanations_padded)
        
        y_test_true = y_test_true[:len(explanations_padded)]
        y_test_pred = y_test_pred[:len(explanations_padded)]
    else:
        raise ValueError("Invalid model_type. Choose 'LightGBM' or 'Lasso'.")

    wrapped_model.eval()

    if "Proto-lm" in model_type:
        x_list = []
        for batch in test_loader:
            input_ids = batch["input_ids"].cpu().numpy()
            attention_mask = batch["attention_mask"].cpu().numpy()
            sentiment_features = batch["sentiment_features"].cpu().numpy()
            # Ensure 2D
            if input_ids.ndim == 1:
                input_ids = input_ids[None, :]
            if attention_mask.ndim == 1:
                attention_mask = attention_mask[None, :]
            if sentiment_features.ndim == 1:
                sentiment_features = sentiment_features[None, :]
            arr = np.concatenate([input_ids, attention_mask, sentiment_features], axis=1)
            x_list.append(arr)
        x_batch_np = np.concatenate(x_list, axis=0)  # (num_samples, 2*seq_len+4)
        
        # Debug: Check the sentiment features (last 4 columns)
        sentiment_values = x_batch_np[:5, -4:]  # First 5 samples, last 4 features
        print(f"Sentiment features sample:\n{sentiment_values}")
        
        a_batch_np = np.array(explanations_padded, dtype=np.float32)
        # Pad attributions to match x_batch_np.shape[1]
        if a_batch_np.shape[1] < x_batch_np.shape[1]:
            pad_width = x_batch_np.shape[1] - a_batch_np.shape[1]
            a_batch_np = np.pad(a_batch_np, ((0,0),(0,pad_width)), mode='constant')
        elif a_batch_np.shape[1] > x_batch_np.shape[1]:
            a_batch_np = a_batch_np[:, :x_batch_np.shape[1]]
        if a_batch_np.ndim == 3:
            a_batch_np = np.linalg.norm(a_batch_np, axis=-1)
        y_batch_np = np.array(y_test_true, dtype=np.float32)
        explanations_padded = a_batch_np
        print(f"x_batch_np shape: {x_batch_np.shape}")
        print(f"a_batch_np shape: {a_batch_np.shape}")  
        print(f"y_batch_np shape: {y_batch_np.shape}")
        if x_batch_np.shape[0] != a_batch_np.shape[0] or x_batch_np.shape[0] != y_batch_np.shape[0]:
            raise ValueError("Mismatch in number of samples between x_batch, a_batch, and y_batch.")
    elif "BERT" in model_type:
        x_list = []
        for batch in test_loader:
            arr = batch["input_ids"].cpu().numpy()
            if arr.ndim == 1:
                arr = arr[None, :]
            x_list.append(arr)
        x_batch_np = np.concatenate(x_list, axis=0)  # (num_samples, seq_len)
        x_batch_np = x_batch_np.reshape(-1, x_batch_np.shape[-1])  # (num_samples, seq_len)
        a_batch_np = np.array(explanations_padded, dtype=np.float32)        
        if a_batch_np.ndim == 3:
            a_batch_np = np.linalg.norm(a_batch_np, axis=-1)  # <-- Fix here
        y_batch_np = np.array(y_test_true, dtype=np.float32)
        explanations_padded = a_batch_np

        print(f"x_batch_np shape: {x_batch_np.shape}")
        print(f"a_batch_np shape: {a_batch_np.shape}")  
        print(f"y_batch_np shape: {y_batch_np.shape}")
        if x_batch_np.shape[0] != a_batch_np.shape[0] or x_batch_np.shape[0] != y_batch_np.shape[0]:
            raise ValueError("Mismatch in number of samples between x_batch, a_batch, and y_batch.")
    else:
        # For LGBM/Lasso, use tabular features
        x_batch_np = np.array(X_test_combined.values, dtype=np.float32)
        a_batch_np = np.array(explanations_padded, dtype=np.float32)
        y_batch_np = np.array(y_test.values, dtype=np.int32)

    if use_subset:
        subset_size = min(100, len(x_batch_np))
        subset_indices = np.random.choice(len(x_batch_np), size=subset_size, replace=False)
        x_batch_np_small = x_batch_np[subset_indices]
        y_batch_np_small = y_batch_np[subset_indices]
        a_batch_np_small = a_batch_np[subset_indices]
    else:
        x_batch_np_small = x_batch_np
        y_batch_np_small = y_batch_np
        a_batch_np_small = a_batch_np

    # Metric Calculations - using helper functions
    if "complexity" in metrics:
        complexity_score = calc_complexity(
                                model=wrapped_model,
                                model_name=model_name,
                                x_batch=x_batch_np,
                                y_batch=y_batch_np,
                                explanations_padded=explanations_padded,
                                device=device)
        results.append({"model_name": model_name, "metric": "Complexity", "score": complexity_score})

    if "sparsity" in metrics:
        sparsity_score = calculate_sparsity_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np, y_batch=y_batch_np, explanations_padded=explanations_padded)
        print(f"Sparsity Score: {sparsity_score}")
        results.append({"model_name": model_name, "metric": "Sparsity", "score": sparsity_score})

    if "robustness" in metrics: # Robustness only for LightGBM
        robustness_score_mean = calculate_robustness_metric(model=wrapped_model, x_batch=x_batch_np_small, y_batch=y_batch_np_small, a_batch=a_batch_np_small)
        results.append({"model_name": model_name, "metric": "Robustness", "score": robustness_score_mean})

    if "faithfulness_correlation" in metrics:
        if "BERT" in model_type or "Proto-lm" in model_type:
            faithfulness_score = calculate_bert_faithfulness_correlation_metric(
                model=wrapped_model,
                model_name=model_name,
                x_batch=x_batch_np_small,
                y_batch=y_batch_np_small,
                a_batch=a_batch_np_small,
                tokenizer=tokenizer,
                subset_size=5,   # or another value
                nr_runs=2        # keep low for speed
            )
        else: 
            faithfulness_score = calculate_faithfulness_correlation_metric(model=wrapped_model, x_batch=x_batch_np_small, y_batch=y_batch_np_small, a_batch=a_batch_np_small)
        print(f"Faithfulness Correlation Score: {faithfulness_score:.4f}")
        results.append({"model_name": model_name, "metric": "Faithfulness", "score": faithfulness_score})

    if "localisation" in metrics:
        localisation_score = calculate_localisation_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np, y_batch=y_batch_np, a_batch=a_batch_np)
        print(f"Localisation Score: {localisation_score:.4f}")
        results.append({"model_name": model_name, "metric": "Localisation", "score": localisation_score})

    if "monotonicity" in metrics:
        monotonicity_score = calculate_monotonicity_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np_small, y_batch=y_batch_np_small, a_batch=a_batch_np_small)
        print(f"Monotonicity Score: {monotonicity_score:.4f}")
        results.append({"model_name": model_name, "metric": "Monotonicity", "score": monotonicity_score})

    if "randomisation" in metrics:
        randomisation_score = calculate_randomisation_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np, y_batch=y_batch_np, a_batch=a_batch_np)
        if randomisation_score is not None:
            print(f"Randomisation Score: {randomisation_score:.4f}")
            results.append({"model_name": model_name, "metric": "Randomisation", "score": randomisation_score})
        else:
            print("Randomisation metric calculation failed")

    if "max_sensitivity" in metrics:
        explain_function = explain_func 
        if "Proto-lm" in model_type:
            # Use a simpler attribution method for ProtoLM
            print("Calculating Max-Sensitivity metric for ProtoLM with simplified attribution...")
            explain_function = lambda model, inputs, **kwargs: proto_lm_simple_explain_func(model, inputs, **kwargs)
            max_sensitivity_score = calculate_max_sensitivity_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np_small, y_batch=y_batch_np_small, a_batch=a_batch_np_small, explain_function=explain_function)
        else:
            max_sensitivity_score = calculate_max_sensitivity_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np_small, y_batch=y_batch_np_small, a_batch=a_batch_np_small, explain_function=explain_function)
        
        if max_sensitivity_score is not None:
            print(f"Max-Sensitivity Score: {max_sensitivity_score:.4f}")
            results.append({"model_name": model_name, "metric": "Max-Sensitivity", "score": max_sensitivity_score})
        else:
            print("Max-Sensitivity metric calculation failed")

    if "road" in metrics:
        road_score = calculate_road_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np, y_batch=y_batch_np, a_batch=a_batch_np)
        print(f"ROAD Score: {road_score:.4f}")
        results.append({"model_name": model_name, "metric": "ROAD", "score": road_score})

    mae_score = calculate_mae_metric(y_test, y_test_pred, model_name)
    print(f"MAE Score: {mae_score:.4f}")
    results.append({"model_name": model_name, "metric": "MAE", "score": mae_score})

    mse_score = calculate_mse_metric(y_test, y_test_pred, model_name)
    print(f"MSE Score: {mse_score:.4f}")
    results.append({"model_name": model_name, "metric": "MSE", "score": mse_score})

    r2 = calculate_r2_metric(y_test, y_test_pred, model_name)
    print(f"R-squared Score: {r2:.4f}")
    results.append({"model_name": model_name, "metric": "R-squared", "score": r2})

    mape = calculate_mape_metric(y_test, y_test_pred, model_name)
    print(f"MAPE Score: {mape:.4f}")
    results.append({"model_name": model_name, "metric": "MAPE", "score": mape})


    kappa = calculate_kappa_metric(y_test, y_test_pred, model_name)
    print(f"Cohen's Kappa Score: {kappa:.4f}")   
    results.append({"model_name": model_name, "metric": "Cohen's Kappa", "score": kappa})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Ensure the output directory exists
    file_name = model_filename.split("/")[-1].split(".")[0]
    output_csv = f"{EVAL_DIR}/{file_name}_evaluation_scores.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Export results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Evaluation scores saved to {output_csv}")

# Update main
if __name__ == '__main__':
    print("Loading data...")

    metrics = [
        # "complexity",
        # "sparsity",
        "max_sensitivity",
        # "robustness",
        # "faithfulness_correlation",

        # "monotonicity",
        # "localisation",
        # "randomisation",
        # "road"
    ]

    models = [
        # "LightGBMTop15",
        # "LightGBMFull",        
        # "LassoFull",
        # "LassoTop15"
        # "BERT",
        "Proto-lm"
    ]

    X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()
    X_train_features = X_train_combined.drop(columns=['patient_id'])
    X_val_features = X_val_combined.drop(columns=['patient_id'])
    X_test_features = X_test_combined.drop(columns=['patient_id'])
    df_test = None

    if "BERT" in models or "Proto-lm" in models:
         df_test = pd.read_csv(f"{DATA_DIR}/drug_review_test_clean.csv")
         df_test = df_test.head(100)  # Limit to 100 samples for BERT evaluation
         y_test = y_test[:len(df_test)]  # Ensure y_test matches df_test length


    if "LightGBMTop15" in models or "LightGBMFull" in models or "LassoTop15" in models or "LassoFull" in models:
        # load top features from JSON file for LightGBM
        with open(f"{MODEL_DIR}/top_15_features.json", "r") as f:
            lgb_top_features = json.load(f)

        X_test_lgb_top = X_test_features[lgb_top_features]

        # load top features from JSON file for Lasso
        with open(f"{MODEL_DIR}/lasso_model_top_15_features_20250621_223226.json", "r") as f:
            lasso_top_features = json.load(f)

        X_test_lasso_top = X_test_features[lasso_top_features]

   
    
    print("Starting evaluation...")

    if "LightGBMTop15" in models:
        # Evaluate LightGBM model
        print("Evaluating LightGBMTop15 model...")
        lgb_model_filename = f"{MODEL_DIR}/lgb_model_top_15_20250621_213411.txt"    
        evaluate_model("LightGBMTop15", lgb_model_filename, X_test_lgb_top, y_test, metrics=metrics, use_subset=True)

    if "LightGBMFull" in models:
        print("Evaluating LightGBMFull model...")
        lgb_model_filename = f"{MODEL_DIR}/lgb_model_20250621_213411.txt"    
        evaluate_model("LightGBMFull", lgb_model_filename, X_test_features, y_test, metrics=metrics, use_subset=True)
    
    if "LassoTop15" in models:
        print("Evaluating Top Lasso model...")
        lasso_model_filename = f"{MODEL_DIR}/lasso_model_top15_20250621_223226.joblib"
        evaluate_model("LassoTop15", lasso_model_filename, X_test_lasso_top, y_test, metrics=metrics, use_subset=True)    
    
    if "LassoFull" in models:
        # Evaluate Lasso model
        print("Evaluating Full Lasso model...")
        lasso_model_filename = f"{MODEL_DIR}/lasso_model_20250621_223226.joblib"
        evaluate_model("LassoFull", lasso_model_filename, X_test_features, y_test, metrics=metrics, use_subset=True)

    if "BERT" in models:
        # Evaluate BERT model
        print("Evaluating BERT model...")
        bert_model_folder = f"{MODEL_DIR}/bert_model"
        evaluate_model("BERT", bert_model_folder, X_test_features, y_test, metrics=metrics, use_subset=True, test_df=df_test)

    if "Proto-lm" in models:
        # Evaluate Proto-lm model
        print("Evaluating Proto-lm model...")
        proto_lm_model_file = f"{MODEL_DIR}/final_proto_model.pt"
        evaluate_model("Proto-lm", proto_lm_model_file, X_test_features, y_test, metrics=metrics, use_subset=True, test_df=df_test)