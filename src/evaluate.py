import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, cohen_kappa_score
from lightgbm import LGBMRegressor
from feature_extraction import load_data   
import quantus
import torch
import os
import json
import shap
from tqdm import tqdm
from scipy.special import expit  # sigmoid
from sklearn.linear_model import Lasso
import joblib
from check_the_data import feature_extraction # Import the function
from scipy.sparse import hstack
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from bert import DrugReviewDataset
from captum.attr import IntegratedGradients, Saliency, GradientShap
from torch.utils.data import DataLoader
import torch


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
    import joblib  # Import joblib here to avoid it being a global requirement
    if filename is None:
        filename = f"{MODEL_DIR}/trained_lasso_model.joblib"  # Example filename
    lasso_model = joblib.load(filename)
    return lasso_model


def load_bert_model(path=None):
    """
    Load a BERT model from a file.

    Parameters:
    path (str): The path to the model file.

    Returns:
    BertModel: The loaded BERT model.
    """    
    if path is None:
        path = f"{MODEL_DIR}/bert_model"
    
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.to(device)  # Move the model to the appropriate device (GPU or CPU)
    return model, tokenizer


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
    def __init__(self, model, feature_names=None):
        super(LassoWrapper, self).__init__()
        self.model = model
        self.feature_names = feature_names  # Save feature names

    def forward(self, x):
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
        Returns: np.ndarray of shape (N, num_features)
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

def calculate_bert_faithfulness_correlation_metric(model, x_batch, y_batch, a_batch, tokenizer, subset_size=5, nr_runs=10):
    """
    Faithfulness correlation for BERT: 
    Perturb top-k tokens (by attribution) to [PAD] and measure prediction change.
    """
    print("Calculating BERT Faithfulness Correlation Metric...")
    n, seq_len = x_batch.shape
    prediction_changes = []
    importance_scores = []

    pad_token_id = tokenizer.pad_token_id

    for run in range(nr_runs):
        for i in range(n):
            input_ids = x_batch[i].copy()
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
            attention_mask = (input_ids_tensor != pad_token_id).long()
            attention_mask_perturbed = (input_ids_perturbed_tensor != pad_token_id).long()

            # Get predictions
            with torch.no_grad():
                inputs_embeds = model.model.bert.embeddings(input_ids_tensor)
                pred = model(inputs_embeds, attention_mask).cpu().numpy().squeeze()
                inputs_embeds_perturbed = model.model.bert.embeddings(input_ids_perturbed_tensor)
                pred_perturbed = model(inputs_embeds_perturbed, attention_mask_perturbed).cpu().numpy().squeeze()

            prediction_change = abs(pred - pred_perturbed)
            importance_sum = np.sum(np.abs(attributions[topk_indices]))

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

def calculate_max_sensitivity_metric(model, model_name, x_batch, y_batch, a_batch):
    """Calculate Max-Sensitivity Metric."""
    print("Calculating Max-Sensitivity Metric...")
    max_sensitivity_metric = quantus.MaxSensitivity(
        nr_samples=10,  # Number of perturbation samples
        perturb_func_kwargs={"lower_bound": -0.1, "upper_bound": 0.1},  # Define noise bounds
        return_aggregate=True,
        disable_warnings=False,
        display_progressbar=True
    )
    max_sensitivity_score = max_sensitivity_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,  # Saliency maps
        explain_func=explain_func,  # Explanation function
        device=device
    )

    if isinstance(max_sensitivity_score, list):
        max_sensitivity_score = max_sensitivity_score[0]
    return max_sensitivity_score

def calculate_road_metric(model, model_name, x_batch, y_batch, a_batch):
    """Calculate ROAD Metric."""
    # Reshape saliency maps to add a third dimension
    x_batch_reshaped = np.expand_dims(x_batch, axis=-1)  # Shape becomes (num_samples, num_features, 1)
    a_batch_reshaped = np.expand_dims(a_batch, axis=-1)  # Shape becomes (num_samples, num_features, 1)

    # ROAD Metric
    print("Calculating ROAD Metric...")
    # Normalize and reshape saliency maps
    a_batch_normalized = (a_batch - np.min(a_batch)) / (np.max(a_batch) - np.min(a_batch))
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
        bert_preds_path = os.path.join(model_filename, "test_preds.npy")
        bert_labels_path = os.path.join(model_filename, "test_labels.npy")

        if os.path.exists(bert_preds_path) and os.path.exists(bert_labels_path):
            print("Loading cached BERT predictions...")
            y_test_true = np.load(bert_labels_path)[:len(test_df)]
            y_test_pred = np.load(bert_preds_path)[:len(test_df)]
        else:
            print("Predicting BERT test set...")
            y_test_true, y_test_pred = wrapped_model.predict(test_loader)
            np.save(bert_labels_path, y_test_true)
            np.save(bert_preds_path, y_test_pred)
        bert_attr_path = os.path.join(model_filename, "test_attributions.npy")
        
        if os.path.exists(bert_attr_path):
            print("Loading cached BERT attributions...")
            explanations_padded = np.load(bert_attr_path)
        else:
            print("Calculating BERT attributions...")
            explanations_padded = wrapped_model.generate_saliency_map(test_loader)
            np.save(bert_attr_path, explanations_padded)
    else:
        raise ValueError("Invalid model_type. Choose 'LightGBM' or 'Lasso'.")

    wrapped_model.eval()

    if "BERT" in model_type:
        x_batch_np = np.stack([batch["input_ids"].cpu().numpy() for batch in test_loader])
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
        if "BERT" in model_type:
            faithfulness_score = calculate_bert_faithfulness_correlation_metric(
                model=wrapped_model,
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
        print(f"Randomisation Score: {randomisation_score:.4f}")
        results.append({"model_name": model_name, "metric": "Randomisation", "score": randomisation_score})

    if "max_sensitivity" in metrics:
        max_sensitivity_score = calculate_max_sensitivity_metric(model=wrapped_model, model_name=model_name, x_batch=x_batch_np_small, y_batch=y_batch_np_small, a_batch=a_batch_np_small)
        print(f"Max-Sensitivity Score: {max_sensitivity_score:.4f}")
        results.append({"model_name": model_name, "metric": "Max-Sensitivity", "score": max_sensitivity_score})

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
        "complexity",
        "sparsity",
        "max_sensitivity",
        "robustness",
        "faithfulness_correlation",

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
        "BERT"
    ]

    X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()
    X_train_features = X_train_combined.drop(columns=['patient_id'])
    X_val_features = X_val_combined.drop(columns=['patient_id'])
    X_test_features = X_test_combined.drop(columns=['patient_id'])
    df_test = None

    if "BERT" in models:
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