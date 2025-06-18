
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from GMB import load_data   
import quantus
import torch
import os
import shap
from tqdm import tqdm
from scipy.special import expit  # sigmoid



# Global variable to hold the loaded model
MODEL_DIR = "Models"
EVAL_DIR = "Evaluation"
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

# def generate_saliency_map_lgb(booster, x_batch):
#     """
#     Generate saliency maps using SHAP TreeExplainer for LightGBM.

#     Parameters:
#     booster: The trained LightGBM Booster object.
#     x_batch: Input features (e.g., X_test_combined).

#     Returns:
#     np.ndarray: SHAP values (feature attributions) for the input features.
#     """
#     # Ensure x_batch is a NumPy array
#     if isinstance(x_batch, pd.DataFrame):
#         x_batch = x_batch.values

#     # Initialize SHAP TreeExplainer
#     explainer = shap.TreeExplainer(booster)

#     # Compute SHAP values
#     shap_values = explainer.shap_values(x_batch)

#     # Return SHAP values
#     return np.array(shap_values)

def generate_saliency_map_lgb(booster, x_batch):
    """
    Generate saliency maps using SHAP TreeExplainer for LightGBM.
    Returns: np.ndarray of shape (N, num_features)
    """
    if isinstance(x_batch, pd.DataFrame):
        x_batch = x_batch.values

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_batch)

    # If shap_values is a list (multi-output), select the correct index
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return shap_values  # Don't wrap again in np.array



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
    # Use the native LightGBM booster for SHAP explanations
    if isinstance(model, LightGBMWrapper):
        booster = model.booster
    else:
        booster = model

    return generate_saliency_map_lgb(booster, inputs)


def evaluate_lgb_model(model_filename, X_test_combined, y_test, metrics=None, use_subset=False):
    """
    Evaluate the LightGBM model using various metrics, save the results into csv in Evaluation folder.
    Parameters:
        model_filename (str): Path to the saved LightGBM model file.
        X_test_combined (pd.DataFrame): Combined test features.
        y_test (pd.Series): True labels for the test set.
        metrics (list): List of metrics to calculate. If None, all metrics are calculated.
        use_subset (bool): Whether to use a smaller subset of the data for evaluation.
    """

    # Initialize a list to store evaluation results
    results = []

    # Model name
    model_name = "LightGBM"
    
    # Load the LightGBM model
    lgb_model = load_lgb_model(filename=model_filename)
    booster = lgb_model._Booster  # Get the Booster object

    # Use the loaded model for predictions
    y_test_pred = booster.predict(X_test_combined)

    # Generate saliency maps for the test dataset
    print("Generating saliency maps...")
    explanations_padded = generate_saliency_map_lgb(booster, X_test_combined)

     # Wrap the LightGBM model
    wrapped_model = LightGBMWrapper(booster)
    wrapped_model.eval()

    # Call calc_complexity for the test dataset
    if "complexity" in metrics:
        print("Calculating Complexity Metric...")
        complexity_score = calc_complexity(
            model=wrapped_model,  # Use the Booster's predict function
            model_name="LightGBM",
            x_batch=X_test_combined,
            y_batch=y_test,
            explanations_padded=explanations_padded,
            device=device
        )
        results.append({"model_name": model_name, "metric": "Complexity", "score": complexity_score})


    # Validate and reshape SHAP output if needed
    if isinstance(explanations_padded, list):
        explanations_padded = np.array(explanations_padded)

    # Convert to NumPy to avoid misinterpretation
    # Ensure NumPy arrays and proper types
    x_batch_np = np.array(X_test_combined.values, dtype=np.float32)  # Ensure it's float32 or float64
    a_batch_np = np.array(explanations_padded, dtype=np.float32)
    y_batch_np = np.array(y_test.values, dtype=np.int32)  # or float32 depending on your task


    # Evaluate the robustness of the model using the Local Lipschitz Estimate metric
    if use_subset:
        subset_size = min(10000, len(x_batch_np))  # Ensure we don't exceed the available data
        subset_indices = np.random.choice(len(x_batch_np), size=subset_size, replace=False)
        x_batch_np_small = x_batch_np[subset_indices]
        y_batch_np_small = y_batch_np[subset_indices]
        a_batch_np_small = a_batch_np[subset_indices]
    else:
        x_batch_np_small = x_batch_np 
        y_batch_np_small = y_batch_np 
        a_batch_np_small = a_batch_np
    

    num_features = x_batch_np.shape[1]
    subset_size = min(10, num_features - 1)  # Ensure it's strictly less
    print(f"Subset size used: {subset_size}")


    # Sparsity Metric
    if "sparsity" in metrics:
        print("Calculating Sparsity Metric...")
        sparsity_func  = quantus.Sparseness(
            return_aggregate=True,
            disable_warnings=False,
            display_progressbar=True
        )
        sparsity_score = sparsity_func(
            model=wrapped_model,
            x_batch=x_batch_np,
            y_batch=y_batch_np,
            a_batch=explanations_padded  # Saliency maps
        )

        if isinstance(sparsity_score, list):
            sparsity_score = sparsity_score[0]

        # sparsity_score = calculate_sparsity(a_batch_np, threshold=1e-5)
        results.append({"model_name": model_name, "metric": "Sparsity", "score": sparsity_score})


    if "robustness" in metrics:
        # Robustness Metric
        print("Calculating Robustness Metric...")
        metric_robustness = quantus.LocalLipschitzEstimate(
            nr_samples=10,  # Number of samples to estimate the Lipschitz constant
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
            model=wrapped_model,  # The wrapped model to evaluate
            x_batch=x_batch_np_small,  # Batch of input features
            y_batch=y_batch_np_small,  # Batch of true labels
            a_batch=a_batch_np_small,  # Batch of saliency maps
            explain_func=lambda model, inputs, **kwargs: generate_saliency_map_lgb(booster, inputs)  # Updated lambda function
        )
        # Aggregate the robustness scores (e.g., take the mean)
        robustness_score_mean = np.mean(robustness_scores)
        print(f"Robustness Score (mean): {robustness_score_mean}")
        results.append({"model_name": model_name, "metric": "Robustness", "score": robustness_score_mean})

    if "faithfulness_correlation" in metrics:
        print("Calculating Faithfulness Correlation Metric...")
        
        num_features = x_batch_np.shape[1]
        subset_size = min(10, num_features)
        
        # Use the corrected faithfulness implementation
        faithfulness_metric = TabularFaithfulnessCorrelation(
            subset_size=subset_size,
            nr_runs=50,  # Reduced for faster computation, increase for more stable results
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
            perturb_baseline="mean",
            display_progressbar=True  # Enable progress bar
        )
        
        try:
            faithfulness_score = faithfulness_metric(
                model=wrapped_model,
                x_batch=x_batch_np_small, #x_batch_np,
                y_batch=y_batch_np_small, # y_batch_np,
                a_batch=a_batch_np_small, #a_batch_np,
                device=device
            )
            
            print(f"Faithfulness Correlation Score: {faithfulness_score:.4f}")
            results.append({"model_name": model_name, "metric": "Faithfulness", "score": faithfulness_score})
        
        except Exception as e:
            print(f"Error in FaithfulnessCorrelation metric: {e}")
            results.append({"model_name": model_name, "metric": "Faithfulness", "score": 0.0})

    if "localisation" in metrics:
        print("Calculating Localisation Metric...")

        # Generate random segmentation masks (binary values: 0 or 1)
        # Each mask will have the same number of features as the input
        num_samples, num_features = x_batch_np.shape
        s_batch = np.zeros((num_samples, 1), dtype=np.float32)  # One segment per input


        localisation_metric = quantus.RelevanceRankAccuracy(
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        )
        localisation_score = localisation_metric(
            model=wrapped_model,
            x_batch=x_batch_np,
            y_batch=y_batch_np,
            a_batch=a_batch_np,  # Saliency maps
            s_batch=s_batch,  # Random segmentation masks
            device=device
        )
        results.append({"model_name": model_name, "metric": "Localisation", "score": localisation_score})

    if "monotonicity" in metrics:
        print("Calculating Monotonicity Metric...")
        y_batch_for_monotonicity = np.zeros_like(y_batch_np_small) # np.zeros_like(y_batch_np)  # Dummy labels for regression
        
        # Debugging: Print the shape of x_batch and y_batch
        print(f"x_batch shape: {x_batch_np_small.shape}")
        print(f"y_batch shape: {y_batch_for_monotonicity.shape}")
        
       
        
        monotonicity_metric = quantus.Monotonicity(
            return_aggregate=True,
            disable_warnings=False,
            display_progressbar=True,
            perturb_func=custom_perturb_func,
            perturb_func_kwargs={
                "x": x_batch_np_small,
                "baseline": np.mean(x_batch_np, axis=0)
            },  # Pass baseline as mean of input data
        )

         # Debugging: Check perturbed inputs
        # Use SHAP values to identify important features
        shap_values = np.abs(a_batch_np_small).mean(axis=0)  # Mean absolute SHAP values for each feature
        num_top_features = min(15, shap_values.shape[0])  # Ensure we don't exceed the number of features
        important_features = np.argsort(shap_values)[-num_top_features:]  # Top 10 important features
        perturbed_inputs = monotonicity_metric.perturb_func(
            arr=x_batch_np_small,
            indices=important_features,
            indexed_axes=[1]
        )
        print("Original Input (First Row):", x_batch_np_small[0])
        print("Perturbed Input (First Row):", perturbed_inputs[0])

        # Debugging: Check predictions for perturbed inputs
        for i in range(5):  # Check the first 5 perturbed inputs
            perturbed_input = perturbed_inputs[i]
            original_prediction = wrapped_model.predict(x_batch_np_small[i].reshape(1, -1))
            perturbed_prediction = wrapped_model.predict(perturbed_input.reshape(1, -1))
            print(f"Original Prediction {i}: {original_prediction}")
            print(f"Perturbed Prediction {i}: {perturbed_prediction}")

        monotonicity_score = monotonicity_metric(
            model=wrapped_model,
            x_batch=x_batch_np_small,# x_batch_np,
            y_batch=y_batch_for_monotonicity,
            a_batch=a_batch_np_small,#a_batch_np,  # Saliency maps
            device=device
        )
        results.append({"model_name": model_name, "metric": "Monotonicity", "score": monotonicity_score})
    
    if "randomisation" in metrics:
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
            disable_warnings=True,
        )

        # Debugging: Check inputs
        print(f"x_batch_np shape: {x_batch_np.shape}")
        print(f"y_batch_np shape: {y_batch_np.shape}")
        print(f"a_batch_np shape: {a_batch_np.shape}")

        randomisation_score = randomisation_metric(
            model=wrapped_model,
            x_batch=x_batch_np,
            y_batch=y_batch_np,
            a_batch=a_batch_np,  # Saliency maps
            explain_func=explain_func,  # Explanation function
            device=device
        )
        results.append({"model_name": model_name, "metric": "Randomisation", "score": randomisation_score})

    if "max_sensitivity" in metrics:
        print("Calculating Max-Sensitivity Metric...")
        max_sensitivity_metric = quantus.MaxSensitivity(
            nr_samples=10,  # Number of perturbation samples
            perturb_func_kwargs={"lower_bound": -0.1, "upper_bound": 0.1},  # Define noise bounds
            return_aggregate=True,
            disable_warnings=False,
            display_progressbar=True
        )
        max_sensitivity_score = max_sensitivity_metric(
            model=wrapped_model,
            x_batch=x_batch_np_small,#x_batch_np,
            y_batch=y_batch_np_small, #y_batch_np,
            a_batch=a_batch_np_small, # a_batch_np,  # Saliency maps
            explain_func=explain_func,  # Explanation function
            device=device
        )

        if isinstance(max_sensitivity_score, list):
            max_sensitivity_score = max_sensitivity_score[0]
        results.append({"model_name": model_name, "metric": "Max-Sensitivity", "score": max_sensitivity_score})
    
    
    if "road" in metrics:
        # Reshape saliency maps to add a third dimension
        x_batch_np_reshaped = np.expand_dims(x_batch_np, axis=-1)  # Shape becomes (num_samples, num_features, 1)
        a_batch_np_reshaped = np.expand_dims(a_batch_np, axis=-1)  # Shape becomes (num_samples, num_features, 1)

        # ROAD Metric
        print("Calculating ROAD Metric...")
        # Normalize and reshape saliency maps
        a_batch_np_normalized = (a_batch_np - np.min(a_batch_np)) / (np.max(a_batch_np) - np.min(a_batch_np))
        a_batch_np_reshaped = np.expand_dims(a_batch_np_normalized, axis=-1)  # Shape becomes (num_samples, num_features, 1)
        
        print(f"x_batch_np_reshaped shape: {x_batch_np_reshaped.shape}")
        print(f"a_batch_np_reshaped shape: {a_batch_np_reshaped.shape}")
        
        road_metric = quantus.ROAD(
            return_aggregate=True,
            perturb_func=custom_perturb_func,
            perturb_func_kwargs={
                # "x": x_batch_np,
                "baseline": np.mean(x_batch_np, axis=0)
            },  # Pass baseline as mean of input data
            normalise=True,  # Normalize the scores
            disable_warnings=False,
            display_progressbar=True
        )
        road_score = road_metric(
            model=wrapped_model,
            x_batch=x_batch_np_reshaped,
            y_batch=y_batch_np,
            a_batch=a_batch_np_reshaped,  # Reshaped saliency maps
            explain_func=explain_func,  # Explanation function
            device=device
        )
        results.append({"model_name": model_name, "metric": "ROAD", "score": road_score})

    # Mean Absolute Error (MAE)
    print("Calculating MAE...")
    mae_score = mean_absolute_error(y_test, y_test_pred)
    results.append({"model_name": model_name, "metric": "MAE", "score": mae_score})

    # Mean Squared Error (MSE)
    print("Calculating MSE...")
    mse_score = mean_squared_error(y_test, y_test_pred)
    results.append({"model_name": model_name, "metric": "MSE", "score": mse_score})

    # R-squared
    print("Calculating R-squared...")
    r2 = r2_score(y_test, y_test_pred)
    results.append({"model_name": model_name, "metric": "R-squared", "score": r2})

    # Mean Absolute Percentage Error (MAPE)
    print("Calculating MAPE...")
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
    results.append({"model_name": model_name, "metric": "MAPE", "score": mape})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Ensure the output directory exists
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"{EVAL_DIR}/{model_name}_evaluation_scores_{timestamp}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Export results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Evaluation scores saved to {output_csv}")

if __name__ == '__main__':
     # load the data
    print("Loading data...")
    X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()   
    
    metrics = [
                "complexity", # checked
                "sparsity", # checked
                "max_sensitivity", # checked
                "robustness", # checked
                "faithfulness_correlation", # checked
                # "monotonicity", # no working                
                # "localisation", # not working
                # "randomisation", # not working
                # "road" # not working
    ]  # Specify the metrics you want to calculate
    # metrics = ["localisation"]
    # Evaluate the LightGBM model

    print("Evaluating LightGBM model...")
    # Change this filename to the path where your model is saved
    lgb_model_filename = f"{MODEL_DIR}/trained_lgb_model_20250617_145741.txt"  # Path to the saved LightGBM model
    # if this takes too long change use_subset=True to use a smaller subset of the data for evaluation
    evaluate_lgb_model(lgb_model_filename, X_test_combined, y_test, metrics=metrics, use_subset=True) 
    