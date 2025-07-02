"""
file: src/feature_extraction.py
This module contains functions for feature extraction from text data, 
including TF-IDF vectorization, SVD decomposition, sentiment analysis, 
and feature selection. 

It also includes functions to save and load features and feature names, 
as well as to compute and save Optuna optimization plots.
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import os
import joblib
import json
from optuna.visualization import plot_optimization_history, plot_param_importances
from plotly.io import write_image
import shap
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm 



def feature_extraction(df, tfidf_vectorizer=None, svd=None, fit=False, selected_feature_names=None):
    df = df.copy()
    text_column = 'review_clean'
    df[text_column] = df[text_column].fillna("")

    if fit:
        # Fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer() 
        tfidf_features = tfidf_vectorizer.fit_transform(df[text_column])
    else:
        tfidf_features = tfidf_vectorizer.transform(df[text_column])

    # PCA on TF-IDF features (convert sparse matrix to dense)
    n_components = 200  # Adjust as needed
    if fit:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_svd = svd.fit_transform(tfidf_features)
    else:
        tfidf_svd = svd.transform(tfidf_features)

    # Sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(lambda text: analyzer.polarity_scores(text)['compound']).values.reshape(-1, 1)

    # Add review length as a feature
    review_length = df[text_column].str.len().values.reshape(-1, 1)    
    
    # Combine features
    X_combined = np.hstack([tfidf_svd, sentiment_scores, review_length])

    # Create feature names for DataFrame
    feature_names = [f"SVD_{i}" for i in range(n_components)] + ["Sentiment", "ReviewLength"]
    X_combined_df = pd.DataFrame(X_combined, columns=feature_names, index=df.index)

    feature_cols = [col for col in X_combined_df.columns if col not in ['patient_id']]
    if fit:
        selector = VarianceThreshold(threshold=1e-5)
        X_selected = selector.fit_transform(X_combined_df[feature_cols])
        selected_feature_names = [feature_cols[i] for i in range(X_selected.shape[1])]
    else:
        # Use only the columns selected during training
        X_selected = X_combined_df[selected_feature_names].values

    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names, index=df.index)

    # Now add patient_id as a new column (if it exists)
    if 'patient_id' in df.columns:
        X_selected_df['patient_id'] = df['patient_id'].values
    

    return X_selected_df, tfidf_vectorizer, svd, selected_feature_names



 # Function to save features to CSV
def save_features_to_csv(features, file_path):
    pd.DataFrame(features).to_csv(file_path, index=False)

# Function to load features from CSV
def load_features_from_csv(file_path):
    return pd.read_csv(file_path)

# Save feature names to a JSON file
def save_feature_names(feature_names, file_path):
    with open(file_path, "w") as f:
        json.dump(feature_names, f)

# Load feature names from a JSON file
def load_feature_names(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    


def load_data():
    #Define the features (X) and the target variable (y)
    label = "rating"
    #text_column = 'review_clean'

    # Define file paths for saving the features
    data_folder = "Data"
    train_features_path = f"{data_folder}/X_train_combined.csv"
    val_features_path = f"{data_folder}/X_val_combined.csv"
    test_features_path = f"{data_folder}/X_test_combined.csv"
    tfidf_vectorizer_path = f"{data_folder}/tfidf_vectorizer.pkl"
    svd_path = f"{data_folder}/svd.pkl"
    # Define file paths for saving feature names
    feature_names_path = f"{data_folder}/feature_names.json"

    train_data = pd.read_csv(f'{data_folder}/drug_review_train_clean.csv')
    val_data = pd.read_csv(f'{data_folder}/drug_review_validation_clean.csv')
    test_data = pd.read_csv(f'{data_folder}/drug_review_test_clean.csv')

    X_train = train_data.drop(columns=[label]) 
    y_train = train_data[label]
    X_val = val_data.drop(columns=[label])
    y_val = val_data[label] 
    X_test = test_data.drop(columns=[label])
    y_test = test_data[label] 

    # Check if train features already exist
    if os.path.exists(train_features_path) and os.path.exists(tfidf_vectorizer_path) and os.path.exists(svd_path): #and os.path.exists(feature_names_path):
        print("Loading precomputed train features...")
        X_train_combined = load_features_from_csv(train_features_path)

        # Load tfidf_vectorizer, svd, and feature names
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        svd = joblib.load(svd_path)
        # feature_names = load_feature_names(feature_names_path)
    else:
        print("Computing train features...")
        X_train_combined, tfidf_vectorizer, svd, selected_feature_names = feature_extraction(X_train, fit=True)

        # Save train features, models, and feature names
        save_features_to_csv(X_train_combined, train_features_path)
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
        joblib.dump(svd, svd_path)

   
    # Save feature names
    feature_names = X_train_combined.columns.tolist()
    save_feature_names(feature_names, feature_names_path)

    X_train_combined = pd.DataFrame(X_train_combined, columns=feature_names)


    # Check if validation features already exist
    # Inside the `load_data` function
    if os.path.exists(val_features_path):
        print("Loading precomputed validation features...")
        X_val_combined = load_features_from_csv(val_features_path)
    else:
        print("Computing validation features...")
        X_val_combined, _, _ , _ = feature_extraction(X_val, tfidf_vectorizer, svd, fit=False, selected_feature_names=selected_feature_names)
        save_features_to_csv(X_val_combined, val_features_path)

    # Convert to DataFrame with feature names
    X_val_combined = pd.DataFrame(X_val_combined, columns=feature_names)

    if os.path.exists(test_features_path):
        print("Loading precomputed test features...")
        X_test_combined = load_features_from_csv(test_features_path)
    else:
        print("Computing test features...")
        X_test_combined, _, _ , _ = feature_extraction(X_test, tfidf_vectorizer, svd, fit=False, selected_feature_names=selected_feature_names)
        #X_test_combined['patient_id'] = X_test['patient_id'].values
        save_features_to_csv(X_test_combined, test_features_path)

    # Convert to DataFrame with feature names
    X_test_combined = pd.DataFrame(X_test_combined, columns=feature_names)
    
    # for df in [X_train_combined, X_val_combined, X_test_combined]:
    #     df.columns.values[-1] = 'patient_id'
    #     df.columns.values[-2] = 'Sentiment'

    return X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd


def save_optuna_plots(study, output_dir, model_name, dtick=5, width=1600):
    """
    Save Optuna optimization history and parameter importances plots with model name in filenames.

    Args:
        study: Optuna study object.
        output_dir: Directory to save plots.
        model_name: String to include in plot filenames.
        dtick: Interval for x-axis ticks (default 5).
        width: Width of the plot (default 1600).
    """

    # Export trials to CSV
    trials_df = study.trials_dataframe()
    csv_path = f"{output_dir}/{model_name}_optuna_trials.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"Optuna trials exported to {csv_path}")

    # Optimization history plot
    optimization_history_plot = plot_optimization_history(study)
    optimization_history_plot.update_layout(
        width=width,
        xaxis=dict(
            tick0=0,
            dtick=dtick,
            title="Trial",
            tickangle=45
        ),
        yaxis_title="MAE"
    )
    plot_path = f"{output_dir}/{model_name}_optuna_optimization_history.png"
    write_image(optimization_history_plot, plot_path)
    print(f"Optuna optimization history plot saved to {plot_path}")

    # Parameter importances plot
    param_importances_plot = plot_param_importances(study)
    plot_path = f"{output_dir}/{model_name}_optuna_param_importances.png"
    write_image(param_importances_plot, plot_path)
    print(f"Optuna parameter importances plot saved to {plot_path}")


def explain_and_plot_top_features(
    model,
    X_test,
    y_test,
    top_n=10,
    patient_ids=None,
    output_dir="Output",
    custom_labels=None,
    model_type="lgb",
    model_name="model"
):
    """
    Print top N features with importances, map to custom labels, and plot SHAP explanations for selected patients.
    Supports both LightGBM and Lasso models.

    Parameters:
        model: Trained model (LightGBM or Lasso).
        X_test: Test features (DataFrame).
        y_test: Test labels (Series).
        top_n: Number of top features to show.
        patient_ids: List of patient IDs to explain.
        output_dir: Directory to save plots.
        custom_labels: Dict mapping feature names to custom labels.
        model_type: "lgb" or "lasso".
        model_name: Name of the model for saving plots.
    """
    if custom_labels is None:
        custom_labels = {
            "svd_1": "Birth control side effect",
            "svd_15": "General effectiveness",
            "svd_11": "Impact on daily life",
            "svd_0": "Treatment timeline",
            "svd_21": "Lifestyle interactions (cravings/withdrawal)",
            "svd_7": "Subjective experience/mood",
            "svd_3": "Hormonal issues",
            "svd_28": "Intense emotional reactions",
            "svd_24": "Dose adjustments",
            "sentiment": "General sentiment"
        }

    def get_feature_label(col):
        return custom_labels.get(str(col).lower(), col)
    
    X_test_features = X_test.drop(columns=['patient_id'])

    # Get importances and top features
    if model_type == "lgb":
        importances = model.feature_importances_
        feature_names = X_test_features.columns
        shap_X = X_test_features  # OK to use subset for tree models
    elif model_type == "lasso":
        importances = np.abs(model.coef_)
        feature_names = X_test_features.columns if hasattr(X_test_features, "columns") else [f"f{i}" for i in range(X_test_features.shape[1])]
        shap_X = X_test_features  # Must use all features for linear models!
    else:
        raise ValueError("model_type must be 'lgb' or 'lasso'")

    
    
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_columns = [feature_names[i] for i in top_idx]

    print(f"\nTop {top_n} features by importance:")
    for col, importance in zip(top_columns, importances[top_idx]):
        print(f"{get_feature_label(col)}: Importance {importance}")

    print("\nTop feature labels:")
    for col in top_columns:
        print(f"{col}: {get_feature_label(col)}")

    # SHAP explanations (only for models supported by SHAP)
    if model_type == "lgb":
        explainer = shap.TreeExplainer(model)
        batch_size = 128  # You can adjust this for your memory/CPU
        if len(model.feature_importances_) == X_test_features.shape[1]:
            shap_values = []
            for start in tqdm(range(0, X_test_features.shape[0], batch_size), desc="Computing SHAP values", unit="batch"):
                end = min(start + batch_size, X_test_features.shape[0])
                batch = X_test_features.iloc[start:end]
                shap_values_batch = explainer.shap_values(batch)
                shap_values.extend(shap_values_batch)
            shap_values = np.array(shap_values)
        else:
            shap_values = []
            for start in tqdm(range(0, X_test_features[top_columns].shape[0], batch_size), desc="Computing SHAP values", unit="batch"):
                end = min(start + batch_size, X_test_features[top_columns].shape[0])
                batch = X_test_features[top_columns].iloc[start:end]
                shap_values_batch = explainer.shap_values(batch)
                shap_values.extend(shap_values_batch)
            shap_values = np.array(shap_values)
    elif model_type == "lasso":
        explainer = shap.Explainer(model, shap_X)  # Use all features!
        shap_values = explainer(shap_X)
    else:
        return

    if patient_ids is not None:
        # Wrap patient_ids with tqdm for a progress bar
        for pid in tqdm(patient_ids, desc="Generating SHAP explanations", unit="patient"):
            if "patient_id" not in X_test.columns:
                print("No patient_id column in X_test.")
                continue
            instance_idx = X_test.index[X_test["patient_id"] == pid].tolist()
            if not instance_idx:
                print(f"Patient ID {pid} not found in test set.")
                continue
            instance_idx = instance_idx[0]
            true_label = y_test.loc[instance_idx]
            instance_df_full = X_test_features.iloc[[instance_idx]]
            pred_label_temp = model.predict(instance_df_full)[0]
            pred_label = int(round(pred_label_temp))
            print("**********************************************************************************************************")
            print(f"\nPatient ID: {pid}, Predicted rating: {pred_label}", f"True rating: {true_label}\n")

            if model_type == "lgb":
                shap_values_instance_full = shap_values[instance_idx]
                expected_value = explainer.expected_value
                top_indices = [list(X_test_features.columns).index(col) for col in top_columns]
                shap_values_instance = shap_values_instance_full[top_indices]
            elif model_type == "lasso":
                shap_values_instance_full = shap_values.values[instance_idx]
                expected_value = shap_values.base_values[instance_idx]
                top_indices = [list(X_test_features.columns).index(col) for col in top_columns]
                shap_values_instance = shap_values_instance_full[top_indices]
            else:
                return

            fig = shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_values_instance,
                feature_names=[get_feature_label(col) for col in top_columns],
                show=False,
            )
            plt.title(f"Patient ID: {pid} | True label: {true_label} | Predicted: {pred_label:.2f}")
            plt.tight_layout()
            plt.subplots_adjust(left=0.35)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/shap_waterfall_{model_name}_patient_{pid}.png", bbox_inches='tight')
            plt.close(fig)