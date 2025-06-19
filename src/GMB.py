
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_extraction import feature_extraction
import numpy as np
import shap
import os
import joblib
import json

class GBM:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    

 # Function to save features to CSV
def save_features_to_csv(features, file_path):
    pd.DataFrame(features).to_csv(file_path, index=False)

# Function to load features from CSV
def load_features_from_csv(file_path):
    return pd.read_csv(file_path).values

# Save feature names to a JSON file
def save_feature_names(feature_names, file_path):
    with open(file_path, "w") as f:
        json.dump(feature_names, f)

# Load feature names from a JSON file
def load_feature_names(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    

def load_data():
    train_data = pd.read_csv('Data/drug_review_train_clean.csv')
    val_data = pd.read_csv('Data/drug_review_validation_clean.csv')
    test_data = pd.read_csv('Data/drug_review_test_clean.csv')

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
    feature_names_path = "Data/feature_names.json"


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
    if os.path.exists(train_features_path) and os.path.exists(tfidf_vectorizer_path) and os.path.exists(svd_path) and os.path.exists(feature_names_path):
        print("Loading precomputed train features...")
        X_train_combined = load_features_from_csv(train_features_path)

        # Load tfidf_vectorizer, svd, and feature names
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        svd = joblib.load(svd_path)
        feature_names = load_feature_names(feature_names_path)
    else:
        print("Computing train features...")
        X_train_combined, tfidf_vectorizer, svd = feature_extraction(X_train, fit=True)

        # Save train features, models, and feature names
        save_features_to_csv(X_train_combined, train_features_path)
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
        joblib.dump(svd, svd_path)

    # Save feature names
    feature_names = [f"feature_{i}" for i in range(X_train_combined.shape[1])]
    save_feature_names(feature_names, feature_names_path)

    # Check if validation features already exist
    # Inside the `load_data` function
    if os.path.exists(val_features_path):
        print("Loading precomputed validation features...")
        X_val_combined = load_features_from_csv(val_features_path)
    else:
        print("Computing validation features...")
        X_val_combined, _, _ = feature_extraction(X_val, tfidf_vectorizer, svd, fit=False)
        save_features_to_csv(X_val_combined, val_features_path)

    # Convert to DataFrame with feature names
    X_val_combined = pd.DataFrame(X_val_combined, columns=feature_names)

    if os.path.exists(test_features_path):
        print("Loading precomputed test features...")
        X_test_combined = load_features_from_csv(test_features_path)
    else:
        print("Computing test features...")
        X_test_combined, _, _ = feature_extraction(X_test, tfidf_vectorizer, svd, fit=False)
        save_features_to_csv(X_test_combined, test_features_path)

    # Convert to DataFrame with feature names
    X_test_combined = pd.DataFrame(X_test_combined, columns=feature_names)

    return X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd

# Train LightGBM
def apply_lgb(num_leaves=47, max_depth=9, learning_rate=0.0998, n_estimators=199):
    lgb_model = lgb.LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, force_col_wise=True, random_state=42)
    lgb_model.fit(X_train_combined, y_train)
    # Predict and evaluate
    y_val_pred = lgb_model.predict(X_val_combined)
    y_test_pred = lgb_model.predict(X_test_combined)

    mae_val = mean_absolute_error(y_val, y_val_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test =  mean_squared_error(y_test, y_test_pred)

    return mae_val, mse_val, mae_test, mse_test, lgb_model


def save_model(model):
    import json

    # Define the directory path
    models_dir = "Models"

    # Check if the directory exists
    if not os.path.exists(models_dir):
        # Create the directory if it doesn't exist
        os.makedirs(models_dir)
        print(f"Directory '{models_dir}' created.")
    else:
        print(f"Directory '{models_dir}' already exists.")

    # Save the trained LightGBM model to a file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{models_dir}/trained_lgb_model_{timestamp}.txt"
    # Save the trained LGBMRegressor model
    model.booster_.save_model(model_filename)

    # Save model parameters to a JSON file
    params = model.get_params()
    params_filename = f"{models_dir}/lgb_model_params_{timestamp}.json"
    with open(params_filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Model parameters saved to {params_filename}")



if __name__ == '__main__':
    # load the data
    X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()
    # train the model and evaluate
    mae_val, mse_val, mae_test, mse_test, lgb_model = apply_lgb(num_leaves=47, max_depth=9, learning_rate=0.0998, n_estimators=199)
    # print the results
    print("LightGBM model training and evaluation completed.")
    print(f"Validation MAE: {mae_val:.4f}")
    print(f"Validation MSE: {mse_val:.4f}")
    print(f"Test MAE: {mae_test:.4f}")
    print(f"Test MSE: {mse_test:.4f}")

    # save the model
    save_model(lgb_model)