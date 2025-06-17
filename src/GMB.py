
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_extraction import feature_extraction
import numpy as np
import shap
import os
import joblib
import json
import matplotlib.pyplot as plt

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
    feature_names = [f"SVD_{i}" for i in range(X_train_combined.shape[1])]
    save_feature_names(feature_names, feature_names_path)

    X_train_combined = pd.DataFrame(X_train_combined, columns=feature_names)


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
        #X_test_combined['patient_id'] = X_test['patient_id'].values
        save_features_to_csv(X_test_combined, test_features_path)

    # Convert to DataFrame with feature names
    X_test_combined = pd.DataFrame(X_test_combined, columns=feature_names)
    
    for df in [X_train_combined, X_val_combined, X_test_combined]:
        df.columns.values[-1] = 'patient_id'
        df.columns.values[-2] = 'Sentiment'

    return X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd

X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()

# Train LightGBM
def apply_lgb(X_train, y_train, X_val, y_val, X_test, y_test, 
              num_leaves=47, max_depth=9, learning_rate=0.0998, n_estimators=199):
    lgb_model = lgb.LGBMRegressor(
        num_leaves=num_leaves, 
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        n_estimators=n_estimators
    )

    lgb_model.fit(X_train, y_train)
    y_val_pred_temp = lgb_model.predict(X_val)
    y_test_pred_temp = lgb_model.predict(X_test)

    # Round predictions to nearest integer
    y_val_pred = np.round(y_val_pred_temp).astype(int)
    y_test_pred = np.round(y_test_pred_temp).astype(int)

    mae_val = mean_absolute_error(y_val, y_val_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return mae_val, mse_val, mae_test, mse_test, lgb_model


X_train_features = X_train_combined.drop(columns=['patient_id'])
X_val_features = X_val_combined.drop(columns=['patient_id'])
X_test_features = X_test_combined.drop(columns=['patient_id'])

# First fit lgb model to all components:
# For all features
mae_val, mse_val, mae_test, mse_test, lgb_model = apply_lgb(
    X_train_features, y_train, X_val_features, y_val, X_test_features, y_test
)

# 2. Get feature importances and top 10 indices
importances = lgb_model.feature_importances_
top_10_idx = np.argsort(importances)[-10:][::-1]  # Indices of 10 most important features
top_10_columns = X_train_combined.columns[top_10_idx]

# 3. Reduce feature matrices to top 10 features
X_train_top10 = X_train_features[top_10_columns]
X_val_top10 = X_val_features[top_10_columns]
X_test_top10 = X_test_features[top_10_columns]

# For top 10 features fit the model again:
mae_val_top10, mse_val_top10, mae_test_top10, mse_test_top10, lgb_model_top10 = apply_lgb(
    X_train_top10, y_train, X_val_top10, y_val, X_test_top10, y_test
)

#results:
print(f"Validation MAE: {mae_val_top10:.4f}")
print(f"Validation MSE: {mse_val_top10:.4f}")
print(f"Test MAE: {mae_test_top10:.4f}")
print(f"Test MSE: {mse_test_top10:.4f}")

# Mapping from feature name to your custom label (all lowercase for robust lookup)
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
    return custom_labels.get(col.lower(), col)

for col, importance in zip(top_10_columns, importances[top_10_idx]):
    print(f"{get_feature_label(col)}: Importance {importance}")

N = 10  # Number of top words to show

for col in top_10_columns:
    label = get_feature_label(col)
    print(f"{col}: {label}")

# Instance-level explanations with SHAP
explainer = shap.TreeExplainer(lgb_model_top10)
shap_values = explainer.shap_values(X_test_top10)

# List of patient_ids to evaluate
patient_ids = [177996, 134603, 70362, 6760]

for pid in patient_ids:
    instance_idx = X_test_combined.index[X_test_combined['patient_id'] == pid].tolist()
    if not instance_idx:
        print(f"Patient ID {pid} not found in test set.")
        continue
    instance_idx = instance_idx[0]

    true_label = y_test.loc[instance_idx]
    instance_df = X_test_top10.iloc[[instance_idx]]
    pred_label_temp = lgb_model_top10.predict(instance_df)[0]
    pred_label = int(round(pred_label_temp))
    print(f"Predicted rating: {pred_label}")

    shap_values_instance = shap_values[instance_idx]

    # Plot waterfall
    fig = shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values_instance,
        feature_names=[get_feature_label(col) for col in top_10_columns],
        show=False  # Prevents immediate display so we can set the title
    )
    plt.title(f"True label: {true_label} | Predicted: {pred_label:.2f}")
    plt.tight_layout()
    plt.subplots_adjust(left=0.35) 
    plt.show()
