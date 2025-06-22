import pandas as pd
from check_the_data import run_lasso_with_optuna, evaluate_lasso_model, analyze_lasso_coefficients, print_top_words_per_svd_component,print_top_features_by_absolute_weight, cohen_kappa_score
from check_the_data import print_most_important_features
import os
import joblib
import numpy as np
from feature_extraction import load_data, explain_and_plot_top_features
import json

print("Hello World")


X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()
X_train_features = X_train_combined.drop(columns=['patient_id'])
X_val_features = X_val_combined.drop(columns=['patient_id'])
X_test_features = X_test_combined.drop(columns=['patient_id'])

# Define the directory path for models
models_dir = "Models"

# Check if the directory exists
if not os.path.exists(models_dir):
    # Create the directory if it doesn't exist
    os.makedirs(models_dir)
    print(f"Directory '{models_dir}' created.")
else:
    print(f"Directory '{models_dir}' already exists.")


# Define the directory path
output_dir = "Output"

# Check if the directory exists
if not os.path.exists(output_dir):
    # Create the directory if it doesn't exist
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")


n_trials = 150  # Number of trials for Optuna optimization

final_model, y_pred = run_lasso_with_optuna(X_train_features, y_train,
                                                              X_val_features, y_val, X_test_features,
                                                                y_test, tfidf_vectorizer, svd, n_trials=n_trials)

# Save the trained Lasso model to a file
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
model_name = "lasso_model"
model_filename = f"{models_dir}/{model_name}_{timestamp}.joblib"
joblib.dump(final_model, model_filename)
print(f"Lasso model saved to {model_filename}")


evaluate_lasso_model(y_test, y_pred)

# 1. Get absolute coefficients as importances
importances = np.abs(final_model.coef_)

# 2. Get indices of top 10 features
num_of_top_features = 15
top_idx = np.argsort(importances)[-num_of_top_features:][::-1]  # Indices of 10 most important features

# 3. Get column names (assuming X_train_combined is a DataFrame)
top_columns = X_train_combined.columns[top_idx]

for col in ["Sentiment", "ReviewLength"]:
    if col in X_train_features.columns and col not in top_columns:
        top_columns = list(top_columns) + [col]

# 4. Reduce feature matrices to top 10 features
X_train_top = X_train_features[top_columns]
X_val_top = X_val_features[top_columns]
X_test_top = X_test_features[top_columns]

top_features_list = list(top_columns)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
with open(f"{models_dir}/{model_name}_top_{num_of_top_features}_features_{timestamp}.json", "w") as f:
    json.dump(top_features_list, f)

# explain the full model with top 10 features
explain_and_plot_top_features(
    final_model,
    X_test_combined,  # DataFrame with patient_id column
    y_test,
    top_n=10,
    patient_ids=[177996, 134603, 70362, 6760],
    output_dir=output_dir,
    model_type="lasso",
    model_name=f"lasso_full_{num_of_top_features}_{timestamp}"
)

# Run Lasso with Optuna on the top features
final_model_top, y_pred_top = run_lasso_with_optuna(
    X_train_top, y_train,
    X_val_top, y_val,
    X_test_top, y_test,
    tfidf_vectorizer, svd, n_trials=n_trials
)

model_filename_top = f"{models_dir}/{model_name}_top{num_of_top_features}_{timestamp}.joblib"
joblib.dump(final_model_top, model_filename_top)
print(f"Lasso top-15 model saved to {model_filename_top}")

explain_and_plot_top_features(
    final_model_top,
    X_test_combined[top_columns + ['patient_id']],  # keep patient_id for SHAP
    y_test,
    top_n=15,
    patient_ids=[177996, 134603, 70362, 6760],
    output_dir=output_dir,
    model_type="lasso",
    model_name=f"lasso_top_{num_of_top_features}_{timestamp}"
)


print("Goodbye World")



