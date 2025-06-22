
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from feature_extraction import load_data, save_optuna_plots, explain_and_plot_top_features
import numpy as np
import json
import shap
import os
import joblib
import json
import kaleido
import matplotlib.pyplot as plt
import matplotlib
import optuna
from tqdm import tqdm
import warnings

# Define the directory path
output_dir = "Output"

# Define the directory path
models_dir = "Models"

# Check if the directory exists
if not os.path.exists(models_dir):
    # Create the directory if it doesn't exist
    os.makedirs(models_dir)
    print(f"Directory '{models_dir}' created.")
else:
    print(f"Directory '{models_dir}' already exists.")

# Check if the directory exists
if not os.path.exists(output_dir):
    # Create the directory if it doesn't exist
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")

class GBM:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
def save_model(model, model_name="trained_lgb_model"):
    # Save the trained LightGBM model to a file   
    model_filename = f"{models_dir}/{model_name}.txt"
    # Save the trained LGBMRegressor model
    model.booster_.save_model(model_filename)

    # Save model parameters to a JSON file
    params = model.get_params()
    params_filename = f"{models_dir}/{model_name}_params.json"
    with open(params_filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Model parameters saved to {params_filename}")
    

X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()

X_train_features = X_train_combined.drop(columns=['patient_id'])
X_val_features = X_val_combined.drop(columns=['patient_id'])
X_test_features = X_test_combined.drop(columns=['patient_id'])


def objective(trial, X_train, y_train, X_val, y_val):
    # Suppress warnings and LightGBM output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb_model = lgb.LGBMRegressor(
            num_leaves=trial.suggest_int('num_leaves', 20, 256),
            max_depth=trial.suggest_int('max_depth', 3, 16),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
            min_child_samples=trial.suggest_int('min_child_samples', 3, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            random_state=42,
            force_col_wise=True,
            verbose=-1  # Suppress LightGBM output
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        y_val_pred_temp = lgb_model.predict(X_val)
        y_val_pred = np.round(y_val_pred_temp).astype(int)
        # mae_val = mean_absolute_error(y_val, y_val_pred)
        kappa = cohen_kappa_score(y_val, y_val_pred, weights='quadratic')
        return 1 - kappa  # minimize 1-kappa




def tune_lgb_with_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    study = optuna.create_study(direction='minimize')
    # Use tqdm for progress bar
    with tqdm(total=n_trials, desc="Optuna Trials", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False  # disables Optuna's own progress bar
        )
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    best_params = trial.params
    return study, best_params

# Train LightGBM with the best hyperparameters
def apply_lgb_with_params(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    lgb_model = lgb.LGBMRegressor(**best_params, random_state=42)

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

# Use Optuna to tune hyperparameters
n_trials = 100
study, best_params = tune_lgb_with_optuna(X_train_features, y_train, X_val_features, y_val, n_trials=n_trials)


# Plot optimization history
save_optuna_plots(study, output_dir, "LightGBM")


# First fit lgb model to all components:
# For all features
# Apply lgb with best params
mae_val, mse_val, mae_test, mse_test, lgb_model = apply_lgb_with_params(
    X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, best_params
)

# 2. Get feature importances and top num_top_features indices
num_top_features = 15  # Set the number of top features you want to select
importances = lgb_model.feature_importances_
top_num_top_features_idx = np.argsort(importances)[-num_top_features:][::-1]  # Indices of num_top_features most important features
top_num_top_features_columns = list(X_train_features.columns[top_num_top_features_idx])

# Ensure "Sentiment" and "ReviewLength" are included
for col in ["Sentiment", "ReviewLength"]:
    if col in X_train_features.columns and col not in top_num_top_features_columns:
        top_num_top_features_columns.append(col)

# After selecting top_num_top_features_columns
top_features_list = list(top_num_top_features_columns)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
with open(f"{models_dir}/top_{num_top_features}_features_{timestamp}.json", "w") as f:
    json.dump(top_features_list, f)

# 3. Reduce feature matrices to top num_top_features features
X_train_top = X_train_features[top_num_top_features_columns]
X_val_top = X_val_features[top_num_top_features_columns]
X_test_top = X_test_features[top_num_top_features_columns]
X_test_combined_top = X_test_combined[top_num_top_features_columns].copy()
X_test_combined_top['patient_id'] = X_test_combined['patient_id']

# For top num_top_features features fit the model again:
study_top, best_params_top = tune_lgb_with_optuna(X_train_top, y_train, X_val_top, y_val, n_trials=n_trials)
save_optuna_plots(study_top, output_dir, "LightGBMtop")
mae_val_top, mse_val_top, mae_test_top, mse_test_top, lgb_model_top = apply_lgb_with_params(
    X_train_top, y_train, X_val_top, y_val, X_test_top, y_test, best_params_top
)

#results:
print(f"Validation MAE: {mae_val_top:.4f}")
print(f"Validation MSE: {mse_val_top:.4f}")
print(f"Test MAE: {mae_test_top:.4f}")
print(f"Test MSE: {mse_test_top:.4f}")

# save the model
model_name = f"lgb_model_{timestamp}"
save_model(lgb_model, model_name=model_name)
explain_and_plot_top_features(
    lgb_model,
    X_test_combined,
    y_test,
    top_n=10,
    # feature_names=list(X_test_top.columns),
    patient_ids=[177996, 134603, 70362, 6760],
    output_dir=output_dir,
    model_type="lgb",
    model_name=model_name
)

model_name = f"lgb_model_top_{num_top_features}_{timestamp}"
save_model(lgb_model_top, model_name=model_name)
explain_and_plot_top_features(
    lgb_model_top,
    X_test_combined_top,
    y_test,
    top_n=10,
    # feature_names=list(X_test_top.columns),
    patient_ids=[177996, 134603, 70362, 6760],
    output_dir=output_dir,
    model_type="lgb",
    model_name=model_name
)
