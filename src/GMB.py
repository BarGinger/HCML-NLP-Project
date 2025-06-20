
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_extraction import load_data
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