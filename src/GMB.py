
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_extraction import feature_extraction
import numpy as np
import shap

class GBM:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


train_data = pd.read_csv('Data/drug_review_train_clean.csv')
val_data = pd.read_csv('Data/drug_review_validation_clean.csv')
test_data = pd.read_csv('Data/drug_review_test_clean.csv')

#Define the features (X) and the target variable (y)
label = "rating"
#text_column = 'review_clean'

X_train = train_data.drop(columns=[label]) 
y_train = train_data[label]
X_val = val_data.drop(columns=[label])
y_val = val_data[label] 
X_test = test_data.drop(columns=[label])
y_test = test_data[label]  

# Fit on train, transform all
X_train_combined, tfidf_vectorizer, svd = feature_extraction(X_train, fit=True)
X_val_combined, _, _ = feature_extraction(X_val, tfidf_vectorizer, svd, fit =False)
X_test_combined, _, _ = feature_extraction(X_test, tfidf_vectorizer, svd, fit =False)

# Train LightGBM
def apply_lgb(num_leaves=47, max_depth=9, learning_rate=0.0998, n_estimators=199):
    lgb_model = lgb.LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
    lgb_model.fit(X_train_combined, y_train)
    # Predict and evaluate
    y_val_pred = lgb_model.predict(X_val_combined)
    y_test_pred = lgb_model.predict(X_test_combined)

    mae_val = mean_absolute_error(y_val, y_val_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test =  mean_squared_error(y_test, y_test_pred)

    return mae_val, mse_val, mae_test, mse_test, lgb_model


mae_val, mse_val, mae_test, mse_test, lgb_model = apply_lgb(num_leaves=47, max_depth=9, learning_rate=0.0998, n_estimators=199)
print(f"Validation MAE: {mae_val:.4f}")
print(f"Validation MSE: {mse_val:.4f}")
print(f"Test MAE: {mae_test:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Feature importances
importances = lgb_model.feature_importances_
feature_names = tfidf_vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    top_words = [feature_names[idx] for idx in comp.argsort()[-10:][::-1]]
    print(f"Component {i}: {', '.join(top_words)}")

# If you have n_components SVD features + 1 sentiment feature:
n_components = svd.n_components
for i, importance in enumerate(importances):
    if i < n_components:
        print(f"SVD Component {i}: Importance {importance}")
    else:
        print(f"Sentiment: Importance {importance}")

# Instance-level explanations with SHAP
import shap
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test_combined)

# Visualize explanation for the first test instance
instance_idx = 0  # Change this to look at other instances
print(f"True rating: {y_test.iloc[instance_idx]}")
print(f"Predicted rating: {lgb_model.predict(X_test_combined[instance_idx].reshape(1, -1))[0]:.2f}")

print("\nFeature contributions for this instance:")
for i, shap_val in enumerate(shap_values[instance_idx]):
    if i < n_components:
        print(f"SVD Component {i} (Top words: {', '.join([feature_names[idx] for idx in svd.components_[i].argsort()[-3:][::-1]])}): SHAP value {shap_val:.3f}")
    else:
        print(f"Sentiment: SHAP value {shap_val:.3f}")

# Optionally, show a SHAP force plot in the notebook or browser
#shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[instance_idx], feature_names=[f"SVD {i}" for i in range(n_components)] + ["Sentiment"])
shap.save_html("shap_force_plot.html", shap.force_plot(
    explainer.expected_value, shap_values[instance_idx],
    feature_names=[f"SVD {i}" for i in range(n_components)] + ["Sentiment"]
))
print("SHAP force plot saved as shap_force_plot.html")

# Find index of a high-score and a low-score instance in the test set
high_idx = y_test.idxmax()
low_idx = y_test.idxmin()

# Get their positions in the test set (iloc expects integer positions)
high_pos = y_test.index.get_loc(high_idx)
low_pos = y_test.index.get_loc(low_idx)

for instance_idx, label in zip([high_pos, low_pos], ["High score", "Low score"]):
    print(f"\n--- {label} instance ---")
    print(f"True rating: {y_test.iloc[instance_idx]}")
    print(f"Predicted rating: {lgb_model.predict(X_test_combined[instance_idx].reshape(1, -1))[0]:.2f}")

    print("\nFeature contributions for this instance:")
    for i, shap_val in enumerate(shap_values[instance_idx]):
        if i < n_components:
            top_words = [feature_names[idx] for idx in svd.components_[i].argsort()[-3:][::-1]]
            print(f"SVD Component {i} (Top words: {', '.join(top_words)}): SHAP value {shap_val:.3f}")
        else:
            print(f"Sentiment: SHAP value {shap_val:.3f}")

    # Save SHAP force plot for each instance
    shap.save_html(f"shap_force_plot_{label.replace(' ', '_').lower()}.html",
        shap.force_plot(
            explainer.expected_value,
            shap_values[instance_idx],
            feature_names=[f"SVD {i}" for i in range(n_components)] + ["Sentiment"]
        )
    )
    print(f"SHAP force plot saved as shap_force_plot_{label.replace(' ', '_').lower()}.html")