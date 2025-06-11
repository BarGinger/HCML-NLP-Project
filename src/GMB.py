
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_extraction import feature_extraction

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
X_train_combined, tfidf_vectorizer, w2v_model, tfidf_weights, pca = feature_extraction(X_train, fit=True)
X_val_combined, _, _, _, _ = feature_extraction(X_val, tfidf_vectorizer, w2v_model, tfidf_weights, pca, fit =False)
X_test_combined, _, _, _, _ = feature_extraction(X_test, tfidf_vectorizer, w2v_model, tfidf_weights, pca, fit =False)

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

    return mae_val, mse_val, mae_test, mse_test


mae_val, mse_val, mae_test, mse_test = apply_lgb(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)
print(f"Validation MAE: {mae_val:.4f}")
print(f"Validation MSE: {mse_val:.4f}")
print(f"Test MAE: {mae_test:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Feature importances
#importances = lgb_model.feature_importances_
#print("Top 10 Feature Importances:")
#for idx in np.argsort(importances)[::-1][:10]:
#    print(f"Feature {idx}: Importance {importances[idx]}")
