
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import src.feature_extraction as feature_extraction # Assuming this is your custom module for feature extraction
from gensim.models import Word2Vec
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA

class GBM:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

train_data = pd.read_csv('Data/drug_review_train_clean.csv')
val_data = pd.read_csv('Data/drug_review_validation_clean.csv')
test_data = pd.read_csv('Data/drug_review_test_clean.csv')

#Define the features (X) and the target variable (y)
label = "rating"
text_column = 'review_clean'

X_train = train_data.drop(columns=[label])  # Drop the target column
y_train = train_data[label]  # Target column
X_val = val_data.drop(columns=[label])  # Drop the target column
y_val = val_data[label]  # Target columnX_train = train_data.drop(columns=[label])  # Drop the target column
X_test = test_data.drop(columns=[label])  # Drop the target column
y_test = test_data[label]  # Target column

# Fill missing values
X_train[text_column] = X_train[text_column].fillna("")
X_val[text_column] = X_val[text_column].fillna("")
X_test[text_column] = X_test[text_column].fillna("")

def preprocess(df, tfidf_vectorizer=None, w2v_model=None, tfidf_weights=None, pca=None, fit=False):
    df = df.copy()
    df[text_column] = df[text_column].fillna("")
    df['tokens'] = df[text_column].apply(lambda x: x.split())
    if fit:
        # Fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(df[text_column])
        tfidf_weights = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
        # Fit Word2Vec
        w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=2, workers=4)
    vector_size = w2v_model.vector_size

    def tfidf_weighted_w2v(tokens):
        vec = np.zeros(vector_size)
        weight_sum = 0
        for word in tokens:
            if word in w2v_model.wv and word in tfidf_weights:
                weight = tfidf_weights[word]
                vec += w2v_model.wv[word] * weight
                weight_sum += weight
        return vec / weight_sum if weight_sum > 0 else vec

    # Document vectors
    df_tfidf_w2v = np.vstack(df['tokens'].apply(tfidf_weighted_w2v))

    # PCA on tfidf_w2v
    n_components = 32  # You can adjust this number
    if fit:
        pca = PCA(n_components=n_components, random_state=42)
        tfidf_w2v_pca = pca.fit_transform(df_tfidf_w2v)
    else:
        tfidf_w2v_pca = pca.transform(df_tfidf_w2v)

    # Sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(lambda text: analyzer.polarity_scores(text)['compound']).values.reshape(-1, 1)

    # Combine features
    X_combined = np.hstack([df_tfidf_w2v, sentiment_scores])
    return X_combined, tfidf_vectorizer, w2v_model, tfidf_weights, pca

# Fit on train, transform all
X_train_combined, tfidf_vectorizer, w2v_model, tfidf_weights, pca = preprocess(X_train, fit=True)
X_val_combined, _, _, _, _ = preprocess(X_val, tfidf_vectorizer, w2v_model, tfidf_weights, pca, fit =False)
X_test_combined, _, _, _, _ = preprocess(X_test, tfidf_vectorizer, w2v_model, tfidf_weights, pca, fit =False)

y_train = train_data[label]
y_val = val_data[label]
y_test = test_data[label]

# Train LightGBM
def apply_lgb(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, random_state=42):
    lgb_model = lgb.LGBMRegressor(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, random_state=42)
    lgb_model.fit(X_train_combined, y_train)
    # Predict and evaluate
    y_val_pred = lgb_model.predict(X_val_combined)
    y_test_pred = lgb_model.predict(X_test_combined)

    mae_val = mean_absolute_error(y_val, y_val_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test =  mean_squared_error(y_test, y_test_pred)

    print(f"Validation MAE: {mae_val:.4f}")
    print(f"Validation MSE: {mse_val:.4f}")
    print(f"Test MAE: {mae_test:.4f}")
    print(f"Test MSE: {mse_test:.4f}")
    return lgb_model


# Feature importances
#importances = lgb_model.feature_importances_
#print("Top 10 Feature Importances:")
#for idx in np.argsort(importances)[::-1][:10]:
#    print(f"Feature {idx}: Importance {importances[idx]}")
