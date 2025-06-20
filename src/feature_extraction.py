from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import os
import joblib
import json


def feature_extraction(df, tfidf_vectorizer=None, pca=None, fit=False):
    df = df.copy()
    text_column = 'review_clean'
    df[text_column] = df[text_column].fillna("")

    if fit:
        # Fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=50)
        tfidf_features = tfidf_vectorizer.fit_transform(df[text_column])
    else:
        tfidf_features = tfidf_vectorizer.transform(df[text_column])

    # PCA on TF-IDF features (convert sparse matrix to dense)
    n_components = 10  # Adjust as needed
    if fit:
        print("PCA start")
        pca = PCA(n_components=n_components, random_state=42)
        tfidf_pca = pca.fit_transform(tfidf_features.toarray())
        print("PCA finished")

    else:
        tfidf_pca = pca.transform(tfidf_features.toarray())

    # Sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(lambda text: analyzer.polarity_scores(text)['compound']).values.reshape(-1, 1)

    # Combine features
    X_combined = np.hstack([tfidf_pca, sentiment_scores])

    return X_combined, tfidf_vectorizer, pca



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