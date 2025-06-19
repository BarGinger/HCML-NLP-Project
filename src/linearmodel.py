
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD



def embedding_function(train_path, test_path, n_components=100):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop unneeded columns
    train_df.drop(['Unnamed: 0', 'date', 'usefulCount', 'review_length'], axis=1, inplace=True)
    test_df.drop(['Unnamed: 0', 'date', 'usefulCount', 'review_length'], axis=1, inplace=True)

    # Feature extraction (BoW)
    vectorizer = CountVectorizer(max_features=60000)
    X_train_bow = vectorizer.fit_transform(train_df['review_clean'])
    X_test_bow = vectorizer.transform(test_df['review_clean'])

    # Apply PCA (TruncatedSVD) on BoW features
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_bow)
    X_test_reduced = svd.transform(X_test_bow)

    # Get sentiment scores
    sentiment_scores_train = sentiment_scores_column(train_df['review_clean']).reshape(-1, 1)
    sentiment_scores_test = sentiment_scores_column(test_df['review_clean']).reshape(-1, 1)

    # Combine reduced BoW features with sentiment
    combined1 = np.hstack([X_train_reduced, sentiment_scores_train])
    combined2 = np.hstack([X_test_reduced, sentiment_scores_test])

    y_train = train_df['rating'].values
    y_test = test_df['rating'].values
    vocab = vectorizer.get_feature_names_out()

    return combined1, y_train, combined2, y_test, vocab, svd

def sentiment_scores_column(series):
    analyzer = SentimentIntensityAnalyzer()
    return series.apply(lambda text: analyzer.polarity_scores(text)['compound']).values


def train_lasso_model(train_path, test_path, alpha=0.01, max_iter=1000):
    X_train_bow, y_train, X_test_bow, y_test, vocab,svd = embedding_function(train_path, test_path)
    # Train Lasso
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_train_bow, y_train)

    return lasso, vocab, X_test_bow, y_test




def print_lasso_results(model, vocab, X_test, y_test, top_n=20):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE with LASSO: {mse:.4f}")

    # Show top non-zero weighted features
    nonzero_indices = np.where(model.coef_ != 0)[0]
    print(f"Number of non-zero features: {len(nonzero_indices)}")
    top_features = sorted(zip(nonzero_indices, model.coef_[nonzero_indices]), key=lambda x: -abs(x[1]))[:top_n]
   
    print(f"\nTop {top_n} LASSO features:")
    for idx, weight in top_features:
        if idx < len(vocab):
            print(f"PCA Component {idx} (from BoW): {weight:.4f}")
        else:
            print(f"Sentiment Feature: {weight:.4f}")

def print_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
     # Compute and print accuracy (after rounding predictions to nearest integer)
    y_pred_rounded = np.clip(np.round(y_pred), 1, 10).astype(int)  # ratings are likely between 1 and 10
    y_test_int = y_test.astype(int)
    acc = accuracy_score(y_test_int, y_pred_rounded)
    print(f"Test Accuracy (rounded predictions): {acc * 100:.2f}%")

