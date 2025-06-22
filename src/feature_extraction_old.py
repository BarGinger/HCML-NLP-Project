from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
#from gensim.models import Word2Vec

def feature_extraction(df, tfidf_vectorizer=None, svd=None, fit=False):
    df = df.copy()
    text_column = 'review_clean'
    df[text_column] = df[text_column].fillna("")
    #df['tokens'] = df[text_column].apply(lambda x: x.split())

    if fit:
        # Fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_features = tfidf_vectorizer.fit_transform(df[text_column])
    else:
        tfidf_features = tfidf_vectorizer.transform(df[text_column])

    #PCA on TF-IDF features
    n_components = 100  # Adjust as needed

    if fit:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_svd = svd.fit_transform(tfidf_features)
    else:
        tfidf_svd = svd.transform(tfidf_features)

    # Sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(lambda text: analyzer.polarity_scores(text)['compound']).values.reshape(-1, 1)

    # Combine features
    X_combined = np.hstack([tfidf_svd, sentiment_scores])

    # Create feature names for DataFrame
    feature_names = [f"SVD_{i}" for i in range(n_components)] + ["Sentiment"]
    X_combined_df = pd.DataFrame(X_combined, columns=feature_names, index=df.index)

    # Now add patient_id as a new column (if it exists)
    if 'patient_id' in df.columns:
        X_combined_df['patient_id'] = df['patient_id'].values

    return X_combined_df, tfidf_vectorizer, svd


