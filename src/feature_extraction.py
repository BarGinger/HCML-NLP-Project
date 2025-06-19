from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def feature_extraction(df, tfidf_vectorizer=None, pca=None, fit=False):
    df = df.copy()
    text_column = 'review_clean'
    df[text_column] = df[text_column].fillna("")

    if fit:
        # Fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=20)
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