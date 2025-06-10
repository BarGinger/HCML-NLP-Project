from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def create_tfidf_features(train_df, test_df, text_column, max_features):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df[text_column])
    X_test_tfidf = tfidf_vectorizer.transform(test_df[text_column])
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def get_word2vec_features(texts, w2v_model, vector_size=300):
    features = []
    for doc in texts:
        tokens = doc.split()
        # average the word vectors for each document
        vectors = [w2v_model[w] for w in tokens if w in w2v_model]
        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features)


def add_sentiment_features(df, text_column):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].fillna("").apply(analyzer.polarity_scores)

    sentiment_df = pd.json_normalize(sentiment_scores)
    # TODO: (ALI) yadet bashe ke rename kardan ro dorost koni
    sentiment_df.rename(columns={
        'neg': 'sentiment_neg',
        'neu': 'sentiment_neu',
        'pos': 'sentiment_pos',
        'compound': 'sentiment_compound'
    }, inplace=True)

    # join with org df
    df = df.join(sentiment_df.set_index(df.index))
    return df


def create_pca_features(df_train, df_valid, text_column, num_components=15):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # ussse a larger vocab for PCA
    X_train_tfidf = vectorizer.fit_transform(df_train[text_column])
    pca = PCA(n_components=num_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_tfidf.toarray())
    feature_names = vectorizer.get_feature_names_out()
    for i, component in enumerate(pca.components_):
        top_word_indices = component.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_word_indices]
        print(f"Component {i}: {', '.join(top_words)}")
    X_valid_tfidf = vectorizer.transform(df_valid[text_column])
    X_valid_pca = pca.transform(X_valid_tfidf.toarray())
    return X_train_pca, X_valid_pca, pca, vectorizer
