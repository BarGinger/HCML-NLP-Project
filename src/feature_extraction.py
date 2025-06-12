from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

def feature_extraction(df, tfidf_vectorizer=None, tfidf_weights=None, pca=None, fit=False):
    df = df.copy()
    text_column = 'review_clean'
    df[text_column] = df[text_column].fillna("")
    #df['tokens'] = df[text_column].apply(lambda x: x.split())

    if fit:
        # Fit TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_features = tfidf_vectorizer.fit_transform(df[text_column])
        #tfidf_vectorizer.fit(df[text_column])
        #tfidf_weights = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
        # Fit Word2Vec
        #w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=2, workers=4)
    else:
        tfidf_features = tfidf_vectorizer.transform(df[text_column])    #vector_size = w2v_model.vector_size

    '''	
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
        '''
    #PCA on TF-IDF features
    n_components = 32  # Adjust as needed
    if fit:
        pca = PCA(n_components=n_components, random_state=42)
        tfidf_pca = pca.fit_transform(tfidf_features.toarray())
    else:
        tfidf_pca = pca.transform(tfidf_features.toarray())

    # Sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(lambda text: analyzer.polarity_scores(text)['compound']).values.reshape(-1, 1)

    # Combine features
    X_combined = np.hstack([tfidf_pca, sentiment_scores])
    return X_combined, tfidf_vectorizer, tfidf_weights, pca


