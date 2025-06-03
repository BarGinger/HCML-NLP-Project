from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_tfidf_features(texts, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

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