import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import optuna
from scipy.sparse import hstack

def feature_extraction(df, tfidf_vectorizer=None, fit=False):
    df = df.copy()
    text_column = 'review_clean'
    df[text_column] = df[text_column].fillna("")

    if fit:
        tfidf_vectorizer = TfidfVectorizer(max_features=20)
        tfidf_features = tfidf_vectorizer.fit_transform(df[text_column])
    else:
        tfidf_features = tfidf_vectorizer.transform(df[text_column])

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df[text_column].apply(lambda text: analyzer.polarity_scores(text)['compound']).values.reshape(-1, 1)

    return tfidf_features, sentiment_scores, tfidf_vectorizer

def run_lasso_model(df, target_column='rating', test_size=0.2, alpha=0.01):
    tfidf_features, sentiment_scores, tfidf_vectorizer = feature_extraction(df, fit=True)

    # Combine TF-IDF features (sparse matrix) and sentiment scores (dense)
    from scipy.sparse import hstack
    X_combined = hstack([tfidf_features, sentiment_scores])

    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=test_size, random_state=42)

    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train.toarray(), y_train)  # Lasso needs dense arrays

    y_pred = lasso.predict(X_test.toarray())
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

    return lasso, tfidf_vectorizer, X_test, y_test, y_pred

def print_top_words_per_svd_component(svd_model, tfidf_vectorizer, top_n=10, show_weights=True, show_positive_and_negative=False):
    terms = tfidf_vectorizer.get_feature_names_out()
    
    for i, comp in enumerate(svd_model.components_):
        print(f"\nComponent {i}:")

        if show_positive_and_negative:
            # Show top positive weights
            top_pos_indices = np.argsort(comp)[-top_n:][::-1]
            print("  Top Positive:")
            for idx in top_pos_indices:
                term = terms[idx]
                weight = comp[idx]
                if show_weights:
                    print(f"    {term}: {weight:.4f}")
                else:
                    print(f"    {term}")

            # Show top negative weights
            top_neg_indices = np.argsort(comp)[:top_n]
            print("  Top Negative:")
            for idx in top_neg_indices:
                term = terms[idx]
                weight = comp[idx]
                if show_weights:
                    print(f"    {term}: {weight:.4f}")
                else:
                    print(f"    {term}")
        else:
            # Only show top absolute weights
            top_indices = np.argsort(np.abs(comp))[-top_n:][::-1]
            print("  Top Words by Absolute Weight:")
            for idx in top_indices:
                term = terms[idx]
                weight = comp[idx]
                if show_weights:
                    print(f"    {term}: {weight:.4f}")
                else:
                    print(f"    {term}")

def print_top_features_by_absolute_weight(lasso_model, svd_model, tfidf_vectorizer, top_n=10):
    import numpy as np

    coef = lasso_model.coef_[:-1]  # exclude sentiment coefficient
    abs_coef = np.abs(coef)
    top_indices = np.argsort(abs_coef)[-top_n:][::-1]

    # Get original vocabulary
    terms = tfidf_vectorizer.get_feature_names_out()

    # Each component in SVD is a linear combination of the original TF-IDF features
    # We need to back-project each SVD component's contribution to estimate word influence
    svd_components = svd_model.components_

    # Project SVD-weighted coefficients back to original word space
    word_scores = np.dot(coef, svd_components)

    # Now get top contributing word indices and their values
    top_word_indices = np.argsort(np.abs(word_scores))[-top_n:][::-1]
    for idx in top_word_indices:
        print(f"{terms[idx]}: {word_scores[idx]:.4f}")

def print_most_important_features(lasso_model, feature_names, top_n=20):
    coefs = lasso_model.coef_

    print(f"Total number of features (including sentiment score): {len(feature_names)}\n")

    # Sort features by absolute coefficient value
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

    print(f"Top {top_n} most important features by absolute coefficient:\n")
    for _, row in coef_df.head(top_n).iterrows():
        sign = '+' if row['coefficient'] > 0 else '-'
        print(f"{row['feature']}: {sign}{abs(row['coefficient']):.4f}")


def evaluate_lasso_model(y_test, y_pred):
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

def analyze_lasso_coefficients(lasso_model, tfidf_vectorizer):
    feature_names = list(tfidf_vectorizer.get_feature_names_out()) + ['sentiment_score']
    coefs = lasso_model.coef_
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    coef_df = coef_df.sort_values(by='coefficient', ascending=False)
    print(coef_df.head(20))
    print("\nTop negative coefficients:")
    print(coef_df.tail(20))

def run_lasso_with_optuna(X_train, y_train, X_val, y_val, X_test, y_test, tfidf_vectorizer, svd,
                           target_column='rating', test_size=0.2, n_trials=30):
    
    # tfidf_features, sentiment_scores, tfidf_vectorizer = feature_extraction(df, fit=True)
    # X_combined = hstack([tfidf_features, sentiment_scores])
    # y = df[target_column].values

    # X_train_full, X_test, y_train_full, y_test = train_test_split(X_combined, y, test_size=test_size, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        model = Lasso(alpha=alpha, max_iter=1000)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        return mae_val

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_alpha = study.best_params['alpha']
    final_model = Lasso(alpha=best_alpha, max_iter=1000)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    print(" Best hyperparameters:", study.best_params)
    print(" Best validation MAE:", study.best_value)
    print(" Test MSE:", mean_squared_error(y_test, y_pred))
    print(" Test MAE:", mean_absolute_error(y_test, y_pred))

    return final_model, y_pred