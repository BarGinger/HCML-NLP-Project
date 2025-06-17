import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from feature_extraction import feature_extraction
from sklearn.model_selection import train_test_split
import optuna



def run_lasso_model(df, target_column='rating', test_size=0.2, alpha=0.01):
    # Step 1: Feature extraction with fit=True
    X, tfidf_vectorizer, svd = feature_extraction(df, fit=True)

    # Step 2: Train-test split
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Step 3: Train Lasso model
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

    return lasso, tfidf_vectorizer, svd, X_test, y_test, y_pred








def evaluate_lasso_model(y_test, y_pred):
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

def analyze_lasso_coefficients(lasso_model, svd_model, tfidf_vectorizer):
    # Get original TF-IDF feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # SVD components: n_components x n_original_features
    # Coefficients from Lasso correspond to SVD dimensions + sentiment
    tfidf_feature_contributions = lasso_model.coef_[:-1]  # Exclude last sentiment feature
    sentiment_coef = lasso_model.coef_[-1]                # Last one is sentiment

    print("Sentiment coefficient:", sentiment_coef)
    print("\nTop TF-IDF SVD component contributions:")

    for i, coef in enumerate(tfidf_feature_contributions):
        print(f"Component {i}: Coef = {coef:.4f}")

        
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




def run_lasso_with_optuna(df, target_column='rating', test_size=0.2, n_trials=30):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import optuna

    # Step 1: Extract features
    X, tfidf_vectorizer, svd = feature_extraction(df, fit=True)
    y = df[target_column].values

    # Step 2: Train/Test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Step 3: Split train into train/validation for tuning
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        model = Lasso(alpha=alpha, max_iter=1000)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        mae_val = mean_absolute_error(y_val, y_val_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        trial.set_user_attr("MAE_val", mae_val)
        trial.set_user_attr("MSE_test", mse_test)
        trial.set_user_attr("MAE_test", mae_test)

        return mse_val

    # Step 4: Optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Step 5: Final model training with best alpha
    best_alpha = study.best_params['alpha']
    final_model = Lasso(alpha=best_alpha, max_iter=1000)
    final_model.fit(X_train_full, y_train_full)

    # Step 6: Predict and return everything needed
    y_pred = final_model.predict(X_test)

    print(" Best hyperparameters:", study.best_params)
    print(" Best validation MSE:", study.best_trial.value)
    print(" Best validation MAE:", study.best_trial.user_attrs["MAE_val"])
    print(" Best test MSE:", study.best_trial.user_attrs["MSE_test"])
    print(" Best test MAE:", study.best_trial.user_attrs["MAE_test"])

    return final_model, tfidf_vectorizer, svd, X_test, y_test, y_pred
