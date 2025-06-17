from feature_extraction import feature_extraction
import pandas as pd
from check_the_data import run_lasso_with_optuna, evaluate_lasso_model, analyze_lasso_coefficients, print_top_words_per_svd_component,print_top_features_by_absolute_weight

# Load data
df = pd.read_csv("/Users/youssefbenmansour/Desktop/HCML-NLP-Project/Data/drug_review_train_clean.csv")

# Run and retrieve components
lasso_model, tfidf_vectorizer, svd, X_test, y_test, y_pred = run_lasso_with_optuna(df)

# Evaluate
evaluate_lasso_model(y_test, y_pred)

# Interpret model
analyze_lasso_coefficients(lasso_model, svd, tfidf_vectorizer)

print_top_features_by_absolute_weight(lasso_model, svd, tfidf_vectorizer, top_n=15)

print_top_words_per_svd_component(svd, tfidf_vectorizer, top_n=5, show_positive_and_negative=True)




