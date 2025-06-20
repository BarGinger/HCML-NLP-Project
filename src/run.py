import pandas as pd
from check_the_data import run_lasso_with_optuna, evaluate_lasso_model, analyze_lasso_coefficients, print_top_words_per_svd_component,print_top_features_by_absolute_weight
from check_the_data import print_most_important_features
import os
import joblib
from feature_extraction import load_data

print("Hello World")


# load the data
X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test, tfidf_vectorizer, svd = load_data()


# Define the directory path for models
models_dir = "Models"

# Check if the directory exists
if not os.path.exists(models_dir):
    # Create the directory if it doesn't exist
    os.makedirs(models_dir)
    print(f"Directory '{models_dir}' created.")
else:
    print(f"Directory '{models_dir}' already exists.")

# Run and retrieve components
# lasso_model, tfidf_vectorizer, X_test, y_test, y_pred = run_lasso_with_optuna(df)
# final_model,tfidf_vectorizer, X_test, y_test, y_pred = run_lasso_with_optuna(df, n_trials=20)
final_model, y_pred = run_lasso_with_optuna(X_train_combined, y_train,
                                                              X_val_combined, y_val, X_test_combined,
                                                                y_test, tfidf_vectorizer, svd, n_trials=200)


# Save the trained Lasso model to a file
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"{models_dir}/trained_lasso_model_{timestamp}.joblib"
joblib.dump(final_model, model_filename)
print(f"Lasso model saved to {model_filename}")


evaluate_lasso_model(y_test, y_pred)
num_features = X_test_combined.shape[1]
print(f"Number of features in the test set: {num_features}")
# feature_names = list(tfidf_vectorizer.get_feature_names_out()) + ['sentiment_score']
feature_names = [f"PCA_feature_{i}" for i in range(num_features - 1)] + ['sentiment_score']

top_features = max(20, num_features)  # Ensure top_n does not exceed number of features
print_most_important_features(final_model, feature_names, top_features)


# print_most_important_features(lasso_model, tfidf_vectorizer, top_n=20)


# # Evaluate
# evaluate_lasso_model(y_test, y_pred)

# # Interpret model
# # analyze_lasso_coefficients(lasso_model, svd, tfidf_vectorizer)

# # print_top_features_by_absolute_weight(lasso_model, svd, tfidf_vectorizer, top_n=15)

# # print_top_words_per_svd_component(svd, tfidf_vectorizer, top_n=5, show_positive_and_negative=True)

print("Goodbye World")



