
import pandas as pd
import torch
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from config import (
    MODELS_DIR,
    TEST_FILE_FEATURES,
    CLEAN_TEXT_COLUMN,
    TARGET_COLUMN
)
from train_models import DrugReviewDataset

def evaluate_pca_model():
    print("--- Evaluating model: PCA + Linear Regression")

    model = joblib.load(os.path.join(MODELS_DIR, "pca_regression_model.joblib"))
    pca = joblib.load(os.path.join(MODELS_DIR, "pca_transformer.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_pca.joblib"))

    df_test = pd.read_csv(TEST_FILE_FEATURES)
    df_test.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    X_test_tfidf = vectorizer.transform(df_test[CLEAN_TEXT_COLUMN])
    X_test_pca = pca.transform(X_test_tfidf.toarray())
    y_test = df_test[TARGET_COLUMN]
    predictions = model.predict(X_test_pca)
    print_metrics(y_test, predictions)


def evaluate_bert_model():
    print("--- evaluating blackbox model: BERT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, "bert_model")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    df_test = pd.read_csv(TEST_FILE_FEATURES)
    df_test.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    test_dataset = DrugReviewDataset(df_test[CLEAN_TEXT_COLUMN].to_numpy(), df_test[TARGET_COLUMN].to_numpy(),
                                     tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.squeeze().tolist())
            true_labels.extend(labels.tolist())

    print_metrics(true_labels, predictions)


def print_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n--- Performance Metrics ---")
    print(f"RMSE :{rmse:.4f}")
    print(f"MAE :  {mae:.4f}")


if __name__ == "__main__":
    while True:
        print("\nWhich model would you like to evaluate?")
        print("  q: PCA + Linear Regression")
        print("  2: BERT")
        choice = input("Please enter a number (1-3) or 'q' to quit: ")
        if choice == '1':
            evaluate_pca_model()
            break
        elif choice == '2':
            evaluate_bert_model()
            break
        elif choice.lower() == 'q':
            print("Exiting")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or q.")