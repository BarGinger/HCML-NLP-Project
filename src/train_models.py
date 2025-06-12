import pandas as pd
import torch
import joblib
import os
from scipy.sparse import hstack
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from config import (
    MODELS_DIR,
    TRAIN_FILE_FEATURES,
    VALID_FILE_FEATURES,
    CLEAN_TEXT_COLUMN,
    TARGET_COLUMN,
    MAX_FEATURES_TFIDF,
    LASSO_PARAMS,
    BERT_MODEL_NAME,
    SEED
)


#INTERPRETABLE MODEL: LASSO REGRESSION

def train_lasso_model():
    """
    Trains and saves an interpretable LASSO regression model.
    This function uses TF-IDF for text features and combines them with
    sentiment scores to predict the user rating.
    """
    print("--- Training Interpretable Model: LASSO Regression ---")

    # Load data
    df_train = pd.read_csv(TRAIN_FILE_FEATURES)
    df_valid = pd.read_csv(VALID_FILE_FEATURES)
    df_train.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    df_valid.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)

    # Define features (X) and target (y)
    sentiment_features = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
    y_train = df_train[TARGET_COLUMN]
    y_valid = df_valid[TARGET_COLUMN]

    # Initialize and fit TF-IDF Vectorizer
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=(1, 2))
    X_train_text = vectorizer.fit_transform(df_train[CLEAN_TEXT_COLUMN])
    X_valid_text = vectorizer.transform(df_valid[CLEAN_TEXT_COLUMN])

    # Combine text features with sentiment features
    X_train = hstack([X_train_text, df_train[sentiment_features]])
    X_valid = hstack([X_valid_text, df_valid[sentiment_features]])

    # Train LASSO model
    model = Lasso(alpha=LASSO_PARAMS.get('alpha', 1.0), random_state=SEED)
    model.fit(X_train, y_train)

    # Evaluate on validation set using metrics from the proposal
    preds = model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)  # Calculate MSE first
    rmse = mse ** 0.5  # Then take the square root to get RMSE
    mae = mean_absolute_error(y_valid, preds)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    # Save model and vectorizer
    joblib.dump(model, os.path.join(MODELS_DIR, "lasso_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_lasso.joblib"))
    print(f"LASSO model and TF-IDF vectorizer saved to '{MODELS_DIR}'")




def train_pca_regression_model():

    df_train = pd.read_csv(TRAIN_FILE_FEATURES)
    df_valid = pd.read_csv(VALID_FILE_FEATURES)
    df_train.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    df_valid.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)

    y_train = df_train[TARGET_COLUMN]
    y_valid = df_valid[TARGET_COLUMN]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(df_train[CLEAN_TEXT_COLUMN])


    num_components = 20
    pca = PCA(n_components=num_components, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_tfidf.toarray())

    feature_names = vectorizer.get_feature_names_out()
    for i, component in enumerate(pca.components_):
         top_word_indices = component.argsort()[-10:][::-1]
         top_words = [feature_names[j] for j in top_word_indices]


    model = LinearRegression()
    model.fit(X_train_pca, y_train)

    X_valid_tfidf = vectorizer.transform(df_valid[CLEAN_TEXT_COLUMN])
    X_valid_pca = pca.transform(X_valid_tfidf.toarray())
    preds = model.predict(X_valid_pca)
    mse = mean_squared_error(y_valid, preds)
    rmse = mse ** 0.5
    print(f"Validation RMSE on PCA features: {rmse:.4f}")

    joblib.dump(model, os.path.join(MODELS_DIR, "pca_regression_model.joblib"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_transformer.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_pca.joblib"))



class DrugReviewDataset(Dataset):
    """Custom PyTorch Dataset for loading drug review data."""

    def __init__(self, texts, ratings, tokenizer, max_len=128):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        rating = self.ratings[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(rating, dtype=torch.float)
        }


def train_bert_model():

    print("Training Black-Box Model: BERT")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Tokenizer and Model for regression (num_labels=1)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=1)
    model.to(device)

    # Load data
    df_train = pd.read_csv(TRAIN_FILE_FEATURES)
    df_valid = pd.read_csv(VALID_FILE_FEATURES)
    df_train.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    df_valid.dropna(subset=[CLEAN_TEXT_COLUMN, TARGET_COLUMN], inplace=True)

    train_dataset = DrugReviewDataset(df_train[CLEAN_TEXT_COLUMN].to_numpy(), df_train[TARGET_COLUMN].to_numpy(),
                                      tokenizer)
    valid_dataset = DrugReviewDataset(df_valid[CLEAN_TEXT_COLUMN].to_numpy(), df_valid[TARGET_COLUMN].to_numpy(),
                                      tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = MSELoss()

    for epoch in range(3):  # Example: 3 epochs
        print(f"\nEpoch {epoch + 1}/3")
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch complete. Evaluating on validation set...")
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
        avg_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_loss:.4f}, Validation RMSE: {avg_loss ** 0.5:.4f}")

    # Save the fine-tuned model
    save_path = os.path.join(MODELS_DIR, "bert_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"BERT model and tokenizer saved to '{save_path}'")


# --- MAIN ORCHESTRATOR ---
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("===== Starting All Model Training =====")

    # Train Model 1: LASSO
    print("\n\nTraining")
    train_lasso_model()
    print("\nLASSO Model training complete.")

    # Train Model 2: PCA + Linear Regression
    print("\n\n[2/3] Training PCA + Linear Regression Model...")
    train_pca_regression_model()
    print("\nPCA + Linear Regression Model training complete.")

    # Train Model 3: BERT
    print("\n\n[3/3] Training BERT Model...")
    # train_bert_model()
    print("\nBERT Model training complete.")

    print("\n\n===== All model training finished. =====")