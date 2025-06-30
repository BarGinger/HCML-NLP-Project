from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TRAIN_FILE_RAW = DATA_DIR / "drug_review_train.csv"
VALID_FILE_RAW = DATA_DIR / "drug_review_validation.csv"
TEST_FILE_RAW = DATA_DIR / "drug_review_test.csv"
TRAIN_FILE_CLEAN = DATA_DIR / "drug_review_train_clean.csv"
VALID_FILE_CLEAN = DATA_DIR / "drug_review_validation_clean.csv"
TEST_FILE_CLEAN = DATA_DIR / "drug_review_test_clean.csv"
TRAIN_FILE_FEATURES = DATA_DIR / "drug_review_train_features.csv"
VALID_FILE_FEATURES = DATA_DIR / "drug_review_validation_features.csv"
TEST_FILE_FEATURES = DATA_DIR / "drug_review_test_features.csv"
TEXT_COLUMN = "review"
CLEAN_TEXT_COLUMN = "review_clean"
TARGET_COLUMN = "rating"
SEED = 42

MAX_FEATURES_TFIDF = 1000
LASSO_PARAMS = {'alpha': 1.0}

GBM_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1}
BERT_MODEL_NAME = 'bert-base-uncased'