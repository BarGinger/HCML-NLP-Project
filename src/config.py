import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, "Data")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ==== Data Files ====
TRAIN_FILE = os.path.join(DATA_DIR, "drug_review_train.csv")
VALID_FILE = os.path.join(DATA_DIR, "drug_review_validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "drug_review_test.csv")

# ==== General Constants ====
SEED = 42        # For reproducibility
MAX_FEATURES = 1000  # For TF-IDF/Word2Vec
TEXT_COLUMN = "review"  # Name of the text column in the dataset
TARGET_COLUMN = "rating"  # Name of the target column

LINEAR_REG_PARAMS = {}
GBM_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1}
BERT_MODEL_NAME = 'bert-base-uncased'