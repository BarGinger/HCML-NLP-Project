import pandas as pd
from config import TRAIN_FILE, VALID_FILE, TEST_FILE

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_data():
    # Load train, validation, and test datasets as a dictionary
    data = {
        'train': load_data(TRAIN_FILE),
        'validation': load_data(VALID_FILE),
        'test': load_data(TEST_FILE)
    }
    return data