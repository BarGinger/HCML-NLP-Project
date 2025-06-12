import pandas as pd
from src.config import TRAIN_FILE_RAW, VALID_FILE_RAW, TEST_FILE_RAW


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loadin {file_path}: {e}")
        return None

def load_all_data():
    data = {
        'train': load_data(TRAIN_FILE_RAW),
        'validation': load_data(VALID_FILE_RAW),
        'test': load_data(TEST_FILE_RAW)
    }
    return data