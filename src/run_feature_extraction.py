import pandas as pd
import os
from config import (
    TRAIN_FILE_CLEAN, VALID_FILE_CLEAN, TEST_FILE_CLEAN,
    TRAIN_FILE_FEATURES, VALID_FILE_FEATURES, TEST_FILE_FEATURES,
    CLEAN_TEXT_COLUMN
)
from feature_extraction import add_sentiment_features


def main():
    path_mapping = {
        'train': (TRAIN_FILE_CLEAN, TRAIN_FILE_FEATURES),
        'validation': (VALID_FILE_CLEAN, VALID_FILE_FEATURES),
        'test': (TEST_FILE_CLEAN, TEST_FILE_FEATURES)
    }

    for split, (input_path, output_path) in path_mapping.items():
        if not os.path.exists(input_path):
            print(f"preprocessed file not found: {input_path}")
            continue

        print("Loadin preprocessed file")
        df = pd.read_csv(input_path)

        print("adding sentiiment features to data")
        df = add_sentiment_features(df, text_column=CLEAN_TEXT_COLUMN)

        df.to_csv(output_path, index=False)
        print("done")


if __name__ == "__main__":
    main()