from data_loader import load_all_data
from preprocess import preprocess_dataframe
from config import TEXT_COLUMN


def main():
    data = load_all_data()
    for split in ['train', 'validation', 'test']:
        print(f"Preprocessing {split} data...")
        df = data[split]
        df = preprocess_dataframe(df, text_column=TEXT_COLUMN, new_column="review_clean")
        data[split] = df
        df.to_csv(f"../Data/drug_review_{split}_clean.csv", index=False)
        print(f"{split} data cleaned and saved.")


if __name__ == "__main__":
    main()