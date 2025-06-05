
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel

def count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

train_data = pd.read_csv('Data/drug_review_train_clean.csv')
val_data = pd.read_csv('Data/drug_review_validation_clean.csv')
test_data = pd.read_csv('Data/drug_review_test_clean.csv')

#Define the features (X) and the target variable (y)
label = "rating"
X_train = train_data.drop(columns=[label])  # Drop the target column
y_train = train_data[label]  # Target column
X_val = val_data.drop(columns=[label])  # Drop the target column
y_val = val_data[label]  # Target columnX_train = train_data.drop(columns=[label])  # Drop the target column
X_test = test_data.drop(columns=[label])  # Drop the target column
y_test = test_data[label]  # Target column

print(print(pd.Series.nunique(X_train['drugName'])))

# Get the text column (update if your text column has a different name)
text_column = 'review'

# Fill missing values
X_train[text_column] = X_train[text_column].fillna("")
X_val[text_column] = X_val[text_column].fillna("")
X_test[text_column] = X_test[text_column].fillna("")

# Initialize TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # You can tune this

# Fit TF-IDF on training data and transform all sets
X_train_tfidf = tfidf.fit_transform(X_train[text_column])
X_val_tfidf = tfidf.transform(X_val[text_column])
X_test_tfidf = tfidf.transform(X_test[text_column])

def map_sentiment(rating):
    if rating >= 7:
        return 'positive'
    elif rating >= 4:
        return 'neutral'
    else:
        return 'negative'

y_train_sent = y_train.apply(map_sentiment)
y_val_sent = y_val.apply(map_sentiment)
y_test_sent = y_test.apply(map_sentiment)

# Encode strings into integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_sent)
y_val_enc = le.transform(y_val_sent)
y_test_enc = le.transform(y_test_sent)

# Train LWGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100)
lgb_model.fit(X_train_tfidf, y_train_enc)

# Select features based on importance
selector = SelectFromModel(lgb_model, prefit=True, threshold='median')  # keep top 50%
X_train_sel = selector.transform(X_train_tfidf)
X_val_sel = selector.transform(X_val_tfidf)
X_test_sel = selector.transform(X_test_tfidf)



