# 1. Linear Regression (e.g., using TF-IDF features)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from quantus import Complexity

# Prepare data
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

lr = LogisticRegression()
lr.fit(X_train, train_labels)
lr_preds = lr.predict(X_test)

# Example explanation: use absolute value of coefficients as feature importances
explanations_lr = [abs(lr.coef_[0])] * len(X_test)
complexity_lr = Complexity()(explanations=explanations_lr, model=lr, x=X_test, y=test_labels)
print("Linear Regression Complexity:", complexity_lr)

# 2. Gradient Boosting Machine (e.g., XGBoost)
from xgboost import XGBClassifier

gbm = XGBClassifier()
gbm.fit(X_train, train_labels)
gbm_preds = gbm.predict(X_test)

# Example explanation: feature importances
explanations_gbm = [gbm.feature_importances_] * len(X_test)
complexity_gbm = Complexity()(explanations=explanations_gbm, model=gbm, x=X_test, y=test_labels)
print("GBM Complexity:", complexity_gbm)

# 3. Fine-tuned BERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased")
# Assume you have fine-tuned the model already

inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
bert_preds = outputs.logits.argmax(dim=1).numpy()

# Example explanation: use attention weights or LIME/SHAP attributions
# Here, we use dummy explanations for illustration
explanations_bert = [torch.rand(inputs['input_ids'].shape[1]).numpy() for _ in range(len(test_texts))]
complexity_bert = Complexity()(explanations=explanations_bert, model=model, x=inputs['input_ids'], y=test_labels)
print("BERT Complexity:", complexity_bert)

# 4. SelfExplain (pseudo-code, replace with your implementation)
# from self_explain import SelfExplainModel
# self_explain = SelfExplainModel(...)
# self_explain.fit(...)
# explanations_self = self_explain.explain(test_texts)
# complexity_self = Complexity()(explanations=explanations_self, model=self_explain, x=test_texts, y=test_labels)
# print("SelfExplain Complexity:", complexity_self)

# 5. Proto-LM (using your PyTorch Lightning setup)
# Assuming you have a trained proto model and datamodule
# proto = ... (your trained Proto-LM model)
# drug_review_dm = ... (your datamodule)
# explanations_proto = proto.explain(test_texts)  # Replace with your method
# complexity_proto = Complexity()(explanations=explanations_proto, model=proto, x=test_texts, y=test_labels)
# print("Proto-LM Complexity:", complexity_proto)