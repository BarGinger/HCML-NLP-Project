import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text):

    if not isinstance(text, str):
        return ""

    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
           and not token.is_punct
           and not token.like_num
           and token.is_alpha
           and len(token) > 1
    ]
    return " ".join(tokens)


def preprocess_dataframe(df, text_column="review", new_column="review_clean"):
    df[new_column] = df[text_column].astype(str).apply(clean_text)
    return df