import spacy
import re
import contractions

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading language model for spaCy...")
    print("python -m spacy download en_core_web_sm")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    # expanding contractions (I'm to I am)
    try:
        text = contractions.fix(text)
    except Exception:
        pass
    text = text.lower()

    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s-]', '', text) # Remove characters that are not word characters, whitespace, or hyphen


    doc = nlp(text)

    #Lemmatize and filter tokens
    clean_tokens = []
    for token in doc:
        is_essentially_alpha_or_hyphenated_word = token.is_alpha or \
                                                 (token.text.count('-') == 1 and \
                                                  all(part.isalpha() for part in token.text.split('-')))


        if (not token.is_stop and
            not token.is_punct and
            not token.like_num and
            is_essentially_alpha_or_hyphenated_word and
            len(token.lemma_) > 1):
            clean_tokens.append(token.lemma_)

    final_text = " ".join(clean_tokens)
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    return final_text


def preprocess_dataframe(df, text_column="review", new_column="review_clean"):
    # this func is just for efficiency
    texts = []
    for text in df[text_column].fillna("").astype(str):
        try:
            text = contractions.fix(text)  # e.g., I'm -> I am
        except Exception:
            pass
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove URLs
        text = re.sub(r'\S*@\S*\s?', '', text)  # remove emails
        text = re.sub(r'[^\w\s-]', '', text)  # remove special characters
        texts.append(text)
    final_clean_texts = []
    for doc in nlp.pipe(texts):
        clean_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and
               not token.is_punct and
               not token.like_num and
               len(token.lemma_) > 1 and
               token.is_alpha
        ]
        final_clean_texts.append(" ".join(clean_tokens))

    df[new_column] = final_clean_texts
    return df