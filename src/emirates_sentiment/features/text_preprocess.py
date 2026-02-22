from __future__ import annotations

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from ..utils.nltk_setup import ensure_nltk

def preprocess_text(text: object, stop_words: set[str] | None = None) -> str:
    """Clean + preprocess a single review.
    Steps:
      1) lowercase
      2) remove punctuation/numbers
      3) tokenize
      4) remove stopwords
      5) lemmatize
    """
    ensure_nltk()
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()

    t = text.lower()
    t = re.sub(r"[^a-z\s]", "", t)
    tokens = word_tokenize(t)

    clean_tokens: list[str] = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tokens.append(lemmatizer.lemmatize(tok))

    return " ".join(clean_tokens)

def add_clean_text(
    df: pd.DataFrame,
    text_col: str = "Review",
    out_col: str = "review_clean",
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found.")
    df = df.copy()
    df[out_col] = df[text_col].apply(preprocess_text)
    return df
