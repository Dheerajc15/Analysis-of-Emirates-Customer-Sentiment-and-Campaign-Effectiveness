from __future__ import annotations

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils.nltk_setup import ensure_nltk

ensure_nltk()
_STOP_WORDS: set[str] = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: object, stop_words: set[str] | None = None) -> str:
    """Clean + preprocess a single review."""
    if stop_words is None:
        stop_words = _STOP_WORDS

    if not isinstance(text, str):
        return ""

    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    tokens = word_tokenize(t)

    return " ".join(
        _LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in stop_words
    )


def add_clean_text(
    df: pd.DataFrame,
    text_col: str = "Review",
    out_col: str = "review_clean",
) -> pd.DataFrame:
    """Add a cleaned text column to the DataFrame."""
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found.")
    df = df.copy()
    df[out_col] = df[text_col].apply(preprocess_text)
    return df