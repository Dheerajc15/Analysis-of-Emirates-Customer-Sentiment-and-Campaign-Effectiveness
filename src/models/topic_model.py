from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

VectorizerKind = Literal["tfidf", "count"]

@dataclass
class TopicResult:
    name: str
    topics: list[str]

def split_by_overall_score(
    df_emirates: pd.DataFrame,
    score_col: str = "OverallScore",
    text_col: str = "review_clean",
    low_threshold: float = 3,
    high_threshold: float = 8,
    min_text_len: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (negative_reviews_df, positive_reviews_df)."""
    df = df_emirates.copy()
    if score_col not in df.columns:
        raise KeyError(f"Expected column '{score_col}' not found.")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col, text_col])
    df = df[df[text_col].astype(str).str.len() > min_text_len]

    neg = df[df[score_col] <= low_threshold].copy()
    pos = df[df[score_col] >= high_threshold].copy()
    return neg, pos

def run_lda_topics(
    texts: pd.Series,
    n_topics: int = 5,
    n_words: int = 10,
    stop_words: list[str] | str = "english",
    max_df: float = 0.9,
    min_df: int = 5,
    vectorizer: VectorizerKind = "tfidf",
    ngram_range: tuple[int, int] = (1, 1),
    random_state: int = 42,
) -> list[str]:
    """Fit LDA and return list of topic strings (top words)."""
    if texts is None or len(texts) == 0:
        return []

    if vectorizer == "count":
        vec = CountVectorizer(
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
        )
    else:
        vec = TfidfVectorizer(
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
        )

    X = vec.fit_transform(texts.astype(str))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda.fit(X)

    feature_names = vec.get_feature_names_out()
    topics: list[str] = []
    for comp in lda.components_:
        top_idx = comp.argsort()[:-n_words - 1:-1]
        topics.append(" ".join(feature_names[i] for i in top_idx))
    return topics
