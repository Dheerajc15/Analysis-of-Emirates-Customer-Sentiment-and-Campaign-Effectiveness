from __future__ import annotations
from typing import Literal
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

VectorizerKind = Literal["tfidf", "count"]


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
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 42,
) -> list[dict]:
    """
    Fit LDA and return list of dicts:
      [{'topic_id': 0, 'label': 'topic_0', 'top_words': 'word1 word2 ...', 'weight': float}, ...]
    """
    if texts is None or len(texts) == 0:
        return []

    safe_min_df = min(min_df, max(1, len(texts) // 2))

    if vectorizer == "count":
        vec = CountVectorizer(
            stop_words=stop_words,
            max_df=max_df,
            min_df=safe_min_df,
            ngram_range=ngram_range,
        )
    else:
        vec = TfidfVectorizer(
            stop_words=stop_words,
            max_df=max_df,
            min_df=safe_min_df,
            ngram_range=ngram_range,
        )

    X = vec.fit_transform(texts.astype(str))

    if X.shape[1] == 0:
        return []

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state
    )
    lda.fit(X)

    feature_names = vec.get_feature_names_out()
    topics: list[dict] = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[: -n_words - 1 : -1]
        top_words = " ".join(feature_names[i] for i in top_idx)
        weight = comp[top_idx].sum()
        topics.append(
            {
                "topic_id": idx,
                "label": f"topic_{idx}",
                "top_words": top_words,
                "weight": round(float(weight), 2),
            }
        )
    return topics