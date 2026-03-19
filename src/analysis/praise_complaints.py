from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PATHS
from models.topic_model import split_by_overall_score
from utils.logging import get_logger

LOGGER = get_logger(__name__)

# ── Internal constants ────────────────────────────────────────────────────────
_KEYWORDS_FILE = "category_keywords.json"
_MIN_REVIEWS    = 10   # minimum reviews needed before running analysis


# ── Keyword loader ────────────────────────────────────────────────────────────

def _load_category_keywords() -> dict[str, dict[str, list[str]]]:
    """
    Load category keyword maps from data/scraped_inputs/category_keywords.json.

    Returns a dict with two keys: 'praise' and 'complaint', each mapping
    category name → list of keywords.

    Falls back to empty dicts if the file is missing.
    """
    json_path: Path = PATHS.scraped_inputs / _KEYWORDS_FILE
    if not json_path.exists():
        LOGGER.warning(
            "Category keywords file not found at %s. "
            "Categories will be labelled 'General'.",
            json_path,
        )
        return {"praise": {}, "complaint": {}}

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        LOGGER.info("Loaded category keywords from %s", json_path.name)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.error("Failed to load category keywords: %s", exc)
        return {"praise": {}, "complaint": {}}


# ── Review splitting ──────────────────────────────────────────────────────────

def _split_by_model_sentiment(
    df: pd.DataFrame,
    sentiment_label_col: str = "sentiment_label",
    text_col: str = "review_clean",
    min_text_len: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into (negative_df, positive_df) using the winning
    model's 'sentiment_label' column ('positive' / 'neutral' / 'negative').

    Only reviews with sufficient text length are kept.
    Returns (neg_df, pos_df).
    """
    df = df.copy()
    df = df[df[text_col].astype(str).str.len() > min_text_len]

    pos = df[df[sentiment_label_col] == "positive"].copy()
    neg = df[df[sentiment_label_col] == "negative"].copy()
    return neg, pos


def _get_review_splits(
    df_emirates: pd.DataFrame,
    text_col: str = "review_clean",
    sentiment_label_col: str = "sentiment_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Determine how to split reviews:
      - PRIMARY:  use sentiment_label produced by the winning NLP model.
      - FALLBACK: use OverallScore thresholds (≤4 negative, ≥7 positive)
                  when sentiment_label column is not available.

    Returns (neg_df, pos_df).
    """
    if sentiment_label_col in df_emirates.columns:
        LOGGER.info(
            "Splitting reviews using winning model's '%s' column.", sentiment_label_col
        )
        neg, pos = _split_by_model_sentiment(
            df_emirates,
            sentiment_label_col=sentiment_label_col,
            text_col=text_col,
        )
    else:
        LOGGER.warning(
            "'%s' column not found — falling back to OverallScore split.",
            sentiment_label_col,
        )
        neg, pos = split_by_overall_score(
            df_emirates, low_threshold=4, high_threshold=7
        )

    LOGGER.info(
        "Split result: %d positive reviews, %d negative reviews.", len(pos), len(neg)
    )
    return neg, pos


# ── TF-IDF category scoring ───────────────────────────────────────────────────

def _score_categories_tfidf(
    texts: pd.Series,
    keyword_map: dict[str, list[str]],
    top_n: int = 5,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """
    Rank categories by how prominently their keywords appear in the review texts,
    using TF-IDF weighting so common stop-words don't dominate.

    For each category:
      - Score = sum of TF-IDF weights across all reviews for all matching keywords.

    Parameters
    ----------
    texts       : pd.Series of cleaned review strings
    keyword_map : dict mapping category name → list of keyword strings
    top_n       : number of top categories to return
    max_features: TfidfVectorizer vocabulary size cap

    Returns
    -------
    pd.DataFrame with columns [rank, category, top_words, weight]
    """
    if texts.empty or not keyword_map:
        return pd.DataFrame(columns=["rank", "category", "top_words", "weight"])

    # Fit TF-IDF on the review corpus
    vec = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=max(2, len(texts) // 100),   # adaptive min_df
        ngram_range=ngram_range,
        max_features=max_features,
    )
    try:
        X = vec.fit_transform(texts.astype(str))
    except ValueError:
        LOGGER.warning("TF-IDF vectorizer failed (too few terms). Returning empty.")
        return pd.DataFrame(columns=["rank", "category", "top_words", "weight"])

    vocab = vec.get_feature_names_out()
    # Mean TF-IDF weight per term across all documents
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    term_weight: dict[str, float] = dict(zip(vocab, mean_tfidf))

    rows = []
    for category, keywords in keyword_map.items():
        # Find which keywords (and n-grams containing them) appear in the vocab
        matched_terms: dict[str, float] = {}
        for kw in keywords:
            kw_lower = kw.lower()
            # Exact match + partial n-gram match
            for term, weight in term_weight.items():
                if kw_lower in term:
                    matched_terms[term] = max(
                        matched_terms.get(term, 0.0), weight
                    )

        if not matched_terms:
            category_weight = 0.0
            top_words_str   = "(no matching terms found)"
        else:
            # Sort matched terms by their TF-IDF weight descending
            sorted_terms    = sorted(matched_terms.items(), key=lambda x: x[1], reverse=True)
            top_words_str   = " | ".join(t for t, _ in sorted_terms[:10])
            category_weight = sum(w for _, w in sorted_terms)

        rows.append(
            {
                "category": category,
                "top_words": top_words_str,
                "weight":    round(category_weight, 4),
            }
        )

    result = pd.DataFrame(rows)
    result = (
        result[result["weight"] > 0]           # drop zero-weight categories
        .sort_values("weight", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def extract_top_praises_and_complaints(
    df_emirates: pd.DataFrame,
    top_n: int = 5,
    text_col: str = "review_clean",
    sentiment_label_col: str = "sentiment_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the top-N praise categories and top-N complaint categories from
    Emirates reviews, driven entirely by the NLP model's predictions and
    TF-IDF scoring.

    Parameters
    ----------
    df_emirates         : DataFrame of Emirates reviews, must include 'review_clean'
                          and ideally 'sentiment_label' (set by the winning model).
    top_n               : How many top categories to return (default 5).
    text_col            : Name of the cleaned text column.
    sentiment_label_col : Column produced by assign_best_sentiment() in the pipeline.

    Returns
    -------
    (praise_df, complaint_df)
    Each DataFrame has columns: [rank, category, top_words, weight]
    """
    # 1. Load category keyword maps from JSON 
    keywords = _load_category_keywords()
    praise_kw_map    = keywords.get("praise",    {})
    complaint_kw_map = keywords.get("complaint", {})

    # 2. Split reviews using the winning model's predictions
    neg_df, pos_df = _get_review_splits(
        df_emirates,
        text_col=text_col,
        sentiment_label_col=sentiment_label_col,
    )

    # 3. Guard: need a minimum number of reviews to run TF-IDF meaningfully
    if len(pos_df) < _MIN_REVIEWS:
        LOGGER.warning(
            "Only %d positive reviews available (need %d). "
            "Praise analysis may be unreliable.",
            len(pos_df), _MIN_REVIEWS,
        )
    if len(neg_df) < _MIN_REVIEWS:
        LOGGER.warning(
            "Only %d negative reviews available (need %d). "
            "Complaint analysis may be unreliable.",
            len(neg_df), _MIN_REVIEWS,
        )

    # 4. Score categories via TF-IDF (data-driven, no manual matching)
    LOGGER.info("Scoring praise categories from %d positive reviews...", len(pos_df))
    praise_df = _score_categories_tfidf(
        pos_df[text_col],
        praise_kw_map,
        top_n=top_n,
    )

    LOGGER.info("Scoring complaint categories from %d negative reviews...", len(neg_df))
    complaint_df = _score_categories_tfidf(
        neg_df[text_col],
        complaint_kw_map,
        top_n=top_n,
    )

    LOGGER.info("═" * 55)
    LOGGER.info("TOP %d PRAISES (data-driven via %s):", top_n, sentiment_label_col)
    LOGGER.info("═" * 55)
    LOGGER.info("\n%s", praise_df.to_string(index=False))

    LOGGER.info("═" * 55)
    LOGGER.info("TOP %d COMPLAINTS (data-driven via %s):", top_n, sentiment_label_col)
    LOGGER.info("═" * 55)
    LOGGER.info("\n%s", complaint_df.to_string(index=False))

    return praise_df, complaint_df