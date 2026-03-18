from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_vader_sentiment(
    df: pd.DataFrame,
    text_col: str = "review_clean",
    out_col: str = "vader_score",
) -> pd.DataFrame:
    """Score each row with VADER compound sentiment."""
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found.")
    df = df.copy()
    sia = SentimentIntensityAnalyzer()
    df[out_col] = df[text_col].apply(
        lambda t: sia.polarity_scores(str(t)).get("compound", 0.0)
    )
    return df


def vader_label(score: float) -> str:
    """Convert VADER compound score to categorical label."""
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def average_sentiment_by_airline(
    df: pd.DataFrame,
    airline_col: str = "AirlineName",
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Average sentiment grouped by airline."""
    if airline_col not in df.columns or score_col not in df.columns:
        raise KeyError("Expected columns not found for sentiment aggregation.")
    return (
        df.groupby(airline_col)[score_col]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="avg_sentiment")
    )