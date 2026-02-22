from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def add_vader_sentiment(
    df: pd.DataFrame,
    text_col: str = "review_clean",
    out_col: str = "sentiment_score",
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found.")
    df = df.copy()
    sia = SentimentIntensityAnalyzer()
    df[out_col] = df[text_col].apply(lambda t: sia.polarity_scores(t).get("compound", 0.0))
    return df

def average_sentiment_by_airline(
    df: pd.DataFrame,
    airline_col: str = "AirlineName",
    score_col: str = "sentiment_score",
) -> pd.DataFrame:
    if airline_col not in df.columns or score_col not in df.columns:
        raise KeyError("Expected columns not found for sentiment aggregation.")
    return (
        df.groupby(airline_col)[score_col]
          .mean()
          .sort_values(ascending=False)
          .reset_index(name="avg_sentiment")
    )
