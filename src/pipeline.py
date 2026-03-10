from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from config import PATHS, TARGET_AIRLINES, get_news_api_key
from data.load import load_reviews, filter_airlines, split_emirates
from data.events import load_sponsorship_events
from features.text_preprocess import add_clean_text
from models.sentiment import add_vader_sentiment, average_sentiment_by_airline
from models.topic_model import split_by_overall_score, run_lda_topics
from external.trends import fetch_google_trends
from external.news import fetch_news_sentiment
from viz.plots import (
    plot_sentiment_distribution,
    plot_service_ratings,
    plot_sentiment_over_time,
    plot_trends_with_events,
)

@dataclass
class PipelineOutputs:
    df_rivals: pd.DataFrame
    df_emirates: pd.DataFrame
    sentiment_by_airline: pd.DataFrame
    pain_topics: list[str]
    praise_topics: list[str]
    trends_df: pd.DataFrame | None = None
    news_avg_sentiment: float | None = None


def run_review_pipeline(reviews_csv: str | Path) -> PipelineOutputs:
    df = load_reviews(reviews_csv)
    df_rivals = filter_airlines(df, TARGET_AIRLINES)
    df_rivals, df_emirates = split_emirates(df_rivals) 

    df_rivals = add_clean_text(df_rivals, text_col="Review", out_col="review_clean")
    df_emirates = add_clean_text(df_emirates, text_col="Review", out_col="review_clean")

    df_rivals = add_vader_sentiment(df_rivals)
    df_emirates = add_vader_sentiment(df_emirates)

    sentiment_tbl = average_sentiment_by_airline(df_rivals)

    neg, pos = split_by_overall_score(df_emirates)
    pain_topics = run_lda_topics(neg["review_clean"], n_topics=5, n_words=10, vectorizer="tfidf")
    praise_topics = run_lda_topics(pos["review_clean"], n_topics=5, n_words=10, vectorizer="tfidf")

    return PipelineOutputs(
        df_rivals=df_rivals,
        df_emirates=df_emirates,
        sentiment_by_airline=sentiment_tbl,
        pain_topics=pain_topics,
        praise_topics=praise_topics,
    )


def run_external_signals(
    keywords: list[str] | None = None,  
    trends_timeframe: str = "today 12-m",
    news_query: str = "Emirates airline",
) -> tuple[pd.DataFrame, float]:
    if keywords is None:
        keywords = ["Emirates", "Qatar Airways"]  # safe default created fresh each call

    trends_df = fetch_google_trends(keywords, timeframe=trends_timeframe)
    api_key = get_news_api_key()

    try:
        news_sent = fetch_news_sentiment(news_query, api_key=api_key or "")
    except ValueError as e:
        print(f"Warning: Could not fetch news sentiment — {e}")
        news_sent = 0.0

    return trends_df, news_sent


def save_tables(outputs: PipelineOutputs) -> None:
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    PATHS.reports_tables.mkdir(parents=True, exist_ok=True)

    outputs.df_rivals.to_csv(PATHS.data_processed / "rivals_scored.csv", index=False)
    outputs.df_emirates.to_csv(PATHS.data_processed / "emirates_scored.csv", index=False)
    outputs.sentiment_by_airline.to_csv(PATHS.reports_tables / "sentiment_by_airline.csv", index=False)


def make_figures(outputs: PipelineOutputs) -> None:
    PATHS.reports_figures.mkdir(parents=True, exist_ok=True)

    plot_sentiment_distribution(outputs.df_rivals, PATHS.reports_figures / "sentiment_distribution.png")
    plot_service_ratings(outputs.df_rivals, PATHS.reports_figures / "service_ratings.png")
    plot_sentiment_over_time(outputs.df_rivals, PATHS.reports_figures / "sentiment_over_time.png")


def make_event_overlay(
    events_csv: str | Path,
    keywords: list[str] | None = None,  
    timeframe: str = "today 12-m",
) -> None:
    """Fetch trends and plot with sponsorship events overlay."""
    if keywords is None:
        keywords = ["Emirates", "Qatar Airways"]

    events_df = load_sponsorship_events(events_csv)
    trends_df = fetch_google_trends(keywords, timeframe=timeframe)
    if trends_df.empty:
        raise RuntimeError("Google Trends returned no data.")
    if list(trends_df.columns) == list(keywords):
        pass
    else:
        trends_df.columns = list(keywords)
    plot_trends_with_events(trends_df, events_df, PATHS.reports_figures / "trends_with_events.png")