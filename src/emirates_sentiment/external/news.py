from __future__ import annotations

from typing import Optional
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_news_sentiment(
    query: str,
    api_key: str,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 100,
) -> float:
    """Return average VADER compound sentiment of recent news headlines+descriptions."""
    if not api_key:
        raise ValueError("NewsAPI key missing. Set NEWS_API_KEY in your environment or .env file.")
    newsapi = NewsApiClient(api_key=api_key)
    sia = SentimentIntensityAnalyzer()

    articles = newsapi.get_everything(
        q=query,
        language=language,
        sort_by=sort_by,
        page_size=page_size,
    )
    scores = []
    for a in articles.get("articles", []):
        title = a.get("title") or ""
        desc = a.get("description") or ""
        text = f"{title}. {desc}".strip()
        if text:
            scores.append(sia.polarity_scores(text)["compound"])

    return sum(scores) / len(scores) if scores else 0.0
