from __future__ import annotations

import pandas as pd
import numpy as np
from collections import Counter

from models.topic_model import split_by_overall_score, run_lda_topics
from utils.logging import get_logger

LOGGER = get_logger(__name__)

# Curated category keywords for mapping LDA topics → actionable labels
_PRAISE_KEYWORDS = {
    "In-flight Entertainment": ["entertainment", "ife", "screen", "movie", "film", "tv"],
    "Cabin Crew Service": ["crew", "staff", "steward", "attendant", "cabin", "friendly", "helpful", "polite"],
    "Seat Comfort": ["seat", "comfort", "legroom", "spacious", "recline", "space", "bed"],
    "Food & Beverage": ["food", "meal", "drink", "wine", "dining", "catering", "menu", "breakfast", "lunch", "dinner"],
    "Lounge Experience": ["lounge", "business", "first", "class", "premium", "chauffeur"],
    "On-time Performance": ["time", "schedule", "punctual", "early", "ontime"],
    "Booking & Check-in": ["booking", "check", "checkin", "online", "app", "website", "smooth"],
    "Aircraft Quality": ["aircraft", "plane", "a380", "boeing", "777", "new", "clean", "modern"],
    "Value for Money": ["value", "price", "worth", "money", "cheap", "affordable", "reasonable"],
    "Overall Experience": ["excellent", "amazing", "best", "great", "wonderful", "fantastic", "outstanding"],
}

_COMPLAINT_KEYWORDS = {
    "Flight Delays & Cancellations": ["delay", "cancel", "late", "hour", "wait", "reschedule", "delayed"],
    "Baggage Issues": ["baggage", "luggage", "bag", "lost", "damaged", "missing", "suitcase"],
    "Poor Customer Service": ["rude", "unhelpful", "ignore", "worst", "terrible", "horrible", "complaint"],
    "Seat Discomfort": ["uncomfortable", "cramped", "narrow", "broken", "hard", "old"],
    "Food Quality": ["cold", "tasteless", "bland", "stale", "inedible", "poor food"],
    "Refund & Compensation": ["refund", "compensation", "voucher", "reimburse", "credit"],
    "Transfer & Connection": ["transfer", "connection", "transit", "missed", "layover"],
    "Cabin Cleanliness": ["dirty", "unclean", "stain", "filthy", "hygiene"],
    "Communication": ["communication", "inform", "update", "email", "response", "phone"],
    "Value for Money": ["overpriced", "expensive", "ripoff", "not worth"],
}


def _assign_category(
    topic_words: str, keyword_map: dict[str, list[str]]
) -> str:
    """Match topic words against keyword map; return best category or 'General'."""
    words = set(topic_words.lower().split())
    best_cat = "General"
    best_score = 0
    for category, keywords in keyword_map.items():
        score = len(words.intersection(keywords))
        if score > best_score:
            best_score = score
            best_cat = category
    return best_cat


def extract_top_praises_and_complaints(
    df_emirates: pd.DataFrame,
    n_topics: int = 5,
    n_words: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract top-5 praising points and top-5 complaint areas from Emirates reviews.

    Returns:
        (praise_df, complaint_df) each with columns:
            ['rank', 'category', 'top_words', 'weight']
    """
    neg, pos = split_by_overall_score(
        df_emirates, low_threshold=4, high_threshold=7
    )

    LOGGER.info(
        "Praise analysis: %d positive reviews | Complaint analysis: %d negative reviews",
        len(pos),
        len(neg),
    )

    # --- Praise topics ---
    praise_topics = run_lda_topics(
        pos["review_clean"],
        n_topics=n_topics,
        n_words=n_words,
        vectorizer="tfidf",
        ngram_range=(1, 2),
    )

    praise_rows = []
    seen_cats = set()
    for t in sorted(praise_topics, key=lambda x: x["weight"], reverse=True):
        cat = _assign_category(t["top_words"], _PRAISE_KEYWORDS)
        if cat in seen_cats:
            cat = f"{cat} (variant)"
        seen_cats.add(cat)
        praise_rows.append(
            {
                "category": cat,
                "top_words": t["top_words"],
                "weight": t["weight"],
            }
        )
    praise_df = pd.DataFrame(praise_rows)
    praise_df = praise_df.head(5).reset_index(drop=True)
    praise_df.insert(0, "rank", range(1, len(praise_df) + 1))

    # --- Complaint topics ---
    complaint_topics = run_lda_topics(
        neg["review_clean"],
        n_topics=n_topics,
        n_words=n_words,
        vectorizer="tfidf",
        ngram_range=(1, 2),
    )

    complaint_rows = []
    seen_cats = set()
    for t in sorted(complaint_topics, key=lambda x: x["weight"], reverse=True):
        cat = _assign_category(t["top_words"], _COMPLAINT_KEYWORDS)
        if cat in seen_cats:
            cat = f"{cat} (variant)"
        seen_cats.add(cat)
        complaint_rows.append(
            {
                "category": cat,
                "top_words": t["top_words"],
                "weight": t["weight"],
            }
        )
    complaint_df = pd.DataFrame(complaint_rows)
    complaint_df = complaint_df.head(5).reset_index(drop=True)
    complaint_df.insert(0, "rank", range(1, len(complaint_df) + 1))

    LOGGER.info("Top 5 Praises:\n%s", praise_df.to_string(index=False))
    LOGGER.info("Top 5 Complaints:\n%s", complaint_df.to_string(index=False))

    return praise_df, complaint_df