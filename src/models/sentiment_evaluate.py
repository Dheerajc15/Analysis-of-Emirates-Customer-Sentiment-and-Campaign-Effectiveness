from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    cohen_kappa_score,
)

from models.sentiment_vader import vader_label
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def _derive_ground_truth(
    df: pd.DataFrame,
    score_col: str = "OverallScore",
) -> pd.Series:
    """
    Derive ground-truth sentiment labels from OverallScore:
      - OverallScore <= 4   → 'negative'
      - OverallScore 5-6    → 'neutral'
      - OverallScore >= 7   → 'positive'
    """
    s = pd.to_numeric(df[score_col], errors="coerce")

    def _map(val):
        if pd.isna(val):
            return np.nan
        if val <= 4:
            return "negative"
        elif val <= 6:
            return "neutral"
        else:
            return "positive"

    return s.apply(_map)


def evaluate_models(
    df: pd.DataFrame,
    vader_score_col: str = "vader_score",
    pretrained_label_col: str = "pretrained_label",
    overall_score_col: str = "OverallScore",
) -> dict:
    """
    Compare VADER vs Pre-trained model against ground truth derived from OverallScore.
    Returns dict with metrics and recommendation of which model to use.
    """
    df = df.copy()

    # Derive ground truth
    df["ground_truth"] = _derive_ground_truth(df, overall_score_col)
    df = df.dropna(subset=["ground_truth"])

    # VADER labels
    df["vader_label"] = df[vader_score_col].apply(vader_label)

    # --- VADER metrics ---
    vader_acc = accuracy_score(df["ground_truth"], df["vader_label"])
    vader_f1 = f1_score(
        df["ground_truth"], df["vader_label"], average="weighted", zero_division=0
    )
    vader_kappa = cohen_kappa_score(df["ground_truth"], df["vader_label"])
    vader_report = classification_report(
        df["ground_truth"], df["vader_label"], zero_division=0, output_dict=True
    )

    # --- Pre-trained metrics ---
    pt_acc = accuracy_score(df["ground_truth"], df[pretrained_label_col])
    pt_f1 = f1_score(
        df["ground_truth"],
        df[pretrained_label_col],
        average="weighted",
        zero_division=0,
    )
    pt_kappa = cohen_kappa_score(df["ground_truth"], df[pretrained_label_col])
    pt_report = classification_report(
        df["ground_truth"], df[pretrained_label_col], zero_division=0, output_dict=True
    )

    # --- Determine winner ---
    # Primary criterion: weighted F1; tiebreaker: Cohen's kappa
    if pt_f1 > vader_f1:
        winner = "pretrained"
    elif vader_f1 > pt_f1:
        winner = "vader"
    else:
        winner = "pretrained" if pt_kappa >= vader_kappa else "vader"

    results = {
        "vader": {
            "accuracy": round(vader_acc, 4),
            "weighted_f1": round(vader_f1, 4),
            "cohens_kappa": round(vader_kappa, 4),
            "classification_report": vader_report,
        },
        "pretrained": {
            "accuracy": round(pt_acc, 4),
            "weighted_f1": round(pt_f1, 4),
            "cohens_kappa": round(pt_kappa, 4),
            "classification_report": pt_report,
        },
        "winner": winner,
    }

    LOGGER.info("═" * 50)
    LOGGER.info("SENTIMENT MODEL EVALUATION")
    LOGGER.info("═" * 50)
    LOGGER.info(
        "VADER       → Accuracy: %.4f | Weighted-F1: %.4f | Kappa: %.4f",
        vader_acc,
        vader_f1,
        vader_kappa,
    )
    LOGGER.info(
        "Pre-trained → Accuracy: %.4f | Weighted-F1: %.4f | Kappa: %.4f",
        pt_acc,
        pt_f1,
        pt_kappa,
    )
    LOGGER.info("═" * 50)
    LOGGER.info("✅ WINNER: %s", winner.upper())
    LOGGER.info("═" * 50)

    return results


def assign_best_sentiment(
    df: pd.DataFrame,
    winner: str,
    vader_score_col: str = "vader_score",
    pretrained_score_col: str = "pretrained_score",
    pretrained_label_col: str = "pretrained_label",
    out_col: str = "sentiment_score",
    out_label_col: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Assign the winning model's scores as the canonical 'sentiment_score'
    and 'sentiment_label' columns for downstream analysis.
    """
    df = df.copy()
    if winner == "pretrained":
        df[out_col] = df[pretrained_score_col]
        df[out_label_col] = df[pretrained_label_col]
        LOGGER.info("Using PRE-TRAINED model scores for further analysis.")
    else:
        df[out_col] = df[vader_score_col]
        df[out_label_col] = df[vader_score_col].apply(vader_label)
        LOGGER.info("Using VADER model scores for further analysis.")
    return df