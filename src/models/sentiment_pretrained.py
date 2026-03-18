from __future__ import annotations

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import PRETRAINED_MODEL_NAME, PATHS
from utils.logging import get_logger

LOGGER = get_logger(__name__)

# Label mapping for cardiffnlp/twitter-roberta-base-sentiment-latest
_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
_SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def _load_model(
    model_name: str = PRETRAINED_MODEL_NAME,
):
    """Load tokenizer + model, caching locally."""
    cache_dir = PATHS.models_cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading pre-trained model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=str(cache_dir)
    )
    model.eval()
    return tokenizer, model


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = 32,
) -> list[dict]:
    """Return list of {'label': str, 'score': float, 'confidence': float}."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        for j in range(len(batch)):
            label = _LABEL_MAP[preds[j].item()]
            confidence = probs[j][preds[j]].item()
            results.append(
                {
                    "label": label,
                    "score": _SCORE_MAP[label],
                    "confidence": confidence,
                }
            )

    return results


def add_pretrained_sentiment(
    df: pd.DataFrame,
    text_col: str = "Review",
    out_label_col: str = "pretrained_label",
    out_score_col: str = "pretrained_score",
    out_conf_col: str = "pretrained_confidence",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Add pre-trained transformer sentiment columns to the DataFrame."""
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found.")

    df = df.copy()
    texts = df[text_col].fillna("").astype(str).tolist()

    tokenizer, model = _load_model()
    preds = predict_batch(texts, tokenizer, model, batch_size=batch_size)

    df[out_label_col] = [p["label"] for p in preds]
    df[out_score_col] = [p["score"] for p in preds]
    df[out_conf_col] = [p["confidence"] for p in preds]

    LOGGER.info("Pre-trained sentiment scored %d reviews", len(df))
    return df