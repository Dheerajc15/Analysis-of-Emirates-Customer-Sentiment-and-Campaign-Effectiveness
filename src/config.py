from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    reports_figures: Path = PROJECT_ROOT / "reports" / "figures"
    reports_tables: Path = PROJECT_ROOT / "reports" / "tables"
    models_cache: Path = PROJECT_ROOT / "models_cache"


PATHS = Paths()

TARGET_AIRLINES = ["Emirates", "Qatar Airways", "Etihad Airways"]

RATING_COLS = [
    "SeatComfortRating",
    "ServiceRating",
    "FoodRating",
    "GroundServiceRating",
    "ValueRating",
]

# Pre-trained sentiment model from HuggingFace
PRETRAINED_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Web scraping settings
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
SCRAPE_DELAY = (1, 3)  