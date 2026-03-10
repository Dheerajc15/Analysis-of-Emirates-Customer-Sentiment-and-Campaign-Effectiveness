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

PATHS = Paths()

TARGET_AIRLINES = ["Emirates", "Qatar Airways", "Etihad Airways"]

RATING_COLS = [
    "SeatComfortRating",
    "ServiceRating",
    "FoodRating",
    "GroundServiceRating",
    "ValueRating",
]

NEWS_API_KEY_ENV = "NEWS_API_KEY"

def get_news_api_key() -> str | None:
    return os.getenv(NEWS_API_KEY_ENV)