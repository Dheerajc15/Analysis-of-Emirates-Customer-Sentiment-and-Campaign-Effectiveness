from __future__ import annotations

from pathlib import Path
import pandas as pd
from config import TARGET_AIRLINES
from utils.logging import get_logger

LOGGER = get_logger(__name__)

def load_reviews(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reviews CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def filter_airlines(df: pd.DataFrame, airlines: list[str] = TARGET_AIRLINES) -> pd.DataFrame:
    if "AirlineName" not in df.columns:
        raise KeyError("Expected column 'AirlineName' not found in reviews dataset.")
    return df[df["AirlineName"].isin(airlines)].copy()

def split_emirates(df_rivals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_emirates = df_rivals[df_rivals["AirlineName"] == "Emirates"].copy()
    return df_rivals, df_emirates
