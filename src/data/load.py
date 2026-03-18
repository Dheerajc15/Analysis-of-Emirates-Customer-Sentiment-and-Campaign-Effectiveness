from __future__ import annotations

from pathlib import Path
import pandas as pd
from config import TARGET_AIRLINES
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_reviews(csv_path: str | Path) -> pd.DataFrame:
    """Load the airline reviews CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reviews CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    LOGGER.info("Loaded %d reviews from %s", len(df), csv_path.name)
    return df


def filter_airlines(
    df: pd.DataFrame, airlines: list[str] = TARGET_AIRLINES
) -> pd.DataFrame:
    """Filter to target airlines only."""
    if "AirlineName" not in df.columns:
        raise KeyError("Expected column 'AirlineName' not found in reviews dataset.")
    filtered = df[df["AirlineName"].isin(airlines)].copy()
    LOGGER.info("Filtered to %d reviews for %s", len(filtered), airlines)
    return filtered


def split_emirates(
    df_rivals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_rivals_df, emirates_only_df)."""
    df_emirates = df_rivals[df_rivals["AirlineName"] == "Emirates"].copy()
    LOGGER.info("Emirates subset: %d reviews", len(df_emirates))
    return df_rivals, df_emirates