from __future__ import annotations

from typing import Sequence
import pandas as pd
from pytrends.request import TrendReq

def fetch_google_trends(
    keywords: Sequence[str],
    timeframe: str = "today 12-m",
    hl: str = "en-US",
    tz: int = 360,
) -> pd.DataFrame:
    """Fetch Google Trends interest over time for keywords."""
    try:
        pytrends = TrendReq(hl=hl, tz=tz)
        pytrends.build_payload(list(keywords), cat=0, timeframe=timeframe, geo="", gprop="")
        df = pytrends.interest_over_time()
    except Exception as e:
        print(f"Warning: Could not fetch Google Trends data — {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df