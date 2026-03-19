from __future__ import annotations

import json
import time
import random
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import SCRAPE_HEADERS, SCRAPE_DELAY, PATHS
from utils.logging import get_logger

LOGGER = get_logger(__name__)

EMIRATES_DESTINATIONS_URL = "https://en.wikipedia.org/wiki/Emirates_(airline)_destinations"


def scrape_top_routes_from_dxb() -> pd.DataFrame:

    LOGGER.info("Scraping Emirates route data...")

    destinations = _scrape_emirates_destinations()

    if destinations.empty:
        LOGGER.info("Wikipedia scrape returned no data. Using JSON fallback.")
        return _fallback_routes_data()

    return destinations


def _scrape_emirates_destinations() -> pd.DataFrame:
    """Attempt to scrape Emirates destinations table from Wikipedia."""
    try:
        resp = requests.get(
            EMIRATES_DESTINATIONS_URL, headers=SCRAPE_HEADERS, timeout=15
        )
        resp.raise_for_status()
        time.sleep(random.uniform(*SCRAPE_DELAY))
    except requests.RequestException as exc:
        LOGGER.warning("Could not fetch destinations page: %s", exc)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        header_text = " ".join(headers)

        if "country" in header_text or "city" in header_text:
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = tr.find_all(["td", "th"])
                if len(cells) >= 2:
                    rows.append([cell.get_text(strip=True) for cell in cells])

            if rows:
                header_cells = table.find_all("tr")[0].find_all("th")
                col_names = [h.get_text(strip=True) for h in header_cells]
                max_cols = max(len(col_names), max(len(r) for r in rows) if rows else 0)
                col_names = col_names + [f"col_{i}" for i in range(len(col_names), max_cols)]
                rows = [r + [""] * (max_cols - len(r)) for r in rows]
                df = pd.DataFrame(rows, columns=col_names[:max_cols])
                df.columns = [c.lower().strip() for c in df.columns]
                LOGGER.info("Scraped %d destination entries", len(df))
                return df

    return pd.DataFrame()


def _fallback_routes_data() -> pd.DataFrame:
    """
    Load route fallback data from data/scraped_inputs/routes_fallback.json.
    Edit that file to update or extend the fallback routes — no code changes needed.
    """
    json_path: Path = PATHS.scraped_inputs / "routes_fallback.json"

    if not json_path.exists():
        LOGGER.error("Routes fallback file not found: %s", json_path)
        return pd.DataFrame()

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        LOGGER.info("Loaded %d route records from %s", len(data), json_path.name)
        return pd.DataFrame(data)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.error("Failed to load routes fallback: %s", exc)
        return pd.DataFrame()