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

EMIRATES_FLEET_URL = "https://en.wikipedia.org/wiki/Emirates_(airline)"


def scrape_emirates_fleet() -> pd.DataFrame:
    """
    Scrape Emirates fleet data from Wikipedia.
    Returns DataFrame with columns: [aircraft, in_service, orders, passengers, notes]
    Falls back to data/scraped_inputs/fleet_fallback.json if scraping fails.
    """
    LOGGER.info("Scraping Emirates fleet data from Wikipedia...")

    try:
        resp = requests.get(EMIRATES_FLEET_URL, headers=SCRAPE_HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(random.uniform(*SCRAPE_DELAY))
    except requests.RequestException as exc:
        LOGGER.warning("Could not fetch Wikipedia page: %s. Using fallback data.", exc)
        return _fallback_fleet_data()

    soup = BeautifulSoup(resp.text, "lxml")
    fleet_df = _parse_fleet_table(soup)

    if fleet_df.empty:
        LOGGER.warning("Could not parse fleet table from Wikipedia. Using fallback.")
        return _fallback_fleet_data()

    LOGGER.info("Scraped fleet data: %d aircraft types", len(fleet_df))
    return fleet_df


def _parse_fleet_table(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse the fleet wikitable from the Wikipedia page."""
    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        header_text = " ".join(headers)

        if "aircraft" in header_text and ("service" in header_text or "order" in header_text):
            rows = []
            for tr in table.find_all("tr")[1:]:  
                cells = tr.find_all(["td", "th"])
                if len(cells) >= 2:
                    rows.append([cell.get_text(strip=True) for cell in cells])

            if rows:
                header_cells = table.find_all("tr")[0].find_all("th")
                col_names = [h.get_text(strip=True) for h in header_cells]

                max_cols = max(len(col_names), max(len(r) for r in rows))
                col_names = col_names + [f"col_{i}" for i in range(len(col_names), max_cols)]
                rows = [r + [""] * (max_cols - len(r)) for r in rows]

                df = pd.DataFrame(rows, columns=col_names[:max_cols])
                df.columns = [c.lower().strip() for c in df.columns]
                return df

    return pd.DataFrame()


def _fallback_fleet_data() -> pd.DataFrame:
    """
    Load fleet fallback data from data/scraped_inputs/fleet_fallback.json.
    Edit that file to update or extend the fallback records — no code changes needed.
    """
    json_path: Path = PATHS.scraped_inputs / "fleet_fallback.json"

    if not json_path.exists():
        LOGGER.error("Fleet fallback file not found: %s", json_path)
        return pd.DataFrame()

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        LOGGER.info("Loaded %d fleet records from %s", len(data), json_path.name)
        return pd.DataFrame(data)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.error("Failed to load fleet fallback: %s", exc)
        return pd.DataFrame()


def get_fleet_summary(fleet_df: pd.DataFrame) -> dict:
    """Compute summary statistics from fleet data."""
    fleet_df = fleet_df.copy()

    if "in_service" in fleet_df.columns:
        fleet_df["in_service_num"] = pd.to_numeric(
            fleet_df["in_service"], errors="coerce"
        ).fillna(0)
        total_active = int(fleet_df["in_service_num"].sum())
    else:
        total_active = "N/A"

    if "orders" in fleet_df.columns:
        fleet_df["orders_num"] = pd.to_numeric(
            fleet_df["orders"], errors="coerce"
        ).fillna(0)
        total_orders = int(fleet_df["orders_num"].sum())
    else:
        total_orders = "N/A"

    summary = {
        "total_active_aircraft": total_active,
        "total_pending_orders": total_orders,
        "aircraft_types": len(fleet_df),
        "as_of": "March 2026",
    }

    LOGGER.info("Fleet Summary: %s", summary)
    return summary