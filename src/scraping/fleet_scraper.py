"""
Scrape Emirates fleet information from Wikipedia (most reliable structured source).
Data as of March 2026.
"""
from __future__ import annotations

import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import SCRAPE_HEADERS, SCRAPE_DELAY
from utils.logging import get_logger

LOGGER = get_logger(__name__)

EMIRATES_FLEET_URL = "https://en.wikipedia.org/wiki/Emirates_(airline)"


def scrape_emirates_fleet() -> pd.DataFrame:
    """
    Scrape Emirates fleet data from Wikipedia.
    Returns DataFrame with columns: [aircraft, in_service, orders, passengers, notes]
    """
    LOGGER.info("Scraping Emirates fleet data from Wikipedia...")

    try:
        resp = requests.get(EMIRATES_FLEET_URL, headers=SCRAPE_HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(random.uniform(*SCRAPE_DELAY))
    except requests.RequestException as e:
        LOGGER.warning("Could not fetch Wikipedia page: %s. Using fallback data.", e)
        return _fallback_fleet_data()

    soup = BeautifulSoup(resp.text, "lxml")

    # Find the fleet table — Wikipedia uses "wikitable" class
    fleet_df = _parse_fleet_table(soup)

    if fleet_df.empty:
        LOGGER.warning("Could not parse fleet table from Wikipedia. Using fallback.")
        return _fallback_fleet_data()

    LOGGER.info("Scraped fleet data: %d aircraft types", len(fleet_df))
    return fleet_df


def _parse_fleet_table(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse the fleet table from the Wikipedia page."""
    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        # Look for a table that contains "Aircraft" and "In service" headers
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        header_text = " ".join(headers)

        if "aircraft" in header_text and ("service" in header_text or "order" in header_text):
            rows = []
            for tr in table.find_all("tr")[1:]:  # skip header row
                cells = tr.find_all(["td", "th"])
                if len(cells) >= 2:
                    row = [cell.get_text(strip=True) for cell in cells]
                    rows.append(row)

            if rows:
                # Determine column names from header
                header_cells = table.find_all("tr")[0].find_all("th")
                col_names = [h.get_text(strip=True) for h in header_cells]

                # Pad rows to match header length
                max_cols = max(len(col_names), max(len(r) for r in rows))
                col_names = col_names + [f"col_{i}" for i in range(len(col_names), max_cols)]
                rows = [r + [""] * (max_cols - len(r)) for r in rows]

                df = pd.DataFrame(rows, columns=col_names[:max_cols])

                # Standardize column names
                df.columns = [c.lower().strip() for c in df.columns]
                return df

    return pd.DataFrame()


def _fallback_fleet_data() -> pd.DataFrame:
    """
    Fallback fleet data based on publicly known Emirates fleet (as of March 2026).
    Sources: Emirates.com, Planespotters.net, ch-aviation.com
    """
    data = [
        {
            "aircraft": "Airbus A380-800",
            "in_service": 116,
            "orders": 0,
            "passengers": "489-615",
            "configuration": "3-class / 2-class",
            "notes": "World's largest A380 fleet operator",
        },
        {
            "aircraft": "Boeing 777-300ER",
            "in_service": 133,
            "orders": 0,
            "passengers": "354-428",
            "configuration": "3-class / 2-class",
            "notes": "Backbone of long-haul fleet",
        },
        {
            "aircraft": "Boeing 777-200LR",
            "in_service": 10,
            "orders": 0,
            "passengers": "266",
            "configuration": "2-class",
            "notes": "Ultra-long-range variant",
        },
        {
            "aircraft": "Boeing 777-F",
            "in_service": 11,
            "orders": 0,
            "passengers": "Cargo",
            "configuration": "Freighter",
            "notes": "Emirates SkyCargo dedicated",
        },
        {
            "aircraft": "Boeing 777-9",
            "in_service": 0,
            "orders": 205,
            "passengers": "~400 (est.)",
            "configuration": "TBD",
            "notes": "Largest 777X order globally; deliveries starting ~2026",
        },
        {
            "aircraft": "Boeing 787-9 Dreamliner",
            "in_service": 0,
            "orders": 35,
            "passengers": "~300 (est.)",
            "configuration": "TBD",
            "notes": "First Dreamliner order by Emirates (2024)",
        },
        {
            "aircraft": "Airbus A350-900",
            "in_service": 5,
            "orders": 60,
            "passengers": "312 (est.)",
            "configuration": "3-class",
            "notes": "New type for Emirates; deliveries began late 2024",
        },
    ]
    return pd.DataFrame(data)


def get_fleet_summary(fleet_df: pd.DataFrame) -> dict:
    """Compute summary statistics from fleet data."""
    # Try to get numeric in_service
    if "in_service" in fleet_df.columns:
        fleet_df = fleet_df.copy()
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