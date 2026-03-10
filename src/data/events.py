from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_sponsorship_events(csv_path: str | Path) -> pd.DataFrame:
    """Load sponsorship events file with columns: Date, Event."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sponsorship events CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Event" not in df.columns:
        raise KeyError("Sponsorship file must contain columns: 'Date' and 'Event'.")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df
