from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from emirates_sentiment.pipeline import run_review_pipeline, save_tables, make_figures
from emirates_sentiment.config import PATHS

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Emirates sentiment review pipeline.")
    parser.add_argument("--reviews", type=str, default=str(PATHS.data_raw / "AirlineReviews.csv"))
    args = parser.parse_args()

    outputs = run_review_pipeline(Path(args.reviews))
    save_tables(outputs)
    make_figures(outputs)

    print("\nSaved outputs:")
    print(f"- {PATHS.data_processed/'rivals_scored.csv'}")
    print(f"- {PATHS.data_processed/'emirates_scored.csv'}")
    print(f"- {PATHS.reports_tables/'sentiment_by_airline.csv'}")
    print(f"- figures in {PATHS.reports_figures}")

if __name__ == "__main__":
    main()
