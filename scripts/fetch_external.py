from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from emirates_sentiment.pipeline import run_external_signals
from emirates_sentiment.config import PATHS

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch external signals (Google Trends + News sentiment).")
    parser.add_argument("--out", type=str, default=str(PATHS.data_processed / "external_signals.csv"))
    args = parser.parse_args()

    trends_df, news_sent = run_external_signals()
    if trends_df.empty:
        print("No Google Trends data returned.")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        trends_df.to_csv(out)
        print(f"Saved trends to {out}")

    print(f"Average News sentiment: {news_sent:.3f}")

if __name__ == "__main__":
    main()
