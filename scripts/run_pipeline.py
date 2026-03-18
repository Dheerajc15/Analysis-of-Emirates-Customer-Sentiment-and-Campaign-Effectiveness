from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from pipeline import run_review_pipeline, save_tables, make_figures
from config import PATHS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Emirates sentiment review pipeline (VADER + Pre-trained + Evaluation + Praises/Complaints)."
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default=str(PATHS.data_raw / "AirlineReviews.csv"),
    )
    args = parser.parse_args()

    outputs = run_review_pipeline(Path(args.reviews))
    save_tables(outputs)
    make_figures(outputs)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Winner model: {outputs.winner.upper()}")
    print(f"  Processed data  → {PATHS.data_processed}")
    print(f"  Tables          → {PATHS.reports_tables}")
    print(f"  Figures         → {PATHS.reports_figures}")


if __name__ == "__main__":
    main()