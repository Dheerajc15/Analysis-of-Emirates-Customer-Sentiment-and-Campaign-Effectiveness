"""
Run the complete end-to-end analysis:
  1. Review sentiment pipeline (VADER + Pre-trained + evaluation)
  2. Top-5 praises & complaints
  3. Web scraping (trends + fleet + routes)
  4. Save all tables & generate all figures
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import fields

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from pipeline import (
    run_review_pipeline,
    run_scraping_pipeline,
    save_tables,
    make_figures,
    PipelineOutputs,
)
from config import PATHS


def merge_outputs(a: PipelineOutputs, b: PipelineOutputs) -> PipelineOutputs:
    """Merge two PipelineOutputs, preferring non-empty values from each."""
    import pandas as pd

    merged = PipelineOutputs()
    for f in fields(PipelineOutputs):
        val_a = getattr(a, f.name)
        val_b = getattr(b, f.name)

        if isinstance(val_a, pd.DataFrame):
            setattr(merged, f.name, val_a if not val_a.empty else val_b)
        elif isinstance(val_a, dict):
            setattr(merged, f.name, val_a if val_a else val_b)
        elif isinstance(val_a, str):
            setattr(merged, f.name, val_a if val_a != "vader" else val_b)
        else:
            setattr(merged, f.name, val_a if val_a else val_b)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Emirates analysis pipeline.")
    parser.add_argument(
        "--reviews",
        type=str,
        default=str(PATHS.data_raw / "AirlineReviews.csv"),
    )
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║   EMIRATES FULL ANALYSIS PIPELINE                        ║")
    print("╚" + "═" * 58 + "╝")

    # Part 1: Review analysis
    print("\n▶ PART 1: Review Sentiment Analysis...")
    review_outputs = run_review_pipeline(Path(args.reviews))

    # Part 2: Scraping
    print("\n▶ PART 2: Web Scraping (Trends + Fleet + Routes)...")
    scraping_outputs = run_scraping_pipeline()

    # Merge & save
    combined = merge_outputs(review_outputs, scraping_outputs)
    # Ensure review-pipeline winner is preserved
    combined.winner = review_outputs.winner
    combined.eval_results = review_outputs.eval_results

    save_tables(combined)
    make_figures(combined)

    print("\n" + "═" * 60)
    print("✅ FULL ANALYSIS COMPLETE")
    print("═" * 60)
    print(f"  Sentiment model winner : {combined.winner.upper()}")
    print(f"  Fleet summary          : {combined.fleet_summary}")
    print(f"  Routes compiled        : {len(combined.routes_df)} entries")
    print(f"  Sponsorships tracked   : EK={len(combined.ek_sponsors)}, QR={len(combined.qr_sponsors)}")
    print(f"  All outputs in         : {PATHS.reports_figures} & {PATHS.reports_tables}")


if __name__ == "__main__":
    main()