from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import run_scraping_pipeline, save_tables, make_figures
from config import PATHS


def main() -> None:
    print("=" * 60)
    print("EMIRATES SCRAPING PIPELINE")
    print("  • Search Trends (2022-2025) — Emirates vs Qatar Airways")
    print("  • Fleet Analysis (till March 2026)")
    print("  • Top Routes To/From DXB")
    print("=" * 60)

    outputs = run_scraping_pipeline()
    save_tables(outputs)
    make_figures(outputs)

    print("\n" + "=" * 60)
    print("SCRAPING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Fleet summary: {outputs.fleet_summary}")
    print(f"  Routes: {len(outputs.routes_df)} entries")
    print(f"  Trends: {len(outputs.trends_df)} monthly data points")
    print(f"  Tables  → {PATHS.reports_tables}")
    print(f"  Figures → {PATHS.reports_figures}")


if __name__ == "__main__":
    main()