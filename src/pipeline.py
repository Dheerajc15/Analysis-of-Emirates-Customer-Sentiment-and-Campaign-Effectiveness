from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from config import PATHS, TARGET_AIRLINES
from data.load import load_reviews, filter_airlines, split_emirates
from features.text_preprocess import add_clean_text
from models.sentiment_vader import add_vader_sentiment, average_sentiment_by_airline
from models.sentiment_pretrained import add_pretrained_sentiment
from models.sentiment_evaluate import evaluate_models, assign_best_sentiment
from models.topic_model import split_by_overall_score, run_lda_topics
from analysis.praise_complaints import extract_top_praises_and_complaints
from scraping.search_trends import scrape_combined_trends, get_sponsorship_data
from scraping.fleet_scraper import scrape_emirates_fleet, get_fleet_summary
from scraping.routes_scraper import scrape_top_routes_from_dxb
from viz.plots import (
    plot_sentiment_distribution,
    plot_service_ratings,
    plot_sentiment_over_time,
    plot_model_comparison,
    plot_top_praises,
    plot_top_complaints,
    plot_search_trends,
    plot_sponsorship_comparison,
    plot_fleet_composition,
    plot_top_routes,
)
from utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PipelineOutputs:
    """Container for all pipeline outputs."""
    df_rivals: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_emirates: pd.DataFrame = field(default_factory=pd.DataFrame)
    sentiment_by_airline: pd.DataFrame = field(default_factory=pd.DataFrame)
    eval_results: dict = field(default_factory=dict)
    winner: str = "vader"
    praise_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    complaint_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trends_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ek_sponsors: pd.DataFrame = field(default_factory=pd.DataFrame)
    qr_sponsors: pd.DataFrame = field(default_factory=pd.DataFrame)
    fleet_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fleet_summary: dict = field(default_factory=dict)
    routes_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Review pipeline (sentiment + evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def run_review_pipeline(reviews_csv: str | Path) -> PipelineOutputs:
    """
    Full review analysis pipeline:
    1. Load & filter reviews
    2. Preprocess text
    3. Score with VADER
    4. Score with pre-trained transformer
    5. Evaluate both models head-to-head
    6. Use the better model for downstream analysis
    7. Extract top-5 praises & complaints
    """
    LOGGER.info("=" * 60)
    LOGGER.info("STEP 1: Loading and preprocessing reviews")
    LOGGER.info("=" * 60)

    df = load_reviews(reviews_csv)
    df_rivals = filter_airlines(df, TARGET_AIRLINES)
    df_rivals, df_emirates = split_emirates(df_rivals)

    # Text preprocessing
    df_rivals = add_clean_text(df_rivals, text_col="Review", out_col="review_clean")
    df_emirates = add_clean_text(df_emirates, text_col="Review", out_col="review_clean")

    LOGGER.info("=" * 60)
    LOGGER.info("STEP 2: Sentiment scoring (VADER)")
    LOGGER.info("=" * 60)
    df_rivals = add_vader_sentiment(df_rivals, text_col="review_clean", out_col="vader_score")
    df_emirates = add_vader_sentiment(df_emirates, text_col="review_clean", out_col="vader_score")

    LOGGER.info("=" * 60)
    LOGGER.info("STEP 3: Sentiment scoring (Pre-trained RoBERTa)")
    LOGGER.info("=" * 60)
    df_rivals = add_pretrained_sentiment(df_rivals, text_col="Review")
    df_emirates = add_pretrained_sentiment(df_emirates, text_col="Review")

    LOGGER.info("=" * 60)
    LOGGER.info("STEP 4: Model evaluation (VADER vs Pre-trained)")
    LOGGER.info("=" * 60)
    eval_results = evaluate_models(df_emirates)
    winner = eval_results["winner"]

    # Assign the winning model's scores
    df_rivals = assign_best_sentiment(df_rivals, winner)
    df_emirates = assign_best_sentiment(df_emirates, winner)

    # Competitive summary
    sentiment_tbl = average_sentiment_by_airline(df_rivals)

    LOGGER.info("=" * 60)
    LOGGER.info("STEP 5: Extracting top-5 praises & complaints")
    LOGGER.info("=" * 60)
    praise_df, complaint_df = extract_top_praises_and_complaints(df_emirates)

    return PipelineOutputs(
        df_rivals=df_rivals,
        df_emirates=df_emirates,
        sentiment_by_airline=sentiment_tbl,
        eval_results=eval_results,
        winner=winner,
        praise_df=praise_df,
        complaint_df=complaint_df,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Scraping pipeline (trends + fleet + routes)
# ═══════════════════════════════════════════════════════════════════════════

def run_scraping_pipeline() -> PipelineOutputs:
    """
    Web scraping pipeline:
    1. Search trend analysis (2022-2025) Emirates vs Qatar Airways
    2. Emirates fleet analysis (till March 2026)
    3. Top-5 most frequent routes to/from DXB
    """
    outputs = PipelineOutputs()

    LOGGER.info("=" * 60)
    LOGGER.info("SCRAPING STEP 1: Search Trends (2022-2025)")
    LOGGER.info("=" * 60)
    outputs.trends_df = scrape_combined_trends(start_year=2022, end_year=2025)
    outputs.ek_sponsors, outputs.qr_sponsors = get_sponsorship_data()

    LOGGER.info("=" * 60)
    LOGGER.info("SCRAPING STEP 2: Emirates Fleet Analysis")
    LOGGER.info("=" * 60)
    outputs.fleet_df = scrape_emirates_fleet()
    outputs.fleet_summary = get_fleet_summary(outputs.fleet_df)

    LOGGER.info("=" * 60)
    LOGGER.info("SCRAPING STEP 3: Top Routes To/From DXB")
    LOGGER.info("=" * 60)
    outputs.routes_df = scrape_top_routes_from_dxb()

    return outputs


# ═══════════════════════════════════════════════════════════════════════════
# SAVE & VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════

def save_tables(outputs: PipelineOutputs) -> None:
    """Save all processed data and tables to disk."""
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    PATHS.reports_tables.mkdir(parents=True, exist_ok=True)

    if not outputs.df_rivals.empty:
        outputs.df_rivals.to_csv(PATHS.data_processed / "rivals_scored.csv", index=False)
    if not outputs.df_emirates.empty:
        outputs.df_emirates.to_csv(PATHS.data_processed / "emirates_scored.csv", index=False)
    if not outputs.sentiment_by_airline.empty:
        outputs.sentiment_by_airline.to_csv(PATHS.reports_tables / "sentiment_by_airline.csv", index=False)
    if not outputs.praise_df.empty:
        outputs.praise_df.to_csv(PATHS.reports_tables / "top5_praises.csv", index=False)
    if not outputs.complaint_df.empty:
        outputs.complaint_df.to_csv(PATHS.reports_tables / "top5_complaints.csv", index=False)
    if not outputs.trends_df.empty:
        outputs.trends_df.to_csv(PATHS.data_processed / "search_trends_2022_2025.csv")
    if not outputs.ek_sponsors.empty:
        outputs.ek_sponsors.to_csv(PATHS.reports_tables / "emirates_sponsorships.csv", index=False)
    if not outputs.qr_sponsors.empty:
        outputs.qr_sponsors.to_csv(PATHS.reports_tables / "qatar_sponsorships.csv", index=False)
    if not outputs.fleet_df.empty:
        outputs.fleet_df.to_csv(PATHS.reports_tables / "emirates_fleet.csv", index=False)
    if not outputs.routes_df.empty:
        outputs.routes_df.to_csv(PATHS.reports_tables / "top_routes_dxb.csv", index=False)

    LOGGER.info("All tables saved to %s and %s", PATHS.data_processed, PATHS.reports_tables)


def make_figures(outputs: PipelineOutputs) -> None:
    """Generate all visualizations."""
    PATHS.reports_figures.mkdir(parents=True, exist_ok=True)
    fig = PATHS.reports_figures

    # Sentiment plots
    if not outputs.df_rivals.empty:
        plot_sentiment_distribution(outputs.df_rivals, out_path=fig / "sentiment_distribution.png")
        plot_service_ratings(outputs.df_rivals, out_path=fig / "service_ratings.png")
        plot_sentiment_over_time(outputs.df_rivals, out_path=fig / "sentiment_over_time.png")

    # Model evaluation
    if outputs.eval_results:
        plot_model_comparison(outputs.eval_results, out_path=fig / "model_comparison.png")

    # Praises & Complaints
    if not outputs.praise_df.empty:
        plot_top_praises(outputs.praise_df, out_path=fig / "top5_praises.png")
    if not outputs.complaint_df.empty:
        plot_top_complaints(outputs.complaint_df, out_path=fig / "top5_complaints.png")

    # Search trends
    if not outputs.trends_df.empty:
        plot_search_trends(
            outputs.trends_df, outputs.ek_sponsors, outputs.qr_sponsors,
            out_path=fig / "search_trends_2022_2025.png",
        )
    if not outputs.ek_sponsors.empty:
        plot_sponsorship_comparison(
            outputs.ek_sponsors, outputs.qr_sponsors,
            out_path=fig / "sponsorship_comparison.png",
        )

    # Fleet & Routes
    if not outputs.fleet_df.empty:
        plot_fleet_composition(outputs.fleet_df, out_path=fig / "fleet_composition.png")
    if not outputs.routes_df.empty:
        plot_top_routes(outputs.routes_df, out_path=fig / "top_routes_dxb.png")

    LOGGER.info("All figures saved to %s", fig)