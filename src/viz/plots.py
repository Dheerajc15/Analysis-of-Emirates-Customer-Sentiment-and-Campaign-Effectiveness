from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import RATING_COLS

# ── Colour palette ─────────────────────────────────────────────────────────
EMIRATES_RED = "#D71921"
QATAR_BURGUNDY = "#5C0632"
NEUTRAL_GREY = "#888888"


# ═══════════════════════════════════════════════════════════════════════════
# 1) SENTIMENT PLOTS (existing, refined)
# ═══════════════════════════════════════════════════════════════════════════


def plot_sentiment_distribution(
    df_rivals: pd.DataFrame, out_path: str | Path | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="AirlineName",
        y="sentiment_score",
        data=df_rivals,
        order=["Emirates", "Qatar Airways", "Etihad Airways"],
        palette=[EMIRATES_RED, QATAR_BURGUNDY, NEUTRAL_GREY],
        ax=ax,
    )
    ax.set_title("Sentiment Score Distribution (Emirates vs. Rivals)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Sentiment Score (-1 to +1)")
    ax.set_xlabel("Airline")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


def plot_service_ratings(
    df_rivals: pd.DataFrame, out_path: str | Path | None = None
) -> pd.DataFrame:
    df = df_rivals.copy()
    for col in RATING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_cols = [col for col in RATING_COLS if col in df.columns]
    if not available_cols:
        raise ValueError(f"None of the expected rating columns found. Expected: {RATING_COLS}")

    df_avg = df.groupby("AirlineName")[available_cols].mean().reset_index()
    df_melted = df_avg.melt(
        id_vars="AirlineName", var_name="ServiceCategory", value_name="AverageRating"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x="ServiceCategory", y="AverageRating", hue="AirlineName",
        data=df_melted, ax=ax,
        palette={"Emirates": EMIRATES_RED, "Qatar Airways": QATAR_BURGUNDY, "Etihad Airways": NEUTRAL_GREY},
    )
    ax.set_title("Comparative Analysis of Service Ratings", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average Rating (1-5)")
    ax.set_xlabel("Service Category")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Airline", loc="lower right")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)
    return df_avg


def plot_sentiment_over_time(
    df_rivals: pd.DataFrame, out_path: str | Path | None = None
) -> pd.DataFrame:
    df = df_rivals.copy()
    if "DatePub" not in df.columns:
        raise KeyError("Expected 'DatePub' column not found.")
    df["DatePub"] = pd.to_datetime(df["DatePub"], errors="coerce")
    df = df.dropna(subset=["DatePub"])

    df_time = (
        df.groupby([pd.Grouper(key="DatePub", freq="ME"), "AirlineName"])["sentiment_score"]
        .mean()
        .reset_index()
    )
    df_time = df_time[df_time["DatePub"] > "2015-01-01"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        x="DatePub", y="sentiment_score", hue="AirlineName",
        data=df_time, marker="o", ax=ax
    )
    ax.set_title("Average Customer Sentiment Over Time (Monthly)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average Sentiment Score")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)
    return df_time


# ═══════════════════════════════════════════════════════════════════════════
# 2) MODEL EVALUATION PLOT (NEW)
# ═══════════════════════════════════════════════════════════════════════════


def plot_model_comparison(
    eval_results: dict, out_path: str | Path | None = None
) -> None:
    """Bar chart comparing VADER vs Pre-trained model metrics."""
    metrics = ["accuracy", "weighted_f1", "cohens_kappa"]
    vader_vals = [eval_results["vader"][m] for m in metrics]
    pt_vals = [eval_results["pretrained"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, vader_vals, width, label="VADER", color="#4285F4", edgecolor="black")
    bars2 = ax.bar(x + width / 2, pt_vals, width, label="Pre-trained (RoBERTa)", color=EMIRATES_RED, edgecolor="black")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sentiment Model Evaluation: VADER vs Pre-trained", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Weighted F1", "Cohen's Kappa"], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add value labels
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

    # Star the winner
    winner = eval_results["winner"]
    ax.text(
        0.5, 0.95, f"★ Winner: {winner.upper()}",
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8),
    )

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3) TOP-5 PRAISES & COMPLAINTS PLOTS (NEW)
# ═══════════════════════════════════════════════════════════════════════════


def plot_top_praises(
    praise_df: pd.DataFrame, out_path: str | Path | None = None
) -> None:
    """Horizontal bar chart of top-5 praising points."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Greens_r", len(praise_df))
    bars = ax.barh(
        praise_df["category"], praise_df["weight"],
        color=colors, edgecolor="black"
    )
    ax.set_xlabel("Topic Weight", fontsize=12)
    ax.set_title("✅ Top 5 Praising Points – What Emirates Should MAINTAIN", fontsize=16, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, words in zip(bars, praise_df["top_words"]):
        short = " | ".join(words.split()[:5]) + "..."
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                short, va="center", fontsize=9, style="italic", color="#333")

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


def plot_top_complaints(
    complaint_df: pd.DataFrame, out_path: str | Path | None = None
) -> None:
    """Horizontal bar chart of top-5 complaint areas."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Reds_r", len(complaint_df))
    bars = ax.barh(
        complaint_df["category"], complaint_df["weight"],
        color=colors, edgecolor="black"
    )
    ax.set_xlabel("Topic Weight", fontsize=12)
    ax.set_title("⚠️ Top 5 Complaints – What Emirates NEEDS TO WORK ON", fontsize=16, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, words in zip(bars, complaint_df["top_words"]):
        short = " | ".join(words.split()[:5]) + "..."
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                short, va="center", fontsize=9, style="italic", color="#333")

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 4) SEARCH TRENDS PLOTS (NEW)
# ═══════════════════════════════════════════════════════════════════════════


def plot_search_trends(
    trends_df: pd.DataFrame,
    ek_sponsors: pd.DataFrame,
    qr_sponsors: pd.DataFrame,
    out_path: str | Path | None = None,
) -> None:
    """Plot Emirates vs Qatar Airways search trends with sponsorship event markers."""
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(trends_df.index, trends_df["Emirates"], color=EMIRATES_RED,
            linewidth=2, label="Emirates Search Interest")
    ax.plot(trends_df.index, trends_df["Qatar Airways"], color=QATAR_BURGUNDY,
            linewidth=2, linestyle="--", label="Qatar Airways Search Interest")

    ax.fill_between(trends_df.index, trends_df["Emirates"],
                    alpha=0.1, color=EMIRATES_RED)
    ax.fill_between(trends_df.index, trends_df["Qatar Airways"],
                    alpha=0.1, color=QATAR_BURGUNDY)

    ax.set_title("Search Trend Analysis: Emirates vs Qatar Airways (2022-2025)\nWith Sponsorship & Endorsement Events",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Relative Search Interest", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


def plot_sponsorship_comparison(
    ek_sponsors: pd.DataFrame,
    qr_sponsors: pd.DataFrame,
    out_path: str | Path | None = None,
) -> None:
    """Grouped bar chart: sponsorship count by year for Emirates vs Qatar Airways."""
    ek_counts = ek_sponsors.groupby("year").size().reset_index(name="count")
    ek_counts["airline"] = "Emirates"
    qr_counts = qr_sponsors.groupby("year").size().reset_index(name="count")
    qr_counts["airline"] = "Qatar Airways"
    combined = pd.concat([ek_counts, qr_counts], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="year", y="count", hue="airline", data=combined,
        palette={"Emirates": EMIRATES_RED, "Qatar Airways": QATAR_BURGUNDY},
        edgecolor="black", ax=ax,
    )
    ax.set_title("Sponsorship & Endorsement Deals per Year (2022-2025)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Sponsorship Deals", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5) FLEET & ROUTES PLOTS (NEW)
# ═══════════════════════════════════════════════════════════════════════════


def plot_fleet_composition(
    fleet_df: pd.DataFrame, out_path: str | Path | None = None
) -> None:
    """Pie + bar chart showing Emirates fleet composition."""
    df = fleet_df.copy()
    df["in_service_num"] = pd.to_numeric(df["in_service"], errors="coerce").fillna(0)
    active = df[df["in_service_num"] > 0].copy()

    if active.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart - active fleet
    colors = sns.color_palette("Set2", len(active))
    wedges, texts, autotexts = ax1.pie(
        active["in_service_num"], labels=active["aircraft"],
        autopct="%1.0f%%", colors=colors, startangle=90,
        pctdistance=0.85, textprops={"fontsize": 10}
    )
    ax1.set_title("Active Fleet Composition", fontsize=14, fontweight="bold")

    # Bar chart - active + orders
    df["orders_num"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0)
    x = range(len(df))
    ax2.barh(df["aircraft"], df["in_service_num"], color=EMIRATES_RED,
             edgecolor="black", label="In Service", alpha=0.9)
    ax2.barh(df["aircraft"], df["orders_num"], left=df["in_service_num"],
             color="#FFD700", edgecolor="black", label="On Order", alpha=0.7)
    ax2.set_xlabel("Number of Aircraft", fontsize=12)
    ax2.set_title("Fleet: In Service vs On Order", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle("Emirates Fleet Analysis (as of March 2026)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)


def plot_top_routes(
    routes_df: pd.DataFrame, out_path: str | Path | None = None
) -> None:
    """Visualize top-5 routes from DXB and top-5 routes to DXB."""
    from_dxb = routes_df[routes_df["direction"] == "from DXB"].head(5)
    to_dxb = routes_df[routes_df["direction"] == "to DXB"].head(5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # From DXB
    ax1.barh(from_dxb["route"], from_dxb["daily_flights"],
             color=EMIRATES_RED, edgecolor="black")
    ax1.set_xlabel("Daily Flights", fontsize=12)
    ax1.set_title("Top 5 Routes FROM Dubai (DXB)", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for i, (flights, dist) in enumerate(zip(from_dxb["daily_flights"], from_dxb["distance_km"])):
        ax1.text(flights + 0.1, i, f"{dist:,} km", va="center", fontsize=9)

    # To DXB
    ax2.barh(to_dxb["route"], to_dxb["daily_flights"],
             color="#FFD700", edgecolor="black")
    ax2.set_xlabel("Daily Flights", fontsize=12)
    ax2.set_title("Top 5 Routes TO Dubai (DXB)", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for i, (flights, dist) in enumerate(zip(to_dxb["daily_flights"], to_dxb["distance_km"])):
        ax2.text(flights + 0.1, i, f"{dist:,} km", va="center", fontsize=9)

    fig.suptitle("Emirates Top Routes To & From DXB", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)