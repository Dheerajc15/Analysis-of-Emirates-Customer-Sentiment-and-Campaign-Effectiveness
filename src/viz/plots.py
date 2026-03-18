from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RATING_COLS


def plot_sentiment_distribution(df_rivals: pd.DataFrame, out_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x="AirlineName",
        y="sentiment_score",
        data=df_rivals,
        order=["Emirates", "Qatar Airways", "Etihad Airways"],
        ax=ax,
    )
    ax.set_title("Sentiment Score Distribution (Emirates vs. Rivals)", fontsize=16)
    ax.set_ylabel("VADER Compound Score (-1 to +1)")
    ax.set_xlabel("Airline")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig) 


def plot_service_ratings(df_rivals: pd.DataFrame, out_path: str | Path | None = None) -> pd.DataFrame:
    df = df_rivals.copy()
    for col in RATING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_cols = [col for col in RATING_COLS if col in df.columns]
    if not available_cols:
        raise ValueError(
            "None of the expected rating columns were found in the DataFrame. "
            f"Expected one or more of: {RATING_COLS}"
        )

    df_avg = df.groupby("AirlineName")[available_cols].mean().reset_index()
    df_melted = df_avg.melt(
        id_vars="AirlineName",
        var_name="ServiceCategory",
        value_name="AverageRating",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="ServiceCategory", y="AverageRating", hue="AirlineName", data=df_melted, ax=ax)
    ax.set_title("Comparative Analysis of Service Ratings", fontsize=16)
    ax.set_ylabel("Average Rating (1-5)")
    ax.set_xlabel("Service Category")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Airline", loc="lower right")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)  
    return df_avg


def plot_sentiment_over_time(df_rivals: pd.DataFrame, out_path: str | Path | None = None) -> pd.DataFrame:
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

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x="DatePub", y="sentiment_score", hue="AirlineName", data=df_time, marker="o", ax=ax)
    ax.set_title("Average Customer Sentiment Over Time (Monthly)", fontsize=16)
    ax.set_ylabel("Average VADER Sentiment Score")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.6)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)  
    return df_time


def plot_trends_with_events(
    trends_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_path: str | Path | None = None,
) -> None:
    
    if trends_df.empty:
        print("Warning: trends_df is empty, skipping plot_trends_with_events.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Date")
    ax.set_ylabel("Google Search Interest")

    if "Emirates" in trends_df.columns:
        emir_col = "Emirates"
        qatar_col = "Qatar Airways" if "Qatar Airways" in trends_df.columns else None
    else:
        # handle renamed columns
        emir_col = (
            "Emirates_Interest"
            if "Emirates_Interest" in trends_df.columns
            else trends_df.columns[0]
        )
        qatar_col = (
            "Qatar_Interest"
            if "Qatar_Interest" in trends_df.columns
            else (trends_df.columns[1] if len(trends_df.columns) > 1 else None)
        )

    ax.plot(trends_df.index, trends_df[emir_col], label="Emirates Search Interest")
    if qatar_col and qatar_col in trends_df.columns:
        ax.plot(trends_df.index, trends_df[qatar_col], linestyle=":", label="Control Interest")

    fig.canvas.draw()

    for _, row in events_df.iterrows():
        ax.axvline(x=row["Date"], color="red", linestyle="--", linewidth=1.2)
        ax.text(
            row["Date"],
            ax.get_ylim()[1] * 0.9,
            str(row["Event"]),
            rotation=90,
            ha="right",
            va="top",
            color="red",
        )

    ax.set_title("Public Interest vs. Sponsorship Events", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig) 