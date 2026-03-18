# ✈️ Emirates Customer Sentiment & Campaign Effectiveness

## Project Overview

This project performs a comprehensive analysis of **Emirates** covering:

1. **Customer Sentiment Analysis** — using VADER and a pre-trained RoBERTa transformer, evaluated head-to-head, with the better model used for all downstream analysis
2. **Top-5 Praises & Complaints** — actionable insights on what Emirates should maintain and improve
3. **Search Trend Analysis (2022–2025)** — Emirates vs Qatar Airways sponsorship/endorsement effectiveness via web scraping
4. **Fleet Analysis** — Emirates fleet composition as of March 2026 (web-scraped from Wikipedia)
5. **Route Analysis** — Top-5 most frequent routes to and from DXB

---

## Repo Structure

```txt
├── data/
│   ├── raw/                          # AirlineReviews.csv (not committed)
│   └── processed/                    # Generated CSVs
│
├── notebooks/
│   └── Final Report.ipynb            # Portfolio report notebook
│
├── reports/
│   ├── figures/                      # All generated charts (10+ visualizations)
│   └── tables/                       # All generated CSV tables
│
├── scripts/
│   ├── run_pipeline.py               # Sentiment analysis pipeline
│   ├── run_scraper.py                # Trends + fleet + routes scraper
│   └── run_full_analysis.py          # End-to-end runner (everything)
│
└── src/
    ├── config.py                     # Paths, constants, model config
    ├── pipeline.py                   # Orchestration
    ├── data/load.py                  # CSV loaders
    ├── features/text_preprocess.py   # NLP text cleaning
    ├── models/
    │   ├── sentiment_vader.py        # VADER sentiment
    │   ├── sentiment_pretrained.py   # RoBERTa transformer sentiment
    │   ├── sentiment_evaluate.py     # Head-to-head evaluation
    │   └── topic_model.py            # LDA topic modeling
    ├── analysis/
    │   └── praise_complaints.py      # Top-5 praises & complaints
    ├── scraping/
    │   ├── search_trends.py          # Google Trends + sponsorship data
    │   ├── fleet_scraper.py          # Emirates fleet from Wikipedia
    │   └── routes_scraper.py         # Top routes to/from DXB
    ├── viz/plots.py                  # All visualizations
    └── utils/                        # Logging + NLTK setup
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Dheerajc15/Analysis-of-Emirates-Customer-Sentiment-and-Campaign-Effectiveness.git
cd Analysis-of-Emirates-Customer-Sentiment-and-Campaign-Effectiveness
pip install -r requirements.txt

# 2. Place data
# Put AirlineReviews.csv in data/raw/

# 3. Run everything
python scripts/run_full_analysis.py

# Or run individually:
python scripts/run_pipeline.py       # Sentiment analysis only
python scripts/run_scraper.py        # Web scraping only
```

---

## Methods

### Sentiment Analysis
- **VADER** (lexicon-based) — fast, rule-based sentiment
- **cardiffnlp/twitter-roberta-base-sentiment-latest** (transformer) — pre-trained on 124M tweets
- **Evaluation**: Accuracy, Weighted F1, Cohen's Kappa against ground truth derived from OverallScore
- The **winning model** is automatically selected for all downstream analysis

### Topic Modeling
- LDA with TF-IDF vectorization and bigrams
- Separate extraction for positive (praises) and negative (complaints) reviews
- Mapped to actionable business categories

### Web Scraping
- **requests + BeautifulSoup + lxml** (most efficient for static HTML)
- Sources: Wikipedia (fleet, destinations), Google Trends pages
- Rate-limited with random delays to be respectful

---

## Outputs

| Output | Location |
|--------|----------|
| Scored reviews (rivals + Emirates) | `data/processed/` |
| Sentiment by airline table | `reports/tables/sentiment_by_airline.csv` |
| Top-5 praises & complaints | `reports/tables/top5_praises.csv`, `top5_complaints.csv` |
| Sponsorship data | `reports/tables/emirates_sponsorships.csv`, `qatar_sponsorships.csv` |
| Fleet data | `reports/tables/emirates_fleet.csv` |
| Route data | `reports/tables/top_routes_dxb.csv` |
| 10+ visualizations | `reports/figures/` |

---

## Key Visualizations

- Sentiment distribution (boxplot)
- Service ratings comparison (grouped bar)
- Sentiment over time (line)
- **Model comparison** (VADER vs RoBERTa bar chart)
- **Top-5 praises** (horizontal bar + keywords)
- **Top-5 complaints** (horizontal bar + keywords)
- **Search trends 2022-2025** (Emirates vs Qatar Airways)
- **Sponsorship comparison** (grouped bar by year)
- **Fleet composition** (pie + bar)
- **Top routes to/from DXB** (dual horizontal bar)

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for all dependencies
- ~500MB disk space for the RoBERTa model download (cached in `models_cache/`)