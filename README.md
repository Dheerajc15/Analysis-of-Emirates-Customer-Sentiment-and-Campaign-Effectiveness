# ✈️ Analysis of Emirates Customer Sentiment & Campaign Effectiveness

> **End-to-end NLP and data analytics project** evaluating Emirates' competitive position, customer experience drivers, fleet strategy, and digital brand visibility against Qatar Airways and Etihad Airways.

---

## 🎯 Project Objective

The global airline industry is highly competitive, with customer experience and brand visibility being critical differentiators. This project aims to:

- **Quantify customer sentiment** for Emirates, Qatar Airways and Etihad Airways using state-of-the-art NLP models applied to **129,455 airline reviews**
- **Identify the top 5 praise points and pain points** specific to Emirates using data-driven topic extraction — eliminating manual keyword guessing
- **Benchmark Emirates' service quality** across 5 sub-dimensions (Seat Comfort, Cabin Service, Food & Beverage, Ground Service, Value for Money) against its two primary rivals
- **Analyse fleet composition and route strategy** to contextualise operational capabilities against customer feedback
- **Measure campaign effectiveness** by correlating sponsorship deal activity with Google Search Interest trends for Emirates vs Qatar Airways from 2022–2025

The project is designed to produce **actionable business intelligence** that can directly inform marketing strategy, service investment priorities, and competitive positioning decisions.

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| **Data Manipulation** | `pandas >= 2.0`, `numpy >= 1.24` |
| **NLP — Rule-based** | `vaderSentiment >= 3.3.2`, `nltk >= 3.8` |
| **NLP — Transformer** | `transformers >= 4.36` (HuggingFace), `torch >= 2.1` |
| **Pretrained Model** | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **ML Evaluation** | `scikit-learn >= 1.3` (F1, accuracy, Cohen's Kappa, TF-IDF) |
| **Web Scraping** | `requests >= 2.31`, `beautifulsoup4 >= 4.12`, `lxml >= 5.1` |
| **Visualisation** | `matplotlib >= 3.7`, `seaborn >= 0.13` |
| **Configuration** | `python-dotenv >= 1.0` |
| **Notebook** | `ipykernel >= 6.29`, Jupyter Notebook |

---

## ⚙️ Implementations

### 1. Data Pipeline & Text Preprocessing
- Loaded and filtered **129,455 airline reviews** from a structured CSV, reducing to **6,126 reviews** across Emirates, Qatar Airways and Etihad Airways
- Applied regex-based text cleaning, lowercasing, stopword removal and lemmatisation via `nltk` to produce a `review_clean` column used across all downstream models

### 2. Dual Sentiment Model Evaluation
- Scored all reviews using **two independent NLP models in parallel**:
  - **VADER** (rule-based lexicon model, fast, no GPU required)
  - **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) — a transformer fine-tuned on 124 million tweets, producing 3-class classification (positive / neutral / negative)
- Evaluated both models head-to-head using **Accuracy, Weighted F1-Score and Cohen's Kappa** against ground-truth `OverallScore` labels
- Automatically selected the winning model and assigned its predictions as the canonical `sentiment_label` and `sentiment_score` columns across both DataFrames — ensuring all downstream analysis is driven by the best-performing model

### 3. Data-Driven Praise & Complaint Extraction
- Replaced a prior hardcoded keyword lookup system with a fully **data-driven TF-IDF pipeline**:
  - Reviews split into positive/negative pools using the winning model's `sentiment_label`
  - **TF-IDF vectorisation** (`max_features=5000`, bigrams) applied to each pool independently
  - Category relevance scored by summing TF-IDF weights of matching n-grams across all reviews — no manual category assignment
  - Category keyword maps externalised to `data/scraped_inputs/category_keywords.json`, making them fully configurable without touching Python source code

### 4. Web Scraping Pipeline
Three independent scrapers collect live data with JSON fallbacks to ensure reproducibility:
- **`fleet_scraper.py`** — scrapes Emirates fleet data (aircraft types, in-service counts, order book) from Wikipedia with structured HTML parsing via `BeautifulSoup`
- **`routes_scraper.py`** — extracts top-frequency routes to/from Dubai International Airport (DXB) including distance (km) and daily flight counts
- **`search_trends.py`** — retrieves monthly Google Search Interest data for Emirates vs Qatar Airways (2022–2025) and loads sponsorship deal records from versioned JSON config files

### 5. Competitive EDA & Visualisation
Nine publication-quality visualisations produced across the analysis:
- Sentiment score distribution boxplot (all 3 airlines)
- Comparative service ratings grouped bar chart (5 dimensions × 3 airlines)
- Service ratings heatmap (colour-coded competitive positioning)
- Monthly sentiment trend line chart (2015–2023, with COVID-19 period annotation)
- Top 5 praise categories horizontal bar chart (TF-IDF weighted)
- Top 5 complaint categories horizontal bar chart (TF-IDF weighted)
- Fleet composition pie chart + in-service vs on-order bar chart
- Top 5 routes to/from DXB bar chart with distance annotations
- Search trend line chart (2022–2026) + sponsorship deal count grouped bar chart

---

## 📊 Results & Key Findings

### Sentiment & Service Ratings

| Metric | Emirates | Qatar Airways | Etihad Airways |
|---|---|---|---|
| Sentiment Distribution | IQR: 0.0 to +1.0 | IQR: +0.05 to +1.0 | IQR: −1.0 to 0.0 |
| Seat Comfort (avg/5) | 3.31 | **3.93** | 2.60 |
| Cabin Service (avg/5) | 3.03 | **4.21** | 2.68 |
| Food & Beverage (avg/5) | 3.02 | **3.81** | 2.45 |
| Ground Service (avg/5) | 2.13 | **3.00** | 1.65 |
| Value for Money (avg/5) | 3.14 | **3.98** | 2.58 |

- Qatar Airways leads across **all five service dimensions**; Emirates leads Etihad in all categories
- **Ground Service is the weakest-rated dimension** across the industry — no airline exceeds 3.0/5
- Sentiment hit its lowest point for all three airlines during **COVID-19 (2020)**, with Emirates experiencing the steepest drop (approaching −1.0)

### Top 5 Praises — What Emirates Should MAINTAIN ✅

| Rank | Category | TF-IDF Topic Weight |
|---|---|---|
| 1 | Cabin Crew Service | ~0.31 |
| 2 | Food & Beverage | ~0.28 |
| 3 | Overall Experience | ~0.26 |
| 4 | Lounge Experience | ~0.24 |
| 5 | Seat Comfort | ~0.24 |

### Top 5 Complaints — What Emirates NEEDS TO WORK ON ⚠️

| Rank | Category | TF-IDF Topic Weight |
|---|---|---|
| 1 | Flight Delays & Cancellations | ~0.205 |
| 2 | Seat Discomfort | ~0.100 |
| 3 | Poor Customer Service | ~0.080 |
| 4 | Baggage Issues | ~0.060 |
| 5 | Transfer & Connection | ~0.055 |

> Flight Delays dominates at **2× the weight** of the second complaint — the single highest-priority operational improvement area.

### Fleet & Routes

- Active fleet: **Boeing 777-300ER (48%)** + **Airbus A380-800 (42%)** = 90% of total
- Order book: **205 × Boeing 777-9** + **60 × Airbus A350-900** + **35 × Boeing 787-9** — one of the largest in commercial aviation
- Highest-frequency route: **DXB ↔ LHR at 6 daily flights** (5,467 km)
- **Mumbai (BOM)** appears in both top-5 outbound and inbound lists — highest-frequency short-haul corridor

### Campaign Effectiveness (2022–2025)

| Year | Emirates Deals | Qatar Airways Deals |
|---|---|---|
| 2022 | **6** | 3 |
| 2023 | **5** | 4 |
| 2024 | 4 | 4 (tied) |
| 2025 | 3 | 3 (tied) |

- Emirates maintained **higher Google Search Interest in every single month** from 2022–2026 (average lead: ~15–20 points)
- Qatar Airways' closest approach to Emirates was **January 2023** (gap: ~9 points), driven by FIFA World Cup 2022 residual interest
- Deal count parity in 2024–2025 signals **Qatar Airways is closing the sponsorship gap** — a strategic risk for Emirates' brand visibility lead

---

## 💡 Key Takeaways

### For Emirates — Business Recommendations
1. **Operational reliability is the #1 priority**: Flight delays dominate the complaint landscape at 2× the weight of any other complaint. Investment in predictive delay management and proactive passenger communication tools would directly address the top three complaints simultaneously (delays → poor service → refunds).
2. **Protect the soft product**: Cabin Crew, Food & Beverage and Lounge Experience are the top three praise categories and Emirates' most defensible competitive advantages. Cost reductions in these areas would directly erode the attributes customers value most.
3. **Close the Qatar Airways service gap**: Qatar Airways leads by an average of **0.85 rating points** across all five service sub-dimensions. The biggest gaps are in Cabin Service (−1.18) and Value for Money (−0.84) — both addressable through targeted investment.
4. **Renew anchor sponsorships before expiry**: With deal counts now tied with Qatar Airways, Emirates' search visibility advantage depends on the legacy reach of existing long-term partnerships (Emirates Stadium, Real Madrid, F1). These must be prioritised for renewal to prevent search interest convergence.

## 📁 Project Structure

```
Analysis-of-Emirates-Customer-Sentiment-and-Campaign-Effectiveness/
│
├── 📓 notebooks/
│   └── Final Report.ipynb          # Main analysis notebook (9 visualisations)
│
├── 📦 src/
│   ├── config.py                   # Central config: paths, target airlines, rating cols
│   ├── pipeline.py                 # Orchestrates review + scraping pipelines end-to-end
│   │
│   ├── data/
│   │   └── load.py                 # CSV loader, airline filter, Emirates split
│   │
│   ├── features/
│   │   └── text_preprocess.py      # Regex cleaning, stopword removal, lemmatisation
│   │
│   ├── models/
│   │   ├── sentiment_vader.py      # VADER sentiment scoring
│   │   ├── sentiment_pretrained.py # RoBERTa transformer scoring (HuggingFace)
│   │   ├── sentiment_evaluate.py   # Head-to-head evaluation, winner selection
│   │   └── topic_model.py          # LDA topic model + score-based split utilities
│   │
│   ├── analysis/
│   │   └── praise_complaints.py    # TF-IDF praise/complaint extraction (data-driven)
│   │
│   ├── scraping/
│   │   ├── fleet_scraper.py        # Emirates fleet data (Wikipedia + JSON fallback)
│   │   ├── routes_scraper.py       # Top routes to/from DXB (+ JSON fallback)
│   │   └── search_trends.py        # Google Trends + sponsorship data loader
│   │
│   ├── viz/
│   │   └── plots.py                # All 9 visualisation functions
│   │
│   └── utils/
│       ├── logging.py              # Structured logger factory
│       └── nltk_setup.py           # One-time NLTK resource downloader
│
├── 📂 data/
│   ├── raw/                        # AirlineReviews.csv (gitignored — add manually)
│   ├── processed/                  # Pipeline outputs: scored CSVs (gitignored)
│   └── scraped_inputs/             # ✅ Versioned JSON config files
│       ├── category_keywords.json  # Praise/complaint category keyword maps
│       ├── emirates_sponsorships.json
│       ├── qatar_sponsorships.json
│       ├── fleet_fallback.json
│       └── routes_fallback.json
│
├── 📂 reports/
│   ├── figures/                    # Saved visualisation PNGs
│   └── tables/                     # Saved analysis CSVs
│
├── 📂 scripts/                     # Utility/one-off execution scripts
├── requirements.txt                # All Python dependencies with version pins
├── pyproject.toml                  # Project metadata
└── .gitignore
```
