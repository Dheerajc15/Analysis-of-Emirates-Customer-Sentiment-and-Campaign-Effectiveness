# Emirates Customer Sentiment & Campaign Effectiveness (NLP + External Signals)

This project analyzes **customer review sentiment for Emirates** and benchmarks it against key competitors, then **contextualizes brand buzz** using external signals (Google Trends + News sentiment) and a sponsorship/event timeline.

The repo is structured in an **industry / portfolio format**:
- reusable modules in `src/`
- reproducible CLI runners in `scripts/`
- a clean “report notebook” in `notebooks/02_report.ipynb`
- artifacts written to `data/processed/` and `reports/`

---

## Objective

1) **Measure Emirates customer sentiment** from reviews (NLP preprocessing + VADER sentiment).  
2) **Benchmark sentiment vs competitors** (Qatar Airways, Etihad Airways).  
3) **Discover themes** driving positive vs negative experiences (topic modeling).  
4) **Connect marketing/campaign context** to public interest (Google Trends + event overlay) and “live” brand sentiment (NewsAPI).

---

## Data

### Core datasets
- **`AirlineReviews.csv`** (reviews + airline name + ratings + date)
- **`emirates_sponsorships.csv`** (optional event timeline for “campaign overlay”)

### External signals (optional)
- **Google Trends** via `pytrends` (search interest over time)
- **News sentiment** via NewsAPI (headline + description sentiment)

> Raw datasets are expected in `data/raw/` (typically not committed).

---

## Methods

### 1) Data ingestion + filtering
- Load `AirlineReviews.csv`
- Filter to target airlines: **Emirates, Qatar Airways, Etihad Airways**
- Split Emirates vs “rivals” for benchmarking + Emirates-only deep dives

**Implementation:** `src/emirates_sentiment/data/load.py`

---

### 2) Text preprocessing (NLP cleaning)
Each review is transformed into a cleaned text field (`review_clean`) using:
- lowercasing
- removing punctuation / digits
- tokenization (NLTK)
- stopword removal
- lemmatization (WordNet)

**Implementation:** `src/emirates_sentiment/features/text_preprocess.py`

---

### 3) Sentiment scoring (VADER)
- Score each cleaned review with **VADER compound sentiment**
- Create competitive summary table: **average sentiment by airline**
- Generate visuals:
  - sentiment distribution (by airline)
  - sentiment over time (if dates exist)
  - service rating comparison (if rating columns exist)

**Implementation:**  
- sentiment scoring: `src/emirates_sentiment/models/sentiment.py`  
- plots: `src/emirates_sentiment/viz/plots.py`

---

### 4) Topic modeling (What drives pain vs praise?)
- Segment Emirates reviews into “negative” vs “positive” buckets using `OverallScore`
- Run **LDA topic modeling** using TF-IDF/CountVectorizer (configurable)
- Print top terms per topic to surface recurring themes (service, comfort, delays, etc.)

**Implementation:** `src/emirates_sentiment/models/topic_model.py`

---

### 5) Campaign effectiveness context (external signals)
- Google Trends interest for `["Emirates", "Qatar Airways"]` (control comparison)
- Event overlay using `emirates_sponsorships.csv`
- News sentiment using NewsAPI (headline + description sentiment)

**Implementation:**  
- trends: `src/emirates_sentiment/external/trends.py`  
- news: `src/emirates_sentiment/external/news.py`  
- event loader: `src/emirates_sentiment/data/events.py`

---

## Outputs

After running the pipeline, artifacts are saved to:

### Processed data
- `data/processed/rivals_scored.csv`
- `data/processed/emirates_scored.csv`

### Tables
- `reports/tables/sentiment_by_airline.csv`

### Figures
- `reports/figures/sentiment_distribution.png`
- `reports/figures/service_ratings.png`
- `reports/figures/sentiment_over_time.png`
- `reports/figures/trends_with_events.png` 

---

## Repo Structure

```txt
data/
  raw/                 # AirlineReviews.csv + emirates_sponsorships.csv (not committed)
  processed/            # generated outputs

notebooks/
  02_report.ipynb       # portfolio-grade report
  Emirates_legacy.ipynb  # original legacy notebook

reports/
  figures/              # saved charts
  tables/               # saved tables
  project_audit.md       # Objective → Data → Methods → Outputs → Gaps → Improvements

scripts/
  run_pipeline.py        # end-to-end review pipeline (tables + figures)
  fetch_external.py      # Google Trends + News sentiment

src/emirates_sentiment/
  pipeline.py            # orchestration (calls modules below)
  config.py              # paths, constants
  data/                  # loaders
  features/              # preprocessing
  models/                # sentiment + topic model
  external/              # trends + news
  viz/                   # plotting helpers
  utils/                 # logging + nltk setup
