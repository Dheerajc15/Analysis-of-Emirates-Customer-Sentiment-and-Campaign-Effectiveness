# Project Audit

## Objective
Measure Emirates customer sentiment and contextualize marketing impact using:
- review sentiment + topic discovery
- competitor benchmarking
- external signals (Google Trends + News sentiment)

## Data
- AirlineReviews.csv (reviews + ratings + date)
- emirates_sponsorships.csv (event timeline; optional)
- Google Trends API (pytrends)
- NewsAPI (headlines/descriptions)

## Methods
- NLP preprocessing (tokenize, stopwords, lemmatize)
- VADER sentiment scoring
- LDA topic modeling (pain vs praise)
- Comparative visuals (sentiment distribution, service ratings, sentiment over time)
- Event overlay on Google Trends (exploratory)

## Outputs
- Processed datasets in `data/processed/`
- Summary tables in `reports/tables/`
- Figures in `reports/figures/`

## Gaps
- Replace hardcoded paths + remove API keys from code
- Improve topic model stability (CountVectorizer + n-grams + custom stopwords)
- Quantify sponsorship effect (pre/post windows, control comparison)

## Recommended improvements
- Add CI-free “Run All” notebook
- Add tests for preprocessing + pipeline sanity
- Add a small sample dataset for demo (if license allows)
