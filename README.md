# Analysis of Emirates Customer Sentiment and Campaign Effectiveness

This repo is structured like an **industry / portfolio** project:
- reusable Python modules in `src/`
- reproducible scripts in `scripts/`
- a clean report notebook in `notebooks/02_report.ipynb`
- outputs saved to `data/processed/` and `reports/`

## Project Audit
See: `reports/project_audit.md`

## Folder structure
```
data/raw/            # put AirlineReviews.csv + emirates_sponsorships.csv here (not committed)
data/processed/      # generated artifacts
notebooks/           # report notebook(s)
src/                 # reusable modules (pipeline, models, plots, etc.)
scripts/             # command-line runners
reports/figures/     # saved charts
reports/tables/      # saved summary tables
```

## Setup
```bash
python -m venv .venv
# activate your venv, then:
pip install -r requirements.txt
pip install -e .
```

Create `.env` from the template:
```bash
cp .env.example .env
# then set NEWS_API_KEY in .env
```

## Run
1) Put your datasets in `data/raw/`:
- `AirlineReviews.csv`
- `emirates_sponsorships.csv` (optional, for event overlay)

2) Run the pipeline:
```bash
python scripts/run_pipeline.py --reviews data/raw/AirlineReviews.csv
```

3) Open the portfolio notebook:
- `notebooks/02_report.ipynb`
