# Explanation — Clean and Preprocess a Dataset (Pandas + NumPy)

## What this is
A reproducible data-cleaning and preprocessing pipeline (Assignment 2) that turns the raw
**Netflix Movies & TV Shows** CSV into an analysis- and ML-ready dataset.

## What it does
- Loads the CSV with Pandas and inspects missing values, duplicates, and data types
- Fills / drops missing values and removes duplicate records
- Fixes data types (date parsing, `90 min` / `2 Seasons` → numeric duration)
- Applies NumPy transformations (percentiles, std, mean-imputation, a log feature)
- Filters, sorts, and groups with Pandas
- Bonus: null-heatmap, correlation matrix, label encoding + one-hot ML-readiness

## Structure
- `scripts/run_cleaning.py` — end-to-end runner
- `scripts/make_notebook.py` — builds + executes the notebook
- `src/` — cleaning functions
- `data/raw/`, `data/processed/` — input and cleaned output
- `notebooks/data_cleaning_preprocessing.ipynb` — executed deliverable
- `reports/figures/` — null heatmap, correlation matrix, etc.
- `tests/` — 17 unit tests

## How to run
```bash
cd projects/02-data-cleaning-preprocessing
python scripts/run_cleaning.py
```

## Key result
The Netflix dataset goes from messy raw rows to a clean, typed, de-duplicated dataset with
documented every-step decisions — the standard "data prep" step before any modeling.
