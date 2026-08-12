"""Cleaning and preprocessing steps for the Netflix titles dataset.

The pipeline follows the assignment requirements:
    1. Load the dataset with Pandas.
    2. Inspect missing values, duplicates and data types.
    3. Handle missing values (drop/fill/replace).
    4. Remove / correct duplicate records.
    5. Convert data types.
    6. Use NumPy for numerical transformations/calculations.
    7. Use Pandas for filtering, sorting and grouping.
    8. Summary statistics and visual insights.
    Bonus: null heatmap, correlation matrix, label encoding for ML-readiness.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]


# --------------------------------------------------------------------------- #
# Step 1: Loading
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_cols}"
        )
    return df


# --------------------------------------------------------------------------- #
# Step 2: Inspection
# --------------------------------------------------------------------------- #
def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of missing values per column."""
    total = df.isnull().sum()
    percent = (total / len(df) * 100).round(2)
    table = pd.DataFrame({"missing_count": total, "percent": percent})
    return table[table["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def duplicate_count(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """Count duplicate rows (optionally considering only a subset of columns)."""
    return int(df.duplicated(subset=subset).sum())


def data_type_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of columns and their data types."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.notnull().sum(),
            "null_count": df.isnull().sum(),
        }
    )
    return report


# --------------------------------------------------------------------------- #
# Step 3: Missing values
# --------------------------------------------------------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column-specific missing-value strategies.

    - director/cast/country/date_added/rating/duration/description: unknown
      fill ("Unknown" / None for description) instead of dropping rows.
    - No columns are dropped wholesale; a few fully-empty rows are removed.
    """
    cleaned = df.copy()

    # A handful of rows are entirely empty (all values NaN) - drop them.
    before = len(cleaned)
    cleaned = cleaned.dropna(how="all")
    print(f"[missing] Dropped {before - len(cleaned)} fully-empty rows.")

    cleaned["director"] = cleaned["director"].fillna("Unknown")
    cleaned["cast"] = cleaned["cast"].fillna("Unknown")
    cleaned["country"] = cleaned["country"].fillna("Unknown")
    cleaned["rating"] = cleaned["rating"].fillna("Unknown")
    cleaned["duration"] = cleaned["duration"].fillna("Unknown")
    cleaned["listed_in"] = cleaned["listed_in"].fillna("Unknown")
    # date_added is filled by forward fill; remaining get a placeholder epoch.
    cleaned["date_added"] = cleaned["date_added"].ffill().fillna("Not available")
    cleaned["description"] = cleaned["description"].fillna("No description")
    return cleaned


# --------------------------------------------------------------------------- #
# Step 4: Duplicates
# --------------------------------------------------------------------------- #
def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows, keeping the first occurrence."""
    cleaned = df.copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=subset, keep="first")
    print(f"[duplicates] Removed {before - len(cleaned)} duplicate rows.")
    return cleaned


# --------------------------------------------------------------------------- #
# Step 5: Data-type conversion
# --------------------------------------------------------------------------- #
def parse_date_added(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date_added (e.g. 'September 25, 2021') to a datetime column."""
    cleaned = df.copy()
    cleaned["date_added_dt"] = pd.to_datetime(
        cleaned["date_added"].replace("Not available", np.nan),
        errors="coerce",
    )
    cleaned["added_year"] = cleaned["date_added_dt"].dt.year
    cleaned["added_month"] = cleaned["date_added_dt"].dt.month
    return cleaned


def extract_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Split duration ('90 min' / '2 Seasons') into numeric value + unit."""
    cleaned = df.copy()
    parts = cleaned["duration"].str.extract(r"(\d+)\s*(min|Seasons|seasons)", expand=True)
    cleaned["duration_value"] = pd.to_numeric(parts[0], errors="coerce")
    cleaned["duration_unit"] = parts[1].str.lower().fillna("unknown")
    cleaned["is_movie"] = (cleaned["type"] == "Movie").astype(int)
    cleaned["is_tv_show"] = (cleaned["type"] == "TV Show").astype(int)
    return cleaned


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric conversions and categorical types."""
    cleaned = df.copy()
    cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
    cleaned["type"] = cleaned["type"].astype("category")
    cleaned["rating"] = cleaned["rating"].astype("category")
    return cleaned


# --------------------------------------------------------------------------- #
# Step 6: NumPy numerical transformations
# --------------------------------------------------------------------------- #
def numpy_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Numerical transformations computed with NumPy.

    - Mean-impute any residual NaN in duration_value.
    - Create numpy arrays for aggregate calculations and percentiles.
    - Add a log-scaled duration feature for better model numerics.
    """
    cleaned = df.copy()

    mean_duration = float(np.nanmean(cleaned["duration_value"]))
    cleaned["duration_value"] = cleaned["duration_value"].fillna(mean_duration)

    # Array-based percentiles across release year and duration.
    years = cleaned["release_year"].to_numpy()
    durations = cleaned["duration_value"].to_numpy()
    for label, arr in [("release_year", years), ("duration_value", durations)]:
        p25, p50, p75 = np.nanpercentile(arr, [25, 50, 75])
        print(
            f"[numpy] {label}: p25={p25:.1f} median={p50:.1f} p75={p75:.1f} "
            f"std={np.nanstd(arr):.2f}"
        )

    cleaned["duration_log"] = np.log1p(cleaned["duration_value"])
    return cleaned


# --------------------------------------------------------------------------- #
# Step 7: Filtering, sorting, grouping
# --------------------------------------------------------------------------- #
def filter_movies_after_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter to Movies released strictly after the given year."""
    return df[(df["type"] == "Movie") & (df["release_year"] > year)].copy()


def sort_by_release_year(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Sort by release year (default: newest first)."""
    return df.sort_values("release_year", ascending=ascending).copy()


def group_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Group by country and compute counts + mean release year."""
    agg = {
        "title_count": ("title", "count"),
        "avg_release_year": ("release_year", "mean"),
    }
    if "duration_value" in df.columns:
        agg["avg_duration_value"] = ("duration_value", "mean")
    return (
        df.groupby("country", observed=True)
        .agg(**agg)
        .sort_values("title_count", ascending=False)
    )


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for numerical columns."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.describe().T


# --------------------------------------------------------------------------- #
# Bonus: visualisations
# --------------------------------------------------------------------------- #
def plot_null_heatmap(df: pd.DataFrame, out_path: str) -> str:
    """Save a heatmap of missing values."""
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_correlation_matrix(df: pd.DataFrame, out_path: str) -> str:
    """Save a correlation heatmap of numerical features."""
    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Fields")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# --------------------------------------------------------------------------- #
# Bonus: ML-readiness encoding
# --------------------------------------------------------------------------- #
def label_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply simple label encoding to the given categorical columns."""
    encoded = df.copy()
    for column in columns:
        codes, _ = pd.factorize(encoded[column], sort=False)
        encoded[f"{column}_encoded"] = codes
    return encoded


def top_country_encoding(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Replace country with top-N one-hot columns (smaller feature space).

    Returns a copy with `country_encoded` (label) and `country_topN` columns.
    """
    encoded = df.copy()
    counts = encoded["country"].value_counts()
    top = counts.head(top_n).index
    encoded["country_encoded"] = np.where(
        encoded["country"].isin(top), encoded["country"], "Other"
    )
    encoded = pd.get_dummies(
        encoded, columns=["country_encoded"], prefix="country"
    )
    return encoded


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    input_path: str,
    output_path: str,
    figures_dir: str,
) -> Dict:
    """Execute the complete cleaning/preprocessing pipeline."""
    print(f"Step 1: Loading dataset from {input_path}")
    df = load_data(input_path)
    print(f"  Shape: {df.shape}")

    print("\nStep 2: Inspecting data")
    print(missing_values_table(df))
    print(f"  Duplicate rows: {duplicate_count(df)}")
    print(data_type_report(df))

    print("\nStep 3: Handling missing values")
    df = handle_missing_values(df)

    print("\nStep 4: Removing duplicates")
    df = remove_duplicates(df)

    print("\nStep 5: Converting data types")
    df = parse_date_added(df)
    df = extract_duration(df)
    df = convert_types(df)

    print("\nStep 6: NumPy transformations")
    df = numpy_transformations(df)

    print("\nStep 7: Filtering, sorting, grouping")
    recent_movies = filter_movies_after_year(df, 2018)
    sorted_df = sort_by_release_year(df)
    by_country = group_by_country(df)
    print(by_country.head(10))

    print("\nSummary statistics")
    print(summary_statistics(df))

    print("\nBonus: visualisations")
    os.makedirs(figures_dir, exist_ok=True)
    null_plot = plot_null_heatmap(df, os.path.join(figures_dir, "null_heatmap.png"))
    corr_plot = plot_correlation_matrix(
        df, os.path.join(figures_dir, "correlation_matrix.png")
    )
    print(f"  Saved: {null_plot}")
    print(f"  Saved: {corr_plot}")

    print("\nBonus: ML-readiness encoding")
    df = label_encode(df, ["type", "rating"])
    df = top_country_encoding(df, top_n=15)
    print(f"  Encoded columns added; final shape: {df.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

    return {
        "input_shape": df.shape,
        "cleaned_path": output_path,
        "null_plot": null_plot,
        "corr_plot": corr_plot,
    }
