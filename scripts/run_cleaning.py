"""End-to-end data cleaning & preprocessing pipeline runner.

Usage:
    python scripts/run_cleaning.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cleaning import run_pipeline  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "netflix_titles.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "netflix_titles_clean.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")


def main() -> None:
    """Run the pipeline."""
    result = run_pipeline(INPUT_PATH, OUTPUT_PATH, FIGURES_DIR)
    print("\nPipeline complete.")
    print(f"Cleaned dataset: {result['cleaned_path']}")
    print(f"Final shape: {result['input_shape']}")


if __name__ == "__main__":
    main()
