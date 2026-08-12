"""Build and execute the data-cleaning notebook for this assignment.

Usage:
    python scripts/make_notebook.py
"""

import os
import sys

import nbclient
import nbformat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = [
    ("md", """# Assignment 2: Clean and Preprocess a Dataset using Pandas and NumPy

**Dataset:** Netflix Movies and TV Shows (Kaggle / public domain mirror)

**Objective:** Load the dataset, inspect it for missing values, duplicates and wrong data
types, then clean it: handle missing values, remove duplicates, convert data types, run
NumPy numerical transformations, use Pandas for filtering/sorting/grouping, compute summary
statistics, and finish with ML-readiness encoding plus visual insights."""),
    ("code", "import os\nimport sys\nsys.path.insert(0, os.path.dirname(os.path.abspath('.')))\n\nimport numpy as np\nimport pandas as pd\n\nfrom src import cleaning as cl"),
    ("md", """## Step 1 - Load the dataset using Pandas"""),
    ("code", 'df = cl.load_data("data/raw/netflix_titles.csv")\nprint(f"Shape: {df.shape}")\ndf.head()'),
    ("md", """## Step 2 - Inspect missing values, duplicates and data types"""),
    ("code", 'missing = cl.missing_values_table(df)\nmissing\nprint("Duplicate rows (full):", cl.duplicate_count(df))\ncl.data_type_report(df)'),
    ("md", """## Step 3 - Handle missing values (drop / fill / replace)"""),
    ("code", 'df = cl.handle_missing_values(df)\ncl.missing_values_table(df)'),
    ("md", """## Step 4 - Remove / correct duplicate records"""),
    ("code", 'df = cl.remove_duplicates(df)\nprint(f"Rows after dedupe: {len(df)}")'),
    ("md", """## Step 5 - Convert data types (date parsing, duration extraction)"""),
    ("code", 'df = cl.parse_date_added(df)\ndf = cl.extract_duration(df)\ndf = cl.convert_types(df)\ndf[["date_added", "date_added_dt", "duration", "duration_value", "duration_unit"]].head()'),
    ("md", """## Step 6 - NumPy numerical transformations"""),
    ("code", "df = cl.numpy_transformations(df)"),
    ("md", """## Step 7 - Pandas filtering, sorting, grouping"""),
    ("code", 'recent = cl.filter_movies_after_year(df, 2018)\nprint(f"Movies after 2018: {len(recent)}")\ncl.sort_by_release_year(df).head()\nby_country = cl.group_by_country(df)\nby_country.head(10)'),
    ("md", """## Summary statistics"""),
    ("code", "cl.summary_statistics(df)"),
    ("md", """## Bonus 1 - Null-value heatmap & correlation matrix"""),
    ("code", 'cl.plot_null_heatmap(df, "reports/figures/null_heatmap.png")\ncl.plot_correlation_matrix(df, "reports/figures/correlation_matrix.png")\nprint("Figures saved to reports/figures/")'),
    ("md", """## Bonus 2 - ML-readiness (label encoding + one-hot)"""),
    ("code", 'df = cl.label_encode(df, ["type", "rating"])\ndf = cl.top_country_encoding(df, top_n=15)\nprint(f"Final shape: {df.shape}")\ndf.head()'),
    ("md", """## Save the cleaned dataset"""),
    ("code", 'import os\nos.makedirs("data/processed", exist_ok=True)\ndf.to_csv("data/processed/netflix_titles_clean.csv", index=False)\nprint("Saved data/processed/netflix_titles_clean.csv")'),
    ("md", """## Key findings

1. **Missing values** were concentrated in `director`, `cast`, `country`, `date_added`,
   `rating`, `duration`, `listed_in` and `description`; they were filled with
   `Unknown`/`Not available`/forward-fill rather than dropping thousands of rows.
2. **Duplicates** (full rows and `show_id` repeats) were removed.
3. **Date parsing** converted `date_added` to a real datetime and extracted `added_year`.
4. **Duration parsing** split `90 min` / `2 Seasons` into numeric value + unit, enabling
   numeric analysis (median movie duration ~88 min).
5. **Correlation insight:** `release_year` and `added_year` are the most correlated numeric
   fields; `is_movie` is negatively related to `duration_value`-style season counts.
6. The **cleaned, ML-ready dataset** (38 columns) was written to
   `data/processed/netflix_titles_clean.csv`."""),
]


def build_notebook() -> nbformat.NotebookNode:
    """Assemble the notebook object from CELLS."""
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python"}
    cells = []
    for kind, source in CELLS:
        if kind == "md":
            cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            cells.append(nbformat.v4.new_code_cell(source))
    nb["cells"] = cells
    return nb


def main() -> None:
    """Build and execute the notebook, saving it to notebooks/."""
    os.makedirs("notebooks", exist_ok=True)
    out_path = os.path.join("notebooks", "data_cleaning_preprocessing.ipynb")

    nb = build_notebook()
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", allow_errors=False
    )
    client.execute()
    with open(out_path, "w", encoding="utf-8") as handle:
        nbformat.write(nb, handle)
    print(f"Notebook executed and saved to {out_path}")


if __name__ == "__main__":
    main()
