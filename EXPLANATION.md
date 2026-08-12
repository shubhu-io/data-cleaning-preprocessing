# Explanation — Clean and Preprocess a Dataset using Pandas and NumPy (Assignment 2)

This document explains **every line of code** in this project, in plain English.
Read it top-to-bottom alongside each source file. No prior data-science knowledge is
assumed — every concept and every function is unpacked from zero.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Concepts you need first](#2-concepts-you-need-first)
3. [`src/cleaning.py` — the cleaning pipeline](#3-srccleaningpy--the-cleaning-pipeline)
4. [`scripts/run_cleaning.py` — the pipeline runner](#4-scriptsrun_cleaningpy--the-pipeline-runner)
5. [`scripts/make_notebook.py` — builds the notebook](#5-scriptsmake_notebookpy--builds-the-notebook)
6. [`notebooks/data_cleaning_preprocessing.ipynb` — the executed deliverable](#6-notebooksdata_cleaning_preprocessingipynb--the-executed-deliverable)
7. [`tests/test_cleaning.py` — the test suite](#7-teststest_cleaningpy--the-test-suite)
8. [Data flow end-to-end](#8-data-flow-end-to-end)
9. [How to run everything](#9-how-to-run-everything)
10. [Glossary (quick lookup)](#10-glossary-quick-lookup)

---

## 1. Project overview

Raw public datasets are messy: cells are empty, rows are duplicated, text is stored
instead of numbers, and dates arrive as sentences. This project builds a **reproducible,
step-by-step cleaning and preprocessing pipeline** using **Pandas** and **NumPy** that
turns the raw *Netflix Movies & TV Shows* CSV (7,787 rows × 12 columns) into an
analysis- and ML-ready dataset (7,787 rows × 38 columns).

The pipeline follows the assignment's numbered requirements:

1. **Load** the dataset with Pandas.
2. **Inspect** missing values, duplicates and data types.
3. **Handle missing values** (drop fully-empty rows; fill the rest).
4. **Remove duplicates.**
5. **Convert data types** (parse dates; turn `90 min` / `2 Seasons` into numbers).
6. **NumPy numerical transformations** (percentiles, std, mean-imputation, log feature).
7. **Pandas filtering, sorting, grouping.**
8. **Summary statistics** and visual insights.
   Bonus: null heatmap, correlation matrix, and label + one-hot encoding for ML.

Everything lives in one module (`src/cleaning.py`), so the same functions are reused by
the command-line runner, the Jupyter notebook, and the tests — the definition of
*reproducibility*.

---

## 2. Concepts you need first

### Pandas
A Python library for tabular data.
- A **DataFrame** is a table: rows = records, columns = fields.
- A **Series** is one column of a DataFrame.
- `df["col"]` selects a column (a Series); `df[["a", "b"]]` selects several (a
  DataFrame); `df.loc[row]` selects a row by label.
- `df.shape` is a tuple `(rows, columns)`.
- `df.columns` is the list of column names.

### NumPy (`np`)
The numeric engine underneath pandas. `np.array([1, 2, 3])` is a fast grid of numbers.
We use it for math that is slow in pure Python: `np.nanmean`, `np.nanpercentile`,
`np.nanstd`, `np.log1p`.

### NaN / null
`NaN` (Not a Number) is Python/pandas' marker for a missing value. It is *contagious*:
`1 + NaN == NaN`. Tools like `df.isnull()` (is this cell missing?) and
`df.notnull()` (is it present?) help you find it. Machine-learning models cannot
learn from NaN, so we must replace or remove it.

### Missing-value strategies
- **`dropna(how="all")`** — remove rows where *every* column is NaN.
- **`fillna(value)`** — replace NaN with a fixed value (e.g. `"Unknown"`).
- **`ffill()`** (forward-fill) — copy the previous valid value down into gaps.
- **Mean-imputation** — fill numeric gaps with the column's mean (`np.nanmean`
  computes the mean *ignoring* NaN).

### Duplicates
`df.duplicated()` marks rows that repeat an earlier row (returns True/False per row).
`df.drop_duplicates()` removes the repeats, keeping the first (`keep="first"`).

### Data types (dtypes)
Every column has a type: `object`/`str` (text), `int64` (integer), `float64`
(decimal), `datetime64` (dates), `category` (fixed set of labels). Correct types enable
correct operations — you can't sort "years" if they're stored as text.

### Regular expressions (regex)
A mini language for finding patterns in text. Examples used here:
- `r"(\d+)\s*(min|Seasons|seasons)"` — one-or-more digits (`\d+`), optional whitespace
  (`\s*`), then `min` or `Seasons` or `seasons`. Parentheses create *capture groups*
  that `.str.extract` can pull out separately.

### `pd.to_datetime(...)` and `errors="coerce"`
Converts text to real dates. `errors="coerce"` means "if a value can't be parsed, turn
it into NaN instead of crashing."

### `astype` / categories
`df["type"].astype("category")` re-labels a column as a fixed set of categories
(memory-efficient, and tools know the legal values).

### `groupby`
Splits rows into groups by a column's values, then applies an operation per group.
`df.groupby("country").size()` → one row per country with its row count.
`df.groupby("country").agg(...)` lets you name the aggregations.

### `select_dtypes(include=[np.number])`
Picks only the numeric columns — handy for statistics and correlation.

### `.describe()`
Computes count, mean, std, min, and quartiles (25/50/75%) for numeric columns.

### Encoding (making text usable by ML)
Machine-learning models need numbers, not words:
- **Label encoding** — replace each category with an integer code (e.g. `Movie`→0,
  `TV Show`→1) using `pd.factorize`.
- **One-hot encoding** — turn one text column into many 0/1 columns, one per category.
  `pd.get_dummies` does this.

### Correlation
A number between −1 and 1 measuring how two numeric columns move together. +1 = both go
up together, −1 = opposite, 0 = unrelated. A *correlation matrix* shows this for every
pair of numeric columns.

### Plotting
- `matplotlib.pyplot as plt` — the base plotting library. `plt.figure` opens a canvas;
  `plt.savefig` saves it; `plt.close` frees it.
- `seaborn as sns` — builds on matplotlib for prettier charts with one line
  (`sns.heatmap`).
- `matplotlib.use("Agg")` — tells matplotlib to render to **files only** (no pop-up
  window), which is what we want in a script/notebook.

### Jupyter / notebook formats
- A `.ipynb` file is a JSON document containing a list of **cells** (markdown text cells
  and code cells).
- `nbformat` is the official library for reading/writing `.ipynb`.
- `nbclient` *executes* notebooks programmatically (headlessly, no browser), so the
  saved notebook contains real outputs.

### f-strings
`f"{x:.1f}"` inserts a value and formats it: `:.1f` = one decimal, `:.2f` = two,
`{x:.0f}` = zero decimals.

---

## 3. `src/cleaning.py` — the cleaning pipeline

This single module contains every cleaning function. Note `src/__init__.py` only has a
docstring and `__version__ = "1.0.0"` — its job is just to mark `src` as a package.

### 3.1 Imports and setup (lines 15–41)

```python
15 from __future__ import annotations
17 import os
18 from typing import Dict, List, Optional
```

- **Line 15** — `from __future__ import annotations` makes all type hints be treated as
  strings, so hints can reference types not yet defined at import time. It's a
  compatibility/forward-reference feature.
- **Line 17** — `os` for path work (checking files exist, joining folders).
- **Line 18** — type hints: `Dict`, `List`, `Optional`.

```python
20 import matplotlib
22 matplotlib.use("Agg")
23 import matplotlib.pyplot as plt
24 import numpy as np
25 import pandas as pd
26 import seaborn as sns
```

- **Line 22** — `matplotlib.use("Agg")` **must** be called before importing
  `pyplot`. The `Agg` backend renders charts directly to image files with no graphical
  window — perfect for scripts, servers and notebooks.
- **Lines 23–26** — import the plotting and numeric libraries with their standard short
  aliases (`plt`, `np`, `pd`, `sns`).

```python
28 REQUIRED_COLUMNS = [
29     "show_id",
30     "type",
31     "title",
32     "director",
33     "cast",
34     "country",
35     "date_added",
36     "release_year",
37     "rating",
38     "duration",
39     "listed_in",
40     "description",
41 ]
```

- The exact 12 columns the Netflix dataset must have. `load_data` uses this list to
  validate any CSV we're handed. These map to the raw CSV header seen in
  `data/raw/netflix_titles.csv`: `show_id,type,title,director,cast,country,date_added,
  release_year,rating,duration,listed_in,description`.

### 3.2 Step 1 — `load_data` (lines 47–57)

```python
47 def load_data(path: str) -> pd.DataFrame:
48     """Load the dataset from a CSV file."""
49     if not os.path.exists(path):
50         raise FileNotFoundError(f"Dataset not found at: {path}")
51     df = pd.read_csv(path)
52     missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
53     if missing_cols:
54         raise ValueError(
55             f"Dataset is missing expected columns: {missing_cols}"
56         )
57     return df
```

- **Lines 49–50** — fail fast with a clear message if the file isn't there.
- **Line 51** — `pd.read_csv(path)` reads the file into a DataFrame. `df.shape` will be
  `(7787, 12)` for the real data.
- **Line 52** — a *list comprehension* that collects any required column missing from
  the file (`c for c in REQUIRED_COLUMNS if c not in df.columns`).
- **Lines 53–56** — if any are missing, raise `ValueError` naming exactly which ones.
- **Line 57** — otherwise return the DataFrame.

### 3.3 Step 2 — Inspection functions (lines 63–87)

#### `missing_values_table` (lines 63–70)

```python
63 def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
64     """Return a table of missing values per column."""
65     total = df.isnull().sum()
66     percent = (total / len(df) * 100).round(2)
67     table = pd.DataFrame({"missing_count": total, "percent": percent})
68     return table[table["missing_count"] > 0].sort_values(
69         "missing_count", ascending=False
70     )
```

- **Line 65** — `df.isnull()` is a DataFrame of True/False (True = missing cell).
  `.sum()` counts the True's per column (True counts as 1). So `total` is a Series:
  missing count per column.
- **Line 66** — missing count ÷ row count × 100 = percentage, rounded to 2 decimals.
- **Line 67** — bundle the two Series into a labelled DataFrame.
- **Lines 68–70** — keep only columns with at least one missing value, sorted from most
  to least missing. For the raw data this shows `director` (2389 missing), `cast` (718),
  `country` (507), etc.

#### `duplicate_count` (lines 73–75)

```python
73 def duplicate_count(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
74     """Count duplicate rows (optionally considering only a subset of columns)."""
75     return int(df.duplicated(subset=subset).sum())
```

- `df.duplicated(subset=subset)` marks each row that duplicates an earlier one
  (`subset` lets you compare on only some columns). `.sum()` counts the True's.
  `int(...)` converts the numpy result to a plain Python integer.

#### `data_type_report` (lines 78–87)

```python
78 def data_type_report(df: pd.DataFrame) -> pd.DataFrame:
79     """Return a summary of columns and their data types."""
80     report = pd.DataFrame(
81         {
82             "dtype": df.dtypes.astype(str),
83             "non_null_count": df.notnull().sum(),
84             "null_count": df.isnull().sum(),
85         }
86     )
87     return report
```

- Builds a per-column report: `df.dtypes` (converted to strings), count of non-null
  cells, and count of null cells. This is the `dtype / non_null_count / null_count`
  table you see in the notebook.

### 3.4 Step 3 — `handle_missing_values` (lines 93–116)

```python
93 def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
99     cleaned = df.copy()
```

- **Line 99** — `df.copy()` makes a copy first. Pandas `fillna`/`dropna` can return new
  objects anyway, but copying guarantees the caller's original DataFrame is never
  mutated (side-effect safety).

```python
103     before = len(cleaned)
104     cleaned = cleaned.dropna(how="all")
105     print(f"[missing] Dropped {before - len(cleaned)} fully-empty rows.")
```

- **Lines 103–105** — `dropna(how="all")` removes rows where *every* cell is NaN
  (completely empty rows carry no information). We print how many were dropped.

```python
107     cleaned["director"] = cleaned["director"].fillna("Unknown")
108     cleaned["cast"] = cleaned["cast"].fillna("Unknown")
109     cleaned["country"] = cleaned["country"].fillna("Unknown")
110     cleaned["rating"] = cleaned["rating"].fillna("Unknown")
111     cleaned["duration"] = cleaned["duration"].fillna("Unknown")
112     cleaned["listed_in"] = cleaned["listed_in"].fillna("Unknown")
```

- **Lines 107–112** — for each *text* column, replace missing cells with the string
  `"Unknown"`. Rationale: `director` and `cast` are missing for ~2000/700 rows, and
  dropping those rows would delete thousands of otherwise-useful records. A placeholder
  keeps every row.

```python
114     cleaned["date_added"] = cleaned["date_added"].ffill().fillna("Not available")
```

- **Line 114** — two-step for the date column:
  - `.ffill()` *forward-fills*: each gap takes the value of the row above it (the
    dataset is roughly sorted by add-date, so "the previous date" is a sensible guess).
  - `.fillna("Not available")` then catches anything still missing (e.g. at the very
    top of the file, where there's no previous row) with an explicit placeholder.

```python
115     cleaned["description"] = cleaned["description"].fillna("No description")
```

- **Line 115** — descriptions missing → `"No description"`.

### 3.5 Step 4 — `remove_duplicates` (lines 122–128)

```python
122 def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
123     """Remove duplicate rows, keeping the first occurrence."""
124     cleaned = df.copy()
125     before = len(cleaned)
126     cleaned = cleaned.drop_duplicates(subset=subset, keep="first")
127     print(f"[duplicates] Removed {before - len(cleaned)} duplicate rows.")
128     return cleaned
```

- Copy, count rows, then `drop_duplicates(subset=subset, keep="first")` removes rows
  that repeat an earlier row (optionally comparing only the given columns), keeping the
  first occurrence. Prints how many were removed.

### 3.6 Step 5 — Data-type conversion (lines 134–163)

#### `parse_date_added` (lines 134–143)

```python
134 def parse_date_added(df: pd.DataFrame) -> pd.DataFrame:
135     """Convert date_added (e.g. 'September 25, 2021') to a datetime column."""
136     cleaned = df.copy()
137     cleaned["date_added_dt"] = pd.to_datetime(
138         cleaned["date_added"].replace("Not available", np.nan),
139         errors="coerce",
140     )
141     cleaned["added_year"] = cleaned["date_added_dt"].dt.year
142     cleaned["added_month"] = cleaned["date_added_dt"].dt.month
143     return cleaned
```

- **Lines 137–140** — `.replace("Not available", np.nan)` swaps our placeholder back to
  NaN (a date parser can't understand it), then `pd.to_datetime(..., errors="coerce")`
  parses text like `"September 25, 2021"` into a real `datetime64` value, putting the
  result in a **new** column `date_added_dt` (we keep the original text column intact).
  Unparseable → NaN, no crash.
- **Line 141** — `.dt.year` extracts the year part into `added_year` (a number).
- **Line 142** — `.dt.month` extracts the month into `added_month`. These become usable
  numeric features — e.g. "most titles were added 2018–2021".

#### `extract_duration` (lines 146–154)

```python
146 def extract_duration(df: pd.DataFrame) -> pd.DataFrame:
147     """Split duration ('90 min' / '2 Seasons') into numeric value + unit."""
148     cleaned = df.copy()
149     parts = cleaned["duration"].str.extract(r"(\d+)\s*(min|Seasons|seasons)", expand=True)
150     cleaned["duration_value"] = pd.to_numeric(parts[0], errors="coerce")
151     cleaned["duration_unit"] = parts[1].str.lower().fillna("unknown")
152     cleaned["is_movie"] = (cleaned["type"] == "Movie").astype(int)
153     cleaned["is_tv_show"] = (cleaned["type"] == "TV Show").astype(int)
154     return cleaned
```

- **Line 149** — `.str.extract(regex)` runs the regular expression
  `r"(\d+)\s*(min|Seasons|seasons)"` on every `duration` string and splits out the two
  *capture groups* into two columns: group 0 = the number, group 1 = the unit.
  So `"93 min"` → `93`/`min` and `"4 Seasons"` → `4`/`seasons`. `expand=True` guarantees
  the result is a DataFrame.
- **Line 150** — `pd.to_numeric(parts[0], errors="coerce")` turns the number text into
  a number; anything weird becomes NaN.
- **Line 151** — the unit column is lowercased (`"Seasons"` → `"seasons"`) so both
  spellings become one value, and missing units become `"unknown"`.
- **Lines 152–153** — create 0/1 flag columns: `is_movie` is 1 when `type == "Movie"`
  else 0; `is_tv_show` is 1 when `type == "TV Show"`. `.astype(int)` converts the
  True/False booleans into 1/0 integers. These are directly usable numeric features.

#### `convert_types` (lines 157–163)

```python
157 def convert_types(df: pd.DataFrame) -> pd.DataFrame:
158     """Apply numeric conversions and categorical types."""
159     cleaned = df.copy()
160     cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
161     cleaned["type"] = cleaned["type"].astype("category")
162     cleaned["rating"] = cleaned["rating"].astype("category")
163     return cleaned
```

- **Line 160** — guarantee `release_year` is numeric (coerce anything odd to NaN).
- **Lines 161–162** — store `type` and `rating` as *categorical* dtypes: pandas records
  the full set of allowed labels, which is efficient and prevents typos from creating
  accidental new categories.

### 3.7 Step 6 — `numpy_transformations` (lines 169–192)

This is the function that satisfies the assignment's "use NumPy for numerical
transformations" requirement.

```python
169 def numpy_transformations(df: pd.DataFrame) -> pd.DataFrame:
176     cleaned = df.copy()
178     mean_duration = float(np.nanmean(cleaned["duration_value"]))
179     cleaned["duration_value"] = cleaned["duration_value"].fillna(mean_duration)
```

- **Line 178** — `np.nanmean` computes the mean of `duration_value` while **ignoring**
  NaN. (A plain `mean()` would return NaN if any value is missing.)
- **Line 179** — replace any leftover NaN durations with that mean. This is
  *mean-imputation* — a "typical" value fills the gap.

```python
182     years = cleaned["release_year"].to_numpy()
183     durations = cleaned["duration_value"].to_numpy()
184     for label, arr in [("release_year", years), ("duration_value", durations)]:
185         p25, p50, p75 = np.nanpercentile(arr, [25, 50, 75])
186         print(
187             f"[numpy] {label}: p25={p25:.1f} median={p50:.1f} p75={p75:.1f} "
188             f"std={np.nanstd(arr):.2f}"
189         )
```

- **Lines 182–183** — `.to_numpy()` converts each column into a raw NumPy array.
- **Lines 184–189** — loop over both arrays. `np.nanpercentile(arr, [25, 50, 75])`
  returns the 25th, 50th (median) and 75th percentiles in one call. `np.nanstd`
  computes the standard deviation (spread) ignoring NaN. We print all five numbers per
  column. On the real data: `duration_value` median ≈ 88 (minutes) with std ≈ 37.

```python
191     cleaned["duration_log"] = np.log1p(cleaned["duration_value"])
192     return cleaned
```

- **Line 191** — `np.log1p(x)` computes `ln(1 + x)`. This **log-squashes** the duration
  so a `312`-minute movie doesn't dwarf a `2`-season show in the eyes of a model.
  Adding `1` first keeps `x = 0` well-defined (ln(0) is undefined). The result is a new
  numeric feature `duration_log` — friendlier for ML than raw duration.

### 3.8 Step 7 — Filtering, sorting, grouping (lines 198–226)

#### `filter_movies_after_year` (lines 198–200)

```python
198 def filter_movies_after_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
199     """Filter to Movies released strictly after the given year."""
200     return df[(df["type"] == "Movie") & (df["release_year"] > year)].copy()
```

- A *boolean mask*: `(df["type"] == "Movie")` gives True/False per row, and so does
  `(df["release_year"] > year)`. `&` combines them element-by-element (AND — both must
  be True). `df[mask]` keeps only the True rows. Parentheses around each comparison are
  required in pandas. `.copy()` protects the original.

#### `sort_by_release_year` (lines 203–205)

```python
203 def sort_by_release_year(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
204     """Sort by release year (default: newest first)."""
205     return df.sort_values("release_year", ascending=ascending).copy()
```

- `sort_values("release_year", ascending=...)` reorders the rows. The default parameter
  `ascending=False` means newest first unless the caller says otherwise.

#### `group_by_country` (lines 208–220)

```python
208 def group_by_country(df: pd.DataFrame) -> pd.DataFrame:
209     """Group by country and compute counts + mean release year."""
210     agg = {
211         "title_count": ("title", "count"),
212         "avg_release_year": ("release_year", "mean"),
213     }
214     if "duration_value" in df.columns:
215         agg["avg_duration_value"] = ("duration_value", "mean")
216     return (
217         df.groupby("country", observed=True)
218         .agg(**agg)
219         .sort_values("title_count", ascending=False)
220     )
```

- **Lines 210–215** — a dict of named aggregations: `title_count` = number of titles,
  `avg_release_year` = mean release year. If the column exists (i.e. we already ran
  duration extraction), also compute mean duration per country. The dict format
  `"name": ("column", "operation")` is the modern, readable way to pass `.agg`.
- **Lines 216–220** — `df.groupby("country", observed=True)` groups rows by country.
  `observed=True` only creates groups that actually appear (avoids phantom empty
  categories). `.agg(**agg)` applies the aggregations (`**` spreads the dict into
  keyword arguments). `.sort_values("title_count", ascending=False)` puts the most
  prolific country first — e.g. **United States (2,555)** then **India (923)**.

#### `summary_statistics` (lines 223–226)

```python
223 def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
224     """Summary statistics for numerical columns."""
225     numeric = df.select_dtypes(include=[np.number])
226     return numeric.describe().T
```

- **Line 225** — `select_dtypes(include=[np.number])` keeps only numeric columns.
- **Line 226** — `.describe()` computes count/mean/std/min/quartiles/max for each, and
  `.T` *transposes* the result so columns are features and rows are statistics (more
  readable in a notebook).

### 3.9 Bonus — `plot_null_heatmap` (lines 232–240)

```python
232 def plot_null_heatmap(df: pd.DataFrame, out_path: str) -> str:
233     """Save a heatmap of missing values."""
234     plt.figure(figsize=(12, 5))
235     sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
236     plt.title("Missing Value Heatmap")
237     plt.tight_layout()
238     plt.savefig(out_path, dpi=120, bbox_inches="tight")
239     plt.close()
240     return out_path
```

- **Line 234** — open a 12×5 inch canvas.
- **Line 235** — `df.isnull()` is the True/False missing mask. `sns.heatmap` colours it:
  `cbar=False` hides the colour bar, `cmap="viridis"` is a colour scheme,
  `yticklabels=False` hides the row labels (7,787 of them would be illegible). Dark rows
  = missing, light = present — missing value *patterns* become visible at a glance.
- **Lines 237–238** — `tight_layout` fixes overlap; `savefig(out_path, dpi=120,
  bbox_inches="tight")` writes a PNG at 120 pixels/inch with no clipped edges.
- **Line 239** — `plt.close()` releases the figure's memory.
- **Line 240** — return the path so callers can report it.

### 3.10 Bonus — `plot_correlation_matrix` (lines 243–252)

```python
243 def plot_correlation_matrix(df: pd.DataFrame, out_path: str) -> str:
244     """Save a correlation heatmap of numerical features."""
245     numeric = df.select_dtypes(include=[np.number])
246     plt.figure(figsize=(8, 6))
247     sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
248     plt.title("Correlation Matrix of Numerical Fields")
249     plt.tight_layout()
250     plt.savefig(out_path, dpi=120, bbox_inches="tight")
251     plt.close()
252     return out_path
```

- **Line 245** — numeric columns only.
- **Line 247** — `numeric.corr()` computes the Pearson correlation between every pair of
  numeric columns (a small square matrix). `annot=True` prints the number inside each
  cell; `cmap="coolwarm"` colours blue = negative, red = positive; `fmt=".2f"` formats
  the numbers. This is where the insight "`release_year` and `added_year` correlate
  most" comes from.

### 3.11 Bonus — `label_encode` (lines 258–264)

```python
258 def label_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
259     """Apply simple label encoding to the given categorical columns."""
260     encoded = df.copy()
261     for column in columns:
262         codes, _ = pd.factorize(encoded[column], sort=False)
263         encoded[f"{column}_encoded"] = codes
264     return encoded
```

- **Line 262** — `pd.factorize(encoded[column], sort=False)` converts each category to
  an integer code. It returns `(codes, uniques)` — we discard the second with `_`.
  Each distinct label gets its own number (order is the order of first appearance).
- **Line 263** — store the codes in a new column named `{column}_encoded` (an f-string),
  e.g. `type_encoded`. The original text column is kept untouched.

### 3.12 Bonus — `top_country_encoding` (lines 267–281)

```python
267 def top_country_encoding(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
268     """Replace country with top-N one-hot columns (smaller feature space).
270     Returns a copy with `country_encoded` (label) and `country_topN` columns.
271     """
272     encoded = df.copy()
273     counts = encoded["country"].value_counts()
274     top = counts.head(top_n).index
```

- **Line 273** — `value_counts()` counts titles per country, most common first.
- **Line 274** — `.head(top_n)` keeps the top 15, and `.index` gives their *names*
  (e.g. "United States", "India", ...).

```python
275     encoded["country_encoded"] = np.where(
276         encoded["country"].isin(top), encoded["country"], "Other"
277     )
```

- `np.where(condition, if_true, if_false)` is a vectorised if/else. For each row: if the
  country is in the top-15 (`isin(top)`), keep its name; otherwise replace with
  `"Other"`. This collapses hundreds of rare countries into one bucket — keeping the
  feature space small.

```python
278     encoded = pd.get_dummies(
279         encoded, columns=["country_encoded"], prefix="country"
280     )
281     return encoded
```

- `pd.get_dummies(..., columns=["country_encoded"], prefix="country")` one-hot encodes
  the prepared column: for each top country (plus `Other` and `Unknown`) it creates one
  0/1 column named `country_<name>` (e.g. `country_United States`). A movie from the
  US has a 1 in `country_United States` and 0 everywhere else. This is what pushes the
  final shape to 38 columns.

### 3.13 `run_pipeline` — the orchestrator (lines 287–348)

```python
287 def run_pipeline(
288     input_path: str,
289     output_path: str,
290     figures_dir: str,
291 ) -> Dict:
```

- Takes input CSV path, output CSV path, and the folder for figures. Returns a results
  dict.

```python
293     print(f"Step 1: Loading dataset from {input_path}")
294     df = load_data(input_path)
295     print(f"  Shape: {df.shape}")
```

- **Lines 293–295** — load and print the raw shape `(7787, 12)`.

```python
297     print("\nStep 2: Inspecting data")
298     print(missing_values_table(df))
299     print(f"  Duplicate rows: {duplicate_count(df)}")
300     print(data_type_report(df))
```

- **Lines 297–300** — run the three inspection helpers and print their results. This is
  the visible "documentation of every step" the assignment wants.

```python
302     print("\nStep 3: Handling missing values")
303     df = handle_missing_values(df)
305     print("\nStep 4: Removing duplicates")
306     df = remove_duplicates(df)
```

- **Lines 302–306** — apply steps 3 and 4, reassigning `df` each time. Each function
  prints what it changed, so the transcript shows the whole journey.

```python
308     print("\nStep 5: Converting data types")
309     df = parse_date_added(df)
310     df = extract_duration(df)
311     df = convert_types(df)
```

- **Lines 308–311** — the three type-conversion functions add `date_added_dt`,
  `added_year`, `added_month`, `duration_value`, `duration_unit`, `is_movie`,
  `is_tv_show`, and re-type `release_year`/`type`/`rating`.

```python
313     print("\nStep 6: NumPy transformations")
314     df = numpy_transformations(df)
```

- **Lines 313–314** — mean-imputation, percentile prints, and the `duration_log` feature.

```python
316     print("\nStep 7: Filtering, sorting, grouping")
317     recent_movies = filter_movies_after_year(df, 2018)
318     sorted_df = sort_by_release_year(df)
319     by_country = group_by_country(df)
320     print(by_country.head(10))
```

- **Lines 316–320** — demonstrate the three pandas operations. `by_country.head(10)`
  prints the 10 most productive countries.

```python
322     print("\nSummary statistics")
323     print(summary_statistics(df))
```

- **Lines 322–323** — print the numeric summary table.

```python
325     print("\nBonus: visualisations")
326     os.makedirs(figures_dir, exist_ok=True)
327     null_plot = plot_null_heatmap(df, os.path.join(figures_dir, "null_heatmap.png"))
328     corr_plot = plot_correlation_matrix(
329         df, os.path.join(figures_dir, "correlation_matrix.png")
330     )
331     print(f"  Saved: {null_plot}")
332     print(f"  Saved: {corr_plot}")
```

- **Line 326** — create the figures folder if needed (`exist_ok=True` = don't error if
  it already exists).
- **Lines 327–330** — generate and save both charts into that folder.
- **Lines 331–332** — report their paths.

```python
334     print("\nBonus: ML-readiness encoding")
335     df = label_encode(df, ["type", "rating"])
336     df = top_country_encoding(df, top_n=15)
337     print(f"  Encoded columns added; final shape: {df.shape}")
```

- **Lines 334–337** — apply the two encoders. The final `df.shape` is `(7787, 38)` —
  the ML-ready table.

```python
339     os.makedirs(os.path.dirname(output_path), exist_ok=True)
340     df.to_csv(output_path, index=False)
341     print(f"\nCleaned dataset saved to: {output_path}")
```

- **Line 339** — create the output folder (e.g. `data/processed`).
- **Line 340** — `df.to_csv(output_path, index=False)` writes the DataFrame to CSV.
  `index=False` means "don't write the row numbers column" — the file stays clean.

```python
343     return {
344         "input_shape": df.shape,
345         "cleaned_path": output_path,
346         "null_plot": null_plot,
347         "corr_plot": corr_plot,
348     }
```

- Returns the final shape, the saved CSV path, and both figure paths. Note the key is
  called `"input_shape"` but holds the **final** shape — `run_cleaning.py` uses it only
  to report the outcome.

---

## 4. `scripts/run_cleaning.py` — the pipeline runner

The file you run from the command line to reproduce the whole pipeline.

```python
1  """End-to-end data cleaning & preprocessing pipeline runner.
4      python scripts/run_cleaning.py
5  """
7  import os
8  import sys
10 sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
12 from src.cleaning import run_pipeline  # noqa: E402
```

- **Lines 7–8** — standard imports.
- **Line 10** — the *sys.path trick*: `__file__` is `.../scripts/run_cleaning.py`;
  `os.path.abspath` makes it absolute; the **outer** `dirname` gives `scripts/`; the
  **second** (outermost) `dirname` gives the project root. We insert the root at the
  front of `sys.path` so `import src.cleaning` works even though the script lives in a
  subfolder.
- **Line 12** — `from src.cleaning import run_pipeline`. `# noqa: E402` tells linters
  "this import intentionally isn't at the top, because line 10 must run first."

```python
14 BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
15 INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "netflix_titles.csv")
16 OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "netflix_titles_clean.csv")
17 FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
```

- **Lines 14–17** — absolute paths built from the project root, so the script works no
  matter which folder the user runs it from: the raw input, the cleaned output, and the
  figures folder.

```python
20 def main() -> None:
21     """Run the pipeline."""
22     result = run_pipeline(INPUT_PATH, OUTPUT_PATH, FIGURES_DIR)
23     print("\nPipeline complete.")
24     print(f"Cleaned dataset: {result['cleaned_path']}")
25     print(f"Final shape: {result['input_shape']}")
```

- Calls `run_pipeline` once with all three paths, then prints a short completion report
  using the returned dict.

```python
28 if __name__ == "__main__":
29     main()
```

- The standard guard: run only when executed directly.

---

## 5. `scripts/make_notebook.py` — builds the notebook

This script **generates and then executes** the Jupyter notebook deliverable
(`notebooks/data_cleaning_preprocessing.ipynb`) from a hard-coded list of cells, so the
saved notebook always contains fresh, real outputs.

### 5.1 Imports (lines 7–13)

```python
7  import os
8  import sys
10 import nbclient
11 import nbformat
13 BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

- `nbformat` — read/write `.ipynb` files. `nbclient` — execute cells headlessly.
- Note there's **no** `sys.path.insert` here: the notebook's own first code cell does
  the path setup when *it* runs inside the executed notebook.

### 5.2 `CELLS` (lines 15–60)

A list of `(kind, source)` tuples. `kind` is `"md"` (a markdown text cell) or `"code"`
(a code cell). Reading through them shows the notebook is a thin *narrative* wrapper
around `src.cleaning`:

- **Markdown intro** — title, dataset, objective (lines 16–23).
- **Code** — imports numpy, pandas, and `from src import cleaning as cl`, plus a
  `sys.path.insert` so the notebook can import `src` no matter where Jupyter was started
  (line 24).
- **Step 1 (md + code)** — `cl.load_data("data/raw/netflix_titles.csv")`, print shape,
  show the first rows.
- **Step 2** — `missing_values_table`, `duplicate_count`, `data_type_report`.
- **Step 3** — `handle_missing_values`, then re-check missing values (should be empty).
- **Step 4** — `remove_duplicates`, print row count.
- **Step 5** — `parse_date_added`, `extract_duration`, `convert_types`, and a peek at the
  new date/duration columns.
- **Step 6** — `numpy_transformations` (prints the percentile/std lines).
- **Step 7** — `filter_movies_after_year`, `sort_by_release_year`,
  `group_by_country(...).head(10)`.
- **Summary statistics** — `summary_statistics`.
- **Bonus 1** — `plot_null_heatmap` + `plot_correlation_matrix` into `reports/figures/`.
- **Bonus 2** — `label_encode(["type", "rating"])` then `top_country_encoding(top_n=15)`,
  print final shape and head.
- **Save** — `df.to_csv("data/processed/netflix_titles_clean.csv", index=False)`.
- **Key findings (markdown)** — a six-point summary of what the cleaning achieved.

Notice every code cell **calls functions from `src.cleaning`** rather than re-implementing
logic — the notebook and the CLI runner produce byte-identical results.

### 5.3 `build_notebook` (lines 63–79)

```python
63 def build_notebook() -> nbformat.NotebookNode:
64     """Assemble the notebook object from CELLS."""
65     nb = nbformat.v4.new_notebook()
66     nb.metadata.kernelspec = {
67         "display_name": "Python 3",
68         "language": "python",
69         "name": "python3",
70     }
71     nb.metadata.language_info = {"name": "python"}
72     cells = []
73     for kind, source in CELLS:
74         if kind == "md":
75             cells.append(nbformat.v4.new_markdown_cell(source))
76         else:
77             cells.append(nbformat.v4.new_code_cell(source))
78     nb["cells"] = cells
79     return nb
```

- **Line 65** — create an empty notebook object.
- **Lines 66–71** — set its kernel metadata (Jupyter needs to know this is a Python 3
  notebook; this is the `metadata.kernelspec` block you can see in the `.ipynb` file).
- **Lines 72–78** — loop over `CELLS`, turn each `(kind, source)` tuple into a proper
  cell object (markdown cell or code cell), and attach the list under `nb["cells"]`.
- **Line 79** — return the finished notebook.

### 5.4 `main` (lines 82–94)

```python
82 def main() -> None:
83     """Build and execute the notebook, saving it to notebooks/."""
84     os.makedirs("notebooks", exist_ok=True)
85     out_path = os.path.join("notebooks", "data_cleaning_preprocessing.ipynb")
87     nb = build_notebook()
88     client = nbclient.NotebookClient(
89         nb, timeout=300, kernel_name="python3", allow_errors=False
90     )
91     client.execute()
92     with open(out_path, "w", encoding="utf-8") as handle:
93         nbformat.write(nb, handle)
94     print(f"Notebook executed and saved to {out_path}")
```

- **Line 84** — create `notebooks/` if missing.
- **Lines 88–90** — build a `NotebookClient`: `timeout=300` (seconds per cell),
  `kernel_name="python3"`, `allow_errors=False` (any cell error fails the whole run
  loudly — no silent half-finished notebooks).
- **Line 91** — `client.execute()` runs every code cell. The notebook object now
  contains the **outputs** (dataframe HTML, printed text, the "Shape: (7787, 12)"
  lines, etc.) that you see in `data_cleaning_preprocessing.ipynb`.
- **Lines 92–93** — write the executed notebook to disk as JSON.

---

## 6. `notebooks/data_cleaning_preprocessing.ipynb` — the executed deliverable

The `.ipynb` is a JSON file. Its structure (simplified):

```
{
  "cells": [ ... 15 cell objects ... ],
  "metadata": { "kernelspec": {...}, "language_info": {...} },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- **`"cells"`** — an ordered list. Each cell has a `"cell_type"` (`markdown` or
  `code`), a `"source"` (array of text lines), and for code cells: `"execution_count"`,
  `"outputs"` (the stored results), and `"metadata"`.
- **`"metadata"`** — kernel info (Python 3) and language info. This is set by
  `build_notebook`.
- **`"nbformat"`** — the notebook format version (4), and `"nbformat_minor"` (5).

The cell **sources** are exactly the `CELLS` list from `make_notebook.py`. The important
difference is the `"outputs"` arrays: they contain the *executed* results — for example,
the Step-1 cell's outputs include the printed `Shape: (7787, 12)` and an HTML-rendered
`df.head()` table; the Bonus-2 cell's output shows `Final shape: (7787, 38)` and the
one-hot encoded country columns (`country_United States`, `country_India`, ...).

The final cell is a markdown summary with the key findings:
1. Missing values were concentrated in director/cast/country/date_added/rating/duration/
   listed_in/description, and were **filled** (Unknown/forward-fill) rather than dropping
   thousands of rows.
2. Duplicates were removed.
3. `date_added` became a real datetime with `added_year`/`added_month` extracted.
4. `90 min` / `2 Seasons` were split into numeric `duration_value` + `duration_unit`
   (median movie duration ≈ 88 min).
5. Correlation insight: `release_year` ↔ `added_year` are most correlated; `is_movie` is
   negatively related to season-style durations.
6. The ML-ready 38-column dataset was saved to `data/processed/netflix_titles_clean.csv`.

The processed CSV's header confirms the end state (38 columns):
`show_id, type, title, director, cast, country, date_added, release_year, rating,
duration, listed_in, description, date_added_dt, added_year, added_month,
duration_value, duration_unit, is_movie, is_tv_show, duration_log, type_encoded,
rating_encoded, country_Australia, ..., country_United States, country_Unknown`.

---

## 7. `tests/test_cleaning.py` — the test suite

17 tests. A shared fixture builds a small Netflix-like DataFrame, and each test checks
one function's contract.

### The `sample_df` fixture (lines 10–28)

```python
10 @pytest.fixture(scope="module")
11 def sample_df():
12     """Small Netflix-like DataFrame used across tests."""
13     return pd.DataFrame(
14         {
15             "show_id": ["s1", ..., "s10"],
16             "type": ["Movie"] * 5 + ["TV Show"] * 5,
17             "title": [f"Title {i}" for i in range(10)],
18             "director": ["Dir A"] * 8 + [np.nan] * 2,
...
25             "duration": ["90 min", "2 Seasons"] * 5,
26             ...
27         }
28     )
```

- **Lines 10–28** — a 10-row mini dataset with all 12 required columns, deliberately
  containing NaN cells (`np.nan` in director, cast, country, date_added, description).
  `scope="module"` means it's built **once per test file**, not once per test.
  List tricks used: `["Movie"] * 5` repeats the list 5 times; `[f"Title {i}" for i in
  range(10)]` builds 10 titles; `["90 min", "2 Seasons"] * 5` interleaves the two
  duration formats.

### The tests

| Test (line) | What it checks |
|---|---|
| `test_load_data_missing_file` (31) | `load_data` on a missing path raises `FileNotFoundError` (`pytest.raises`). |
| `test_load_data_valid` (36) | Loading the sample (written to a temp CSV) returns the same shape. |
| `test_missing_values_table` (43) | The report lists `director` with `missing_count == 2`. |
| `test_duplicate_count` (49) | `[1,1,2,3,3]` → 2 duplicates. |
| `test_handle_missing_values` (54) | After cleaning, **no** NaN remains anywhere (`isnull().sum().sum() == 0`) and `"Unknown"` appears in the country column. |
| `test_remove_duplicates` (60) | `[1,1,2]` → 2 rows. |
| `test_parse_date_added` (66) | `date_added_dt` column exists and has ≥ 8 non-null values (8 of the sample rows had dates). |
| `test_extract_duration` (72) | Row 0: `"90 min"` → value 90, unit `min`; Row 1: `"2 Seasons"` → value 2, unit `seasons`. |
| `test_convert_types` (80) | `type`'s dtype is `pd.CategoricalDtype`. |
| `test_numpy_transformations` (85) | After duration extraction + numpy step, `duration_value` has no NaN and `duration_log` exists. |
| `test_filter_movies_after_year` (92) | All results are Movies with `release_year > 2004`. |
| `test_sort_by_release_year` (99) | Sorted ascending → `is_monotonic_increasing` is True. |
| `test_group_by_country` (104) | `title_count` column exists; India has 4 titles. |
| `test_summary_statistics` (110) | `release_year` appears as a row (numeric columns become the index after `.T`). |
| `test_label_encode` (115) | `type_encoded` exists and has exactly 2 distinct codes (Movie / TV Show). |
| `test_top_country_encoding` (121) | With `top_n=1`, at least one `country_*` column is created. |

### `test_run_pipeline` (lines 127–153) — the integration test

```python
127 def test_run_pipeline(tmp_path):
128     df = pd.DataFrame({ ... "show_id": ["s1", "s1", "s2"], ... })
```

- Builds a 3-row DataFrame that **includes a real duplicate** (`s1` appears twice) and a
  NaN director — so the pipeline actually has something to do.

```python
144     raw_dir = tmp_path / "raw"
145     raw_dir.mkdir()
146     in_path = str(raw_dir / "netflix.csv")
147     out_path = str(tmp_path / "processed" / "clean.csv")
148     df.to_csv(in_path, index=False)
150     result = cl.run_pipeline(in_path, out_path, str(tmp_path / "fig"))
151     assert os.path.exists(out_path)
152     assert os.path.exists(result["null_plot"])
153     assert os.path.exists(result["corr_plot"])
```

- **Lines 144–148** — lay out a fake project structure inside the temp folder: a `raw`
  folder with the input CSV; an output path that doesn't exist yet (its parent
  `processed` also doesn't exist — good, we're testing that the pipeline creates it).
- **Line 150** — run the **entire pipeline** on the tiny dataset.
- **Lines 151–153** — assert the cleaned CSV was written and both figure files exist.
  This proves the whole chain works end to end, including folder creation.

---

## 8. Data flow end-to-end

```
data/raw/netflix_titles.csv (7,787 × 12)
   │  load_data()                          pd.read_csv + column validation
   ▼
raw DataFrame
   │  missing_values_table / duplicate_count / data_type_report   (Step 2, printed)
   │  handle_missing_values()              drop fully-empty rows, fill "Unknown",
   │                                       ffill date_added, "No description"
   │  remove_duplicates()                  drop_duplicates(keep="first")
   │  parse_date_added()                   date_added_dt, added_year, added_month
   │  extract_duration()                   90 min / 2 Seasons → value + unit,
   │                                       is_movie / is_tv_show flags
   │  convert_types()                      numeric release_year, categorical type/rating
   │  numpy_transformations()              mean-imputation, percentiles/std printed,
   │                                       duration_log = log1p(duration)
   │  filter / sort / group_by_country     demonstrated (Step 7)
   │  summary_statistics()                 describe().T
   │  plot_null_heatmap()  ──► reports/figures/null_heatmap.png
   │  plot_correlation_matrix() ──► reports/figures/correlation_matrix.png
   │  label_encode(type, rating)           type_encoded, rating_encoded
   │  top_country_encoding(top_n=15)       16 country_* one-hot columns
   ▼
ML-ready DataFrame (7,787 × 38)
   │  df.to_csv(index=False)
   ▼
data/processed/netflix_titles_clean.csv

Also produced:
   notebooks/data_cleaning_preprocessing.ipynb   (built+executed by make_notebook.py)
   tests run the same src.cleaning functions      (reproducibility check)
```

---

## 9. How to run everything

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn nbformat nbclient ipykernel pytest

# 2. Run the full cleaning pipeline (prints every step + saves CSV and figures)
python scripts/run_cleaning.py

# 3. Rebuild and execute the notebook (regenerates real outputs)
python scripts/make_notebook.py

# 4. Open the notebook interactively (optional)
jupyter notebook notebooks/data_cleaning_preprocessing.ipynb

# 5. Run the tests (17 tests)
python -m pytest tests/ -v
```

---

## 10. Glossary (quick lookup)

| Term | Meaning |
|---|---|
| DataFrame | Pandas table: rows = records, columns = fields. |
| Series | One column of a DataFrame. |
| NaN / null | Marker for a missing value. |
| `isnull()` / `notnull()` | Test which cells are / aren't missing. |
| `dropna(how="all")` | Remove rows where every column is NaN. |
| `fillna(value)` | Replace NaN with a value. |
| `ffill()` | Forward-fill: gaps take the previous row's value. |
| Mean-imputation | Filling numeric gaps with the column mean. |
| Duplicate | A row identical to an earlier row. |
| dtype | A column's data type (str, int, float, datetime, category). |
| `to_datetime(...)` | Parse text dates; `errors="coerce"` → NaN on failure. |
| Regex | Pattern language; `\d+` = digits, `\s*` = spaces, `( )` = capture groups. |
| `.str.extract()` | Pull regex capture groups out of text into columns. |
| Category dtype | Fixed set of allowed labels for a column. |
| `groupby` | Split into groups, apply an aggregation per group. |
| `.agg()` | Named aggregations like count/mean per group. |
| Percentile | A value below which N% of data falls (p50 = median). |
| `np.nanpercentile` | Percentile ignoring NaN. |
| Standard deviation (std) | Spread of values around the mean. |
| `np.log1p(x)` | `ln(1+x)` — log-squashes a column. |
| `select_dtypes` | Pick columns by dtype (e.g. only numeric). |
| `.describe()` | Count/mean/std/min/quartiles/max summary. |
| `.T` | Transpose — swap rows and columns. |
| Correlation | −1..1 measure of how two columns move together. |
| Label encoding | Each category → an integer code (`pd.factorize`). |
| One-hot encoding | One 0/1 column per category (`pd.get_dummies`). |
| `np.where` | Vectorised if/else over an array. |
| `matplotlib.use("Agg")` | Render charts to files, no window. |
| `sns.heatmap` | Seaborn heatmap (missing-mask / correlation). |
| `.ipynb` | JSON notebook file: cells + metadata + outputs. |
| nbformat / nbclient | Read/write notebooks / execute them headlessly. |
| `scope="module"` | Build a pytest fixture once per test module. |
| `pytest.raises` | Assert that an exception is raised. |
