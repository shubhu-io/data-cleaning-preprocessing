import os

import numpy as np
import pandas as pd
import pytest

from src import cleaning as cl


@pytest.fixture(scope="module")
def sample_df():
    """Small Netflix-like DataFrame used across tests."""
    return pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
            "type": ["Movie"] * 5 + ["TV Show"] * 5,
            "title": [f"Title {i}" for i in range(10)],
            "director": ["Dir A"] * 8 + [np.nan] * 2,
            "cast": ["Cast A"] * 9 + [np.nan],
            "country": ["India"] * 4 + ["USA"] * 4 + [np.nan] * 2,
            "date_added": ["September 25, 2021"] * 8 + [np.nan] * 2,
            "release_year": list(range(2001, 2011)),
            "rating": ["TV-MA"] * 10,
            "duration": ["90 min", "2 Seasons"] * 5,
            "listed_in": ["Drama"] * 10,
            "description": ["Desc"] * 9 + [np.nan],
        }
    )


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        cl.load_data(str(tmp_path / "nope.csv"))


def test_load_data_valid(sample_df, tmp_path):
    path = str(tmp_path / "netflix.csv")
    sample_df.to_csv(path, index=False)
    df = cl.load_data(path)
    assert df.shape == sample_df.shape


def test_missing_values_table(sample_df):
    table = cl.missing_values_table(sample_df)
    assert "director" in table.index
    assert table.loc["director", "missing_count"] == 2


def test_duplicate_count():
    df = pd.DataFrame({"a": [1, 1, 2, 3, 3]})
    assert cl.duplicate_count(df) == 2


def test_handle_missing_values(sample_df):
    cleaned = cl.handle_missing_values(sample_df)
    assert cleaned.isnull().sum().sum() == 0
    assert "Unknown" in cleaned["country"].values


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    cleaned = cl.remove_duplicates(df)
    assert len(cleaned) == 2


def test_parse_date_added(sample_df):
    cleaned = cl.parse_date_added(sample_df)
    assert "date_added_dt" in cleaned.columns
    assert cleaned["date_added_dt"].notna().sum() >= 8


def test_extract_duration(sample_df):
    cleaned = cl.extract_duration(sample_df)
    assert cleaned.loc[0, "duration_value"] == 90
    assert cleaned.loc[0, "duration_unit"] == "min"
    assert cleaned.loc[1, "duration_value"] == 2
    assert cleaned.loc[1, "duration_unit"] == "seasons"


def test_convert_types(sample_df):
    cleaned = cl.convert_types(sample_df)
    assert isinstance(cleaned["type"].dtype, pd.CategoricalDtype)


def test_numpy_transformations(sample_df):
    cleaned = cl.extract_duration(sample_df)
    cleaned = cl.numpy_transformations(cleaned)
    assert cleaned["duration_value"].isna().sum() == 0
    assert "duration_log" in cleaned.columns


def test_filter_movies_after_year(sample_df):
    cleaned = cl.extract_duration(sample_df)
    result = cl.filter_movies_after_year(cleaned, 2004)
    assert (result["type"] == "Movie").all()
    assert (result["release_year"] > 2004).all()


def test_sort_by_release_year(sample_df):
    result = cl.sort_by_release_year(sample_df, ascending=True)
    assert result["release_year"].is_monotonic_increasing


def test_group_by_country(sample_df):
    grouped = cl.group_by_country(sample_df)
    assert "title_count" in grouped.columns
    assert grouped.loc["India", "title_count"] == 4


def test_summary_statistics(sample_df):
    stats = cl.summary_statistics(sample_df)
    assert "release_year" in stats.index


def test_label_encode(sample_df):
    encoded = cl.label_encode(sample_df, ["type"])
    assert "type_encoded" in encoded.columns
    assert encoded["type_encoded"].nunique() == 2


def test_top_country_encoding(sample_df):
    encoded = cl.top_country_encoding(sample_df, top_n=1)
    country_cols = [c for c in encoded.columns if c.startswith("country_")]
    assert country_cols


def test_run_pipeline(tmp_path):
    df = pd.DataFrame(
        {
            "show_id": ["s1", "s1", "s2"],
            "type": ["Movie", "Movie", "TV Show"],
            "title": ["A", "A", "B"],
            "director": ["D", "D", np.nan],
            "cast": ["C", "C", "C"],
            "country": ["India", "India", "USA"],
            "date_added": ["September 25, 2021"] * 3,
            "release_year": [2020, 2020, 2021],
            "rating": ["TV-MA"] * 3,
            "duration": ["90 min", "90 min", "2 Seasons"],
            "listed_in": ["Drama"] * 3,
            "description": ["x", "x", "y"],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    in_path = str(raw_dir / "netflix.csv")
    out_path = str(tmp_path / "processed" / "clean.csv")
    df.to_csv(in_path, index=False)

    result = cl.run_pipeline(in_path, out_path, str(tmp_path / "fig"))
    assert os.path.exists(out_path)
    assert os.path.exists(result["null_plot"])
    assert os.path.exists(result["corr_plot"])
