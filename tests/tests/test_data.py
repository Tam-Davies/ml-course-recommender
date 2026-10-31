import pandas as pd

from src.data.preprocess import (
    get_dataset_summary,
    validate_no_duplicates,
    validate_no_missing_values,
)


def test_validate_no_missing_values():
    datasets = {
        "courses": pd.DataFrame({"COURSE_ID": ["A", "B"], "TITLE": ["x", "y"]}),
        "ratings": pd.DataFrame({"user": [1, 2], "item": ["A", "B"], "rating": [3, 2]}),
    }

    result = validate_no_missing_values(datasets)

    assert result == {"courses": True, "ratings": True}


def test_validate_no_duplicates_detects_duplicates():
    datasets = {
        "courses": pd.DataFrame(
            {"COURSE_ID": ["A", "A"], "TITLE": ["x", "x"]}
        )
    }

    result = validate_no_duplicates(datasets)

    assert result["courses"] is False


def test_dataset_summary_returns_rows_and_columns():
    datasets = {
        "courses": pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]}),
        "ratings": pd.DataFrame({"user": [1], "rating": [3]}),
    }

    summary = get_dataset_summary(datasets)

    assert summary.loc["courses", "rows"] == 2
    assert summary.loc["courses", "columns"] == 3
    assert summary.loc["ratings", "rows"] == 1
    assert summary.loc["ratings", "columns"] == 2
