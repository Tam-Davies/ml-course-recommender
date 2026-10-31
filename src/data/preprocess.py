import pandas as pd


def validate_no_missing_values(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, bool]:
    """Check whether each dataset contains missing values."""
    return {
        name: not df.isnull().values.any()
        for name, df in datasets.items()
    }


def validate_no_duplicates(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, bool]:
    """Check whether each dataset contains duplicate rows."""
    return {
        name: not df.duplicated().any()
        for name, df in datasets.items()
    }


def get_dataset_summary(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return a compact summary of dataset dimensions."""
    return pd.DataFrame(
        {
            name: {
                "rows": df.shape[0],
                "columns": df.shape[1],
            }
            for name, df in datasets.items()
        }
    ).T