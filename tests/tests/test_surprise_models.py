import pandas as pd

from src.models.knn import prepare_surprise_data as prepare_knn_data
from src.models.nmf import prepare_surprise_data as prepare_nmf_data


def sample_ratings():
    return pd.DataFrame(
        {
            "user": [1, 1, 2, 2],
            "item": ["A", "B", "A", "B"],
            "rating": [3, 2, 2, 3],
        }
    )


def test_knn_prepare_surprise_data():
    dataset = prepare_knn_data(sample_ratings())
    assert dataset.raw_ratings
    assert len(dataset.raw_ratings) == 4


def test_nmf_prepare_surprise_data():
    dataset = prepare_nmf_data(sample_ratings())
    assert dataset.raw_ratings
    assert len(dataset.raw_ratings) == 4
