import pandas as pd
from surprise import NMF
from surprise import Reader
from surprise import Dataset


def prepare_surprise_data(
    ratings: pd.DataFrame,
    rating_scale: tuple[int, int] = (1, 5),
):
    """Convert ratings into a Surprise dataset."""

    reader = Reader(rating_scale=rating_scale)

    return Dataset.load_from_df(
        ratings[["user", "item", "rating"]],
        reader,
    )


def train_nmf_model(
    trainset,
    n_factors: int = 15,
    random_state: int = 42,
):
    """Train an NMF collaborative filtering model."""

    model = NMF(
        n_factors=n_factors,
        random_state=random_state,
    )

    model.fit(trainset)

    return model