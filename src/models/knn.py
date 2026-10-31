import pandas as pd
from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset


def prepare_surprise_data(
    ratings: pd.DataFrame,
    rating_scale: tuple[int, int] = (1, 5),
):
    """Convert the ratings dataframe into a Surprise dataset."""

    reader = Reader(rating_scale=rating_scale)

    data = Dataset.load_from_df(
        ratings[["user", "item", "rating"]],
        reader,
    )

    return data


def train_knn_model(
    trainset,
    user_based: bool = False,
    similarity_name: str = "pearson",
):
    """Train a KNN collaborative filtering model."""

    sim_options = {
        "name": similarity_name,
        "user_based": user_based,
    }

    model = KNNBasic(
        sim_options=sim_options,
    )

    model.fit(trainset)

    return model