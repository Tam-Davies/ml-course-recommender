import numpy as np
import pandas as pd

from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def extract_embedding_features(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
) -> np.ndarray:
    """Combine user and item embeddings using element-wise addition."""
    return user_embeddings + item_embeddings


def train_regression_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Train the regression models used in the course workflow."""

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def train_classification_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Train classification models using embedding features."""

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=42,
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models