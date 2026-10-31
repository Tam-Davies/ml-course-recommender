import numpy as np

from src.models.embedding_models import (
    extract_embedding_features,
    train_classification_models,
    train_regression_models,
)


def test_extract_embedding_features_adds_vectors():
    user = np.array([[1.0, 2.0], [3.0, 4.0]])
    item = np.array([[0.5, 1.5], [1.0, 2.0]])

    result = extract_embedding_features(user, item)

    expected = np.array([[1.5, 3.5], [4.0, 6.0]])

    np.testing.assert_allclose(result, expected)


def test_regression_models_train_and_predict():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    models = train_regression_models(X, y)

    assert set(models) == {
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
    }

    for model in models.values():
        predictions = model.predict(X)
        assert predictions.shape == y.shape


def test_classification_models_train_and_predict():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]]
    )
    y = np.array([0, 0, 1, 1])

    models = train_classification_models(X, y)

    assert set(models) == {"LogisticRegression", "RandomForest"}

    for model in models.values():
        predictions = model.predict(X)
        assert predictions.shape == y.shape
