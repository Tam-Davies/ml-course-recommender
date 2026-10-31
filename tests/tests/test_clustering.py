import pandas as pd

from src.models.clustering import (
    apply_pca,
    fit_kmeans,
    prepare_user_features,
)


def test_prepare_user_features_excludes_user_id():
    profiles = pd.DataFrame(
        {
            "user": [1, 2, 3],
            "DataScience": [1, 0, 1],
            "Database": [0, 1, 1],
        }
    )

    features, scaler = prepare_user_features(profiles)

    assert "user" not in features.columns
    assert features.shape == (3, 2)
    assert scaler is not None


def test_pca_preserves_requested_variance_threshold():
    profiles = pd.DataFrame(
        {
            "user": [1, 2, 3, 4],
            "A": [1, 1, 0, 0],
            "B": [0, 1, 1, 0],
            "C": [0, 0, 1, 1],
        }
    )

    features, _ = prepare_user_features(profiles)
    pca_features, pca = apply_pca(features, variance_threshold=0.90)

    assert len(pca_features) == len(profiles)
    assert pca.n_components == pca_features.shape[1]


def test_fit_kmeans_returns_cluster_labels():
    features = pd.DataFrame(
        {
            "A": [0.0, 0.1, 10.0, 10.1],
            "B": [0.0, 0.1, 10.0, 10.1],
        }
    )

    clustered, model = fit_kmeans(features, n_clusters=2)

    assert "cluster" in clustered.columns
    assert len(clustered["cluster"].unique()) == 2
    assert model.n_clusters == 2
