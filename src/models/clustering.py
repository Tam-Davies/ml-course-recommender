import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prepare_user_features(
    user_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize user-profile genre features.
    """
    feature_columns = [
        column
        for column in user_profile.columns
        if column != "user"
    ]

    features = user_profile[feature_columns]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        scaled_features,
        index=user_profile["user"],
        columns=feature_columns,
    )

    return scaled_df, scaler


def apply_pca(
    scaled_features: pd.DataFrame,
    variance_threshold: float = 0.90,
) -> tuple[pd.DataFrame, PCA]:
    """
    Reduce user-profile dimensions while retaining the requested
    proportion of explained variance.
    """
    pca_full = PCA()
    pca_full.fit(scaled_features)

    cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

    n_components = (
        (cumulative_variance >= variance_threshold).argmax() + 1
    )

    pca = PCA(n_components=n_components)

    transformed = pca.fit_transform(scaled_features)

    columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    pca_df = pd.DataFrame(
        transformed,
        index=scaled_features.index,
        columns=columns,
    )

    return pca_df, pca


def calculate_kmeans_inertia(
    features: pd.DataFrame,
    cluster_range: range,
) -> pd.DataFrame:
    """
    Calculate K-means inertia for a range of cluster counts.
    """
    results = []

    for n_clusters in cluster_range:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )

        model.fit(features)

        results.append(
            {
                "n_clusters": n_clusters,
                "inertia": model.inertia_,
            }
        )

    return pd.DataFrame(results)


def fit_kmeans(
    features: pd.DataFrame,
    n_clusters: int,
) -> tuple[pd.DataFrame, KMeans]:
    """
    Fit K-means and return cluster assignments.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(features)

    clustered_features = features.copy()
    clustered_features["cluster"] = labels

    return clustered_features, model