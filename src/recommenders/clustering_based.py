import pandas as pd


class ClusterBasedRecommender:
    """Recommend popular unseen courses from a user's cluster."""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n
        self.user_clusters = None
        self.cluster_popular_courses = None

    def fit(
        self,
        user_clusters: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> "ClusterBasedRecommender":
        """
        Build a ranking of popular courses within each user cluster.
        """
        cluster_mapping = user_clusters["cluster"].to_dict()

        ratings_with_clusters = ratings.copy()

        ratings_with_clusters["cluster"] = (
            ratings_with_clusters["user"].map(cluster_mapping)
        )

        ratings_with_clusters = ratings_with_clusters.dropna(
            subset=["cluster"]
        )

        self.user_clusters = cluster_mapping

        self.cluster_popular_courses = (
            ratings_with_clusters
            .groupby(["cluster", "item"])
            .size()
            .reset_index(name="rating_count")
            .sort_values(
                ["cluster", "rating_count"],
                ascending=[True, False],
            )
        )

        return self

    def recommend(
        self,
        user_id: int,
        ratings: pd.DataFrame,
    ) -> pd.DataFrame:
        """Recommend unseen popular courses from the user's cluster."""
        if self.user_clusters is None:
            raise RuntimeError(
                "The recommender must be fitted before generating recommendations."
            )

        if user_id not in self.user_clusters:
            raise ValueError(
                f"User {user_id} is not present in the cluster assignments."
            )

        cluster = self.user_clusters[user_id]

        enrolled_courses = set(
            ratings.loc[
                ratings["user"] == user_id,
                "item",
            ]
        )

        recommendations = (
            self.cluster_popular_courses
            .loc[
                self.cluster_popular_courses["cluster"] == cluster
            ]
            .loc[
                ~self.cluster_popular_courses["item"].isin(
                    enrolled_courses
                )
            ]
            .head(self.top_n)
            .rename(columns={"item": "COURSE_ID"})
        )

        return recommendations[
            ["COURSE_ID", "rating_count"]
        ].reset_index(drop=True)