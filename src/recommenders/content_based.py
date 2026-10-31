import pandas as pd

from src.features.user_profiles import (
    calculate_recommendation_scores,
    get_user_recommendations,
)


class UserProfileRecommender:
    """Content-based recommender using user profiles and course genres."""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n
        self.scores = None

    def fit(
        self,
        user_profile: pd.DataFrame,
        course_genre: pd.DataFrame,
    ) -> "UserProfileRecommender":
        """Calculate recommendation scores for all users."""
        self.scores = calculate_recommendation_scores(
            user_profile,
            course_genre,
        )

        return self

    def recommend(
        self,
        user_id: int,
        course_genre: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate recommendations for one user."""
        if self.scores is None:
            raise RuntimeError(
                "The recommender must be fitted before generating recommendations."
            )

        return get_user_recommendations(
            user_id=user_id,
            recommendation_scores=self.scores,
            course_genre=course_genre,
            ratings=ratings,
            top_n=self.top_n,
        )


def recommend_from_course_similarity(
    user_id: int,
    ratings: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    courses: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend unseen courses based on courses previously rated
    by the user and their course-to-course similarity.
    """
    user_courses = ratings.loc[
        ratings["user"] == user_id,
        "item",
    ].unique()

    user_courses = [
        course
        for course in user_courses
        if course in similarity_matrix.index
    ]

    if not user_courses:
        return pd.DataFrame(
            columns=["COURSE_ID", "TITLE", "similarity"]
        )

    candidate_scores = similarity_matrix.loc[user_courses].mean(axis=0)

    candidate_scores = candidate_scores.drop(
        labels=user_courses,
        errors="ignore",
    )

    recommendations = (
        candidate_scores
        .sort_values(ascending=False)
        .head(top_n)
        .rename("similarity")
        .reset_index()
        .rename(columns={"index": "COURSE_ID"})
    )

    return recommendations.merge(
        courses[["COURSE_ID", "TITLE"]],
        on="COURSE_ID",
        how="left",
    )[["COURSE_ID", "TITLE", "similarity"]]