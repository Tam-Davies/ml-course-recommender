import numpy as np
import pandas as pd


def get_genre_columns(course_genre: pd.DataFrame) -> list[str]:
    """Return course genre columns."""
    return [
        column
        for column in course_genre.columns
        if column not in ["COURSE_ID", "TITLE"]
    ]


def calculate_recommendation_scores(
    user_profile: pd.DataFrame,
    course_genre: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate user-course recommendation scores using
    the dot product between user profiles and course genres.
    """
    genre_columns = get_genre_columns(course_genre)

    user_ids = user_profile["user"].values
    user_features = user_profile[genre_columns].to_numpy()

    course_ids = course_genre["COURSE_ID"].values
    course_features = course_genre[genre_columns].to_numpy()

    scores = np.matmul(user_features, course_features.T)

    return pd.DataFrame(
        scores,
        index=user_ids,
        columns=course_ids,
    )


def get_user_recommendations(
    user_id: int,
    recommendation_scores: pd.DataFrame,
    course_genre: pd.DataFrame,
    ratings: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the highest-scoring courses that the user
    has not previously rated.
    """
    if user_id not in recommendation_scores.index:
        raise ValueError(f"User {user_id} is not present in the user profiles.")

    user_scores = recommendation_scores.loc[user_id].copy()

    enrolled_courses = set(
        ratings.loc[ratings["user"] == user_id, "item"]
    )

    user_scores = user_scores.drop(
        labels=[course for course in enrolled_courses if course in user_scores.index],
        errors="ignore",
    )

    recommendations = (
        user_scores
        .sort_values(ascending=False)
        .head(top_n)
        .rename("score")
        .reset_index()
        .rename(columns={"index": "COURSE_ID"})
    )

    recommendations = recommendations.merge(
        course_genre[["COURSE_ID", "TITLE"]],
        on="COURSE_ID",
        how="left",
    )

    return recommendations[
        ["COURSE_ID", "TITLE", "score"]
    ]