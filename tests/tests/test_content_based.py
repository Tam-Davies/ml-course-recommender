import numpy as np
import pandas as pd

from src.features.user_profiles import calculate_recommendation_scores
from src.recommenders.content_based import (
    UserProfileRecommender,
    recommend_from_course_similarity,
)


def sample_course_genre():
    return pd.DataFrame(
        {
            "COURSE_ID": ["A", "B", "C"],
            "TITLE": ["Python", "SQL", "Machine Learning"],
            "DataScience": [1, 0, 1],
            "Database": [0, 1, 0],
        }
    )


def sample_profiles():
    return pd.DataFrame(
        {
            "user": [1, 2],
            "DataScience": [1, 0],
            "Database": [0, 1],
        }
    )


def test_calculate_recommendation_scores_shape():
    scores = calculate_recommendation_scores(
        sample_profiles(), sample_course_genre()
    )

    assert scores.shape == (2, 3)
    assert list(scores.index) == [1, 2]
    assert list(scores.columns) == ["A", "B", "C"]


def test_user_profile_recommender_excludes_seen_courses():
    profiles = sample_profiles()
    courses = sample_course_genre()
    ratings = pd.DataFrame(
        {"user": [1], "item": ["A"], "rating": [3]}
    )

    recommender = UserProfileRecommender(top_n=2).fit(profiles, courses)
    recommendations = recommender.recommend(1, courses, ratings)

    assert "A" not in recommendations["COURSE_ID"].tolist()
    assert len(recommendations) <= 2


def test_course_similarity_recommender_returns_expected_columns():
    courses = sample_course_genre()
    ratings = pd.DataFrame(
        {"user": [1], "item": ["A"], "rating": [3]}
    )
    similarity = pd.DataFrame(
        np.array(
            [
                [1.0, 0.2, 0.9],
                [0.2, 1.0, 0.1],
                [0.9, 0.1, 1.0],
            ]
        ),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    result = recommend_from_course_similarity(
        1, ratings, similarity, courses, top_n=2
    )

    assert list(result.columns) == ["COURSE_ID", "TITLE", "similarity"]
    assert "A" not in result["COURSE_ID"].tolist()
    assert result.iloc[0]["COURSE_ID"] == "C"
