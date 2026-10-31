import pandas as pd

from src.features.course_similarity import (
    calculate_course_similarity,
    create_bow_matrix,
    get_similar_courses,
)


def test_create_bow_matrix():
    bow = pd.DataFrame(
        {
            "doc_id": ["A", "A", "B"],
            "token": ["python", "data", "python"],
            "bow": [2, 1, 3],
        }
    )

    matrix = create_bow_matrix(bow)

    assert matrix.loc["A", "python"] == 2
    assert matrix.loc["A", "data"] == 1
    assert matrix.loc["B", "python"] == 3
    assert matrix.loc["B", "data"] == 0


def test_similarity_diagonal_is_one():
    matrix = pd.DataFrame(
        {"python": [2, 2], "data": [1, 0]},
        index=["A", "B"],
    )

    similarity = calculate_course_similarity(matrix)

    assert similarity.loc["A", "A"] == 1.0
    assert similarity.loc["B", "B"] == 1.0


def test_get_similar_courses_excludes_source_course():
    matrix = pd.DataFrame(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.1], [0.2, 0.1, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    courses = pd.DataFrame(
        {
            "COURSE_ID": ["A", "B", "C"],
            "TITLE": ["A title", "B title", "C title"],
        }
    )

    result = get_similar_courses("A", matrix, courses, top_n=2)

    assert "A" not in result["COURSE_ID"].tolist()
    assert result.iloc[0]["COURSE_ID"] == "B"
