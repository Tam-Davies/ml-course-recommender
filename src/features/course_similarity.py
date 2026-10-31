import pandas as pd
from scipy.spatial.distance import cosine


def create_bow_matrix(
    courses_bow: pd.DataFrame,
) -> pd.DataFrame:
    """Convert long-format BoW data into a course-term matrix."""
    return (
        courses_bow
        .pivot_table(
            index="doc_id",
            columns="token",
            values="bow",
            fill_value=0,
        )
        .sort_index()
    )


def calculate_course_similarity(
    bow_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pairwise cosine similarity between courses."""
    course_ids = bow_matrix.index.tolist()

    similarity_matrix = []

    for course_id in course_ids:
        base_vector = bow_matrix.loc[course_id].to_numpy()

        similarities = []

        for compared_id in course_ids:
            compared_vector = bow_matrix.loc[compared_id].to_numpy()

            if course_id == compared_id:
                similarity = 1.0
            else:
                similarity = 1 - cosine(
                    base_vector,
                    compared_vector,
                )

            similarities.append(similarity)

        similarity_matrix.append(similarities)

    return pd.DataFrame(
        similarity_matrix,
        index=course_ids,
        columns=course_ids,
    )


def get_similar_courses(
    course_id: str,
    similarity_matrix: pd.DataFrame,
    courses: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the most similar courses to a selected course."""
    scores = (
        similarity_matrix.loc[course_id]
        .drop(labels=course_id)
        .sort_values(ascending=False)
        .head(top_n)
        .rename("similarity")
        .reset_index()
        .rename(columns={"index": "COURSE_ID"})
    )

    return scores.merge(
        courses[["COURSE_ID", "TITLE"]],
        on="COURSE_ID",
        how="left",
    )[["COURSE_ID", "TITLE", "similarity"]]