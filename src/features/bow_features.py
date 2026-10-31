import re

import pandas as pd
from gensim.corpora import Dictionary


def tokenize_course(text: str) -> list[str]:
    """
    Tokenize course text, remove stopwords and numeric tokens.
    """
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


def prepare_course_text(courses: pd.DataFrame) -> pd.DataFrame:
    """Combine course title and description into one text field."""
    result = courses.copy()

    result["course_text"] = (
        result["TITLE"].fillna("")
        + " "
        + result["DESCRIPTION"].fillna("")
    )

    return result


def build_course_bow(courses: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-format Bag-of-Words representation
    using a Gensim Dictionary.
    """
    course_data = prepare_course_text(courses)

    tokenized_courses = [
        tokenize_course(text)
        for text in course_data["course_text"]
    ]

    dictionary = Dictionary(tokenized_courses)

    records = []

    for course_id, tokens in zip(
        course_data["COURSE_ID"],
        tokenized_courses,
    ):
        bow = dictionary.doc2bow(tokens)

        for token_id, count in bow:
            records.append(
                {
                    "doc_id": course_id,
                    "token": dictionary[token_id],
                    "bow": count,
                }
            )

    return pd.DataFrame(records)