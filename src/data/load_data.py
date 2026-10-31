from pathlib import Path

import pandas as pd


def load_course_genre(data_dir: Path) -> pd.DataFrame:
    """Load course genre/features data."""
    return pd.read_csv(data_dir / "course_genre.csv")


def load_courses(data_dir: Path) -> pd.DataFrame:
    """Load processed course text data."""
    return pd.read_csv(data_dir / "course_processed.csv")


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load user-course rating data."""
    return pd.read_csv(data_dir / "ratings.csv")


def load_user_profiles(data_dir: Path) -> pd.DataFrame:
    """Load user profile feature data."""
    return pd.read_csv(data_dir / "user_profile.csv")