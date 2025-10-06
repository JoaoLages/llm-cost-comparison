"""Data utilities specific to per-request pricing page."""

import math
import pandas as pd


def prepare_performance_scores(performance_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create mappings from model name to LMArena score and category.

    Args:
        performance_df: DataFrame with Model, LMArena, and Category columns

    Returns:
        Tuple of (scores_dict, categories_dict) where:
        - scores_dict: Dictionary mapping model names to rounded LMArena scores
        - categories_dict: Dictionary mapping model names to categories
    """
    scores = dict(zip(performance_df["Model"], performance_df["LMArena"]))
    scores_dict = {
        k: round(v) if (isinstance(v, (int, float)) and not math.isnan(v)) else v
        for k, v in scores.items()
    }
    categories_dict = dict(zip(performance_df["Model"], performance_df["Category"]))
    return scores_dict, categories_dict
