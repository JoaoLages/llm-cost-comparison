"""Data loading utilities for LLM costs framework."""

import math
import pandas as pd
import streamlit as st
# import gspread
# from google.auth import default


@st.cache_data(ttl=60)
def load_spreadsheet() -> dict[str, pd.DataFrame]:
    """
    Load all sheets from the Excel spreadsheet.

    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    return pd.read_excel(
        "llm_costs_framework/LLM Costs (September 2025).xlsx",
        sheet_name=None
    )

    # TODO: Use Google Sheets once permissions are configured
    # sheet_id = "1sB6qmpUIg60kWDVTK6lA5PLXjomzVmwSAj7BZMvXHEw"
    # credentials, _ = default(scopes=[
    #     "https://www.googleapis.com/auth/spreadsheets.readonly",
    #     "https://www.googleapis.com/auth/drive.readonly"
    # ])
    # gc = gspread.authorize(credentials)
    # spreadsheet = gc.open_by_key(sheet_id)
    # sheet_name_to_df = {}
    # for worksheet in spreadsheet.worksheets():
    #     df = pd.DataFrame(worksheet.get_all_records())
    #     sheet_name_to_df[worksheet.title] = df
    # return sheet_name_to_df


def prepare_performance_scores(performance_df: pd.DataFrame) -> dict:
    """
    Create a mapping from model name to LMArena score.

    Args:
        performance_df: DataFrame with Model and LMArena columns

    Returns:
        Dictionary mapping model names to rounded LMArena scores
    """
    scores = dict(zip(performance_df["Model"], performance_df["LMArena"]))
    return {
        k: round(v) if not math.isnan(v) else v
        for k, v in scores.items()
    }


def prepare_pricing_dataframe(
    df: pd.DataFrame,
    numeric_columns: list[str]
) -> pd.DataFrame:
    """
    Prepare pricing dataframe by converting columns to numeric.

    Args:
        df: Input dataframe
        numeric_columns: List of column names to convert to numeric

    Returns:
        DataFrame with numeric columns converted
    """
    df = df.copy()
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
