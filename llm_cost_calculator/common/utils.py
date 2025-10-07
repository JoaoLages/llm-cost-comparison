import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Column name constants - centralized to avoid duplication
COL_ON_DEMAND_HOURLY = "On-Demand Price/hr ($)"
COL_SPOT_HOURLY = "Spot Price/hr ($)"
COL_ON_DEMAND_MONTHLY = "On-Demand Price/month ($)"
COL_SPOT_MONTHLY = "Spot Price/month ($)"
COL_REGULAR_NO_SUB = "Regular Provisioning Model (per month)No subscription"
COL_REGULAR_1_YEAR = "Regular Provisioning Model (per month)1 year subscription"
COL_REGULAR_3_YEAR = "Regular Provisioning Model (per month)3 year subscription"
COL_SPOT_PROVISIONING = "Spot Provisioning Model (per month)"

HOURLY_PRICE_COLUMNS = {
    COL_ON_DEMAND_HOURLY,
    COL_SPOT_HOURLY,
}

MONTHLY_PRICE_COLUMNS = {
    COL_ON_DEMAND_MONTHLY,
    COL_SPOT_MONTHLY,
    COL_REGULAR_NO_SUB,
    COL_REGULAR_1_YEAR,
    COL_REGULAR_3_YEAR,
    COL_SPOT_PROVISIONING,
}

# Pricing model mapping: policy name -> list of actual column names in spreadsheet
PRICING_MODEL_MAPPING = {
    "On-Demand": [COL_ON_DEMAND_MONTHLY, COL_REGULAR_NO_SUB],
    "Spot": [COL_SPOT_MONTHLY, COL_SPOT_PROVISIONING],
    "1-Year Subscription": [COL_REGULAR_1_YEAR],
    "3-Year Subscription": [COL_REGULAR_3_YEAR]
}

# Reverse mapping for display: actual column name -> policy name
REVERSE_PRICING_MODEL_MAPPING = {
    col: policy_name
    for policy_name, col_list in PRICING_MODEL_MAPPING.items()
    for col in col_list
}


def get_pricing_policies():
    return (
        "**Pricing policy differences:**\n"
        "- **On-Demand**: Pay only for actual usage, can suspend anytime\n"
        "- **Subscription**: Pay for full year(s) of subscription regardless of usage\n"
        "- **Spot**: Cheaper but bills in full-hour increments (rounded up) and be terminated by provider with only 2-minutes notice\n"
        "\n"
        "Note: [Vertex AI Model Garden](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models)"
        " allows us to deploy LLMs on-demand with no infrastructure management (pricing not included in this tool, but they are usually 20-30% higher than the 'On-Demand')\n"
    )


def filter_dataframe(df: pd.DataFrame, key_prefix: str = "") -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.

    Args:
        df: DataFrame to filter
        key_prefix: Prefix for widget keys to avoid conflicts when using multiple filters

    Returns:
        Filtered dataframe
    """
    modify = st.checkbox("Add filters", key=f"{key_prefix}_add_filters")

    if not modify:
        return df

    df = df.copy()

    # Convert potential datetime columns
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            df.columns,
            key=f"{key_prefix}_filter_columns"
        )

        for column in to_filter_columns:
            left, right = st.columns((1, 20))

            # Treat objects as categorical if they have few unique values
            if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=f"{key_prefix}_cat_{column}"
                )
                df = df[df[column].isin(user_cat_input)]

            # Numeric column handling
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100 if _max != _min else 1.0
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                    key=f"{key_prefix}_num_{column}"
                )
                df = df[df[column].between(*user_num_input)]

            # Datetime column handling
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    key=f"{key_prefix}_date_{column}"
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]

            # Text column handling
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=f"{key_prefix}_text_{column}"
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input, case=False)]

    return df