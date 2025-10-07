"""Always-on hosting cost comparison page implementation."""

import pandas as pd
import streamlit as st

from llm_cost_calculator.common.vram_calculator import calculate_min_vram
from llm_cost_calculator.common.data_loader import prepare_pricing_dataframe, extract_hyperlinks_from_ods
from llm_cost_calculator.common.utils import (
    get_pricing_policies,
    filter_dataframe,
    PRICING_MODEL_MAPPING,
    REVERSE_PRICING_MODEL_MAPPING,
    HOURLY_PRICE_COLUMNS,
    MONTHLY_PRICE_COLUMNS,
    COL_ON_DEMAND_HOURLY,
    COL_SPOT_HOURLY,
    COL_ON_DEMAND_MONTHLY,
    COL_SPOT_MONTHLY
)


def always_on_hosting_page(sheet_name_to_df: dict[str, pd.DataFrame]):
    """Always-on hosting cost comparison page."""
    st.markdown(
        "This page helps you compare the cost of different **open-source LLMs** for always-on hosting. "
        "For comparison, check Google's [Provisioned Throughput Estimator]"
        "(https://console.cloud.google.com/vertex-ai/provisioned-throughput/price-estimate) "
        "for proprietary model pricing."
    )
    st.warning(
        "⚠️ **Important:** These estimates are for **text-only modality** and **contexts under 200K tokens**. "
        "Actual costs may vary for multimodal inputs or longer contexts."
    )

    st.info(
        get_pricing_policies()
    )

    model_name_to_rows = {
        row["LLM"]: row
        for _, row in sheet_name_to_df["OpenSource-LLMs"].iterrows()
        if pd.notna(row["LLM"]) and str(row["LLM"]).strip()
    }

    # Extract hyperlinks from ODS file
    hyperlinks = extract_hyperlinks_from_ods()

    # Load performance data for categories
    performance_df = sheet_name_to_df.get("Performance comparison")
    model_categories = {}
    if performance_df is not None:
        model_categories = dict(zip(performance_df["Model"], performance_df["Category"]))

    # Sidebar inputs
    st.sidebar.header("Always-on Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(model_name_to_rows.keys()),
        key="always_on_model"
    )
    input_tokens = st.sidebar.number_input(
        "Input Tokens Per Request",
        min_value=1, value=1000, step=100,
        key="always_on_input_tokens"
    )
    output_tokens = st.sidebar.number_input(
        "Output Tokens Per Request",
        min_value=1, value=500, step=100,
        key="always_on_output_tokens"
    )
    number_of_requests_in_parallel = st.sidebar.number_input(
        "Number of Parallel Requests",
        min_value=1, value=1, step=1,
        key="always_on_parallel"
    )
    precision = st.sidebar.selectbox(
        "Model Precision",
        options=["FP8", "FP16"],
        index=0,
        help="FP8 uses less memory and is cheaper, FP16 may have better quality",
        key="always_on_precision"
    )

    # Pricing model filter
    selected_pricing_models_display = st.sidebar.multiselect(
        "Pricing Models",
        options=list(PRICING_MODEL_MAPPING.keys()),
        default=list(PRICING_MODEL_MAPPING.keys()),
        key="always_on_pricing_models"
    )
    selected_pricing_models = []
    for m in selected_pricing_models_display:
        selected_pricing_models.extend(PRICING_MODEL_MAPPING[m])

    # Get model parameters and calculate VRAM for selected precision
    row = model_name_to_rows[model_name]
    min_vram = calculate_min_vram(
        model_vram_fp8=row["Model VRAM in FP8"],
        model_vram_fp16=row["Model VRAM in FP16"],
        precision=precision,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_layers=row["Number layers"],
        hidden_dim=row["Hidden dimension"],
        batch_size=number_of_requests_in_parallel
    )

    gpu_pricing_df = sheet_name_to_df["GPU pricing comparison"]

    # Prepare GPU pricing dataframe
    all_columns = list(HOURLY_PRICE_COLUMNS | MONTHLY_PRICE_COLUMNS) + ["GPU total VRAM"]
    gpu_pricing_df = prepare_pricing_dataframe(gpu_pricing_df, all_columns)

    # Convert hourly prices to monthly (730 hours per month average)
    if COL_ON_DEMAND_HOURLY in gpu_pricing_df.columns:
        gpu_pricing_df[COL_ON_DEMAND_MONTHLY] = pd.to_numeric(gpu_pricing_df[COL_ON_DEMAND_HOURLY], errors='coerce') * 730
    if COL_SPOT_HOURLY in gpu_pricing_df.columns:
        gpu_pricing_df[COL_SPOT_MONTHLY] = pd.to_numeric(gpu_pricing_df[COL_SPOT_HOURLY], errors='coerce') * 730

    # Price columns to display (all monthly)
    price_columns = list(MONTHLY_PRICE_COLUMNS) + ["GPU total VRAM"]

    valid_gpus = gpu_pricing_df[gpu_pricing_df["GPU total VRAM"] >= min_vram]

    # Display VRAM requirements and category
    # Add hyperlink to model name if available
    if model_name in hyperlinks["data"]:
        st.subheader(f"[{model_name}]({hyperlinks['data'][model_name]}) - Minimum VRAM Requirements ({precision})")
    else:
        st.subheader(f"{model_name} - Minimum VRAM Requirements ({precision})")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("VRAM Required", f"{min_vram:.2f} GB")
    with col2:
        category = model_categories.get(model_name, "N/A")
        st.metric("Model Category", category)

    st.divider()

    # Display GPU options
    st.subheader(f"GPU Options for {precision}")
    if not valid_gpus.empty:
        results = []

        for col in price_columns[:-1]:  # Exclude GPU total VRAM
            if col in valid_gpus.columns and col in selected_pricing_models:
                col_data = valid_gpus[[col, "Instance", "Provider", "GPU total VRAM"]].dropna(subset=[col])
                if not col_data.empty:
                    for idx, row in col_data.iterrows():
                        results.append({
                            "Pricing Model": REVERSE_PRICING_MODEL_MAPPING.get(col, col),
                            "Cost ($/month)": row[col],
                            "Provider": row['Provider'],
                            "Instance": row['Instance'],
                            "VRAM (GB)": row['GPU total VRAM'],
                        })

        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by="Cost ($/month)", ascending=True)

            # Apply filters
            results_df = filter_dataframe(results_df, key_prefix="always_on")

            st.dataframe(results_df, width="stretch", hide_index=True)
        else:
            st.warning("No valid pricing data available")
    else:
        st.warning(f"No GPU found that meets the {precision} VRAM requirement")
