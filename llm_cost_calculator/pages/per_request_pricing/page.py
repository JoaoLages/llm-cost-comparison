"""Per-request pricing comparison page implementation."""

import pandas as pd
import streamlit as st

from llm_cost_calculator.common import (
    get_pricing_policies,
    prepare_pricing_dataframe,
    filter_dataframe
)
from llm_cost_calculator.pages.per_request_pricing.data_utils import prepare_performance_scores
from llm_cost_calculator.pages.per_request_pricing.cost_calculator import (
    calculate_paid_api_costs,
    calculate_opensource_costs
)


def per_request_pricing_page(sheet_name_to_df: dict[str, pd.DataFrame]):
    """Per-request pricing comparison page."""
    st.markdown(
        "Compare the cost of running a specific number of requests across "
        "**paid APIs** and **open-source models**."
    )

    st.warning(
        "⚠️ **Important:** These estimates are for **text-only modality** and **contexts under 200K tokens**. "
        "Actual costs may vary for multimodal inputs or longer contexts."
    )

    st.info(
        get_pricing_policies()
    )

    st.info(
        "**Note on open-source estimates:** Costs are estimated based on throughput benchmarks:\n"
        "- Small models (7B params): ~50-100 tokens/sec\n"
        "- Medium models (13-70B params): ~20-50 tokens/sec\n"
        "- Large models (70B+ params): ~5-20 tokens/sec"
    )

    # Sidebar inputs
    st.sidebar.header("Per-Request Configuration")
    num_requests = st.sidebar.number_input(
        "Number of Requests",
        min_value=1, value=1000, step=100,
        key="per_request_num"
    )
    input_tokens = st.sidebar.number_input(
        "Input Tokens Per Request",
        min_value=1, value=1000, step=100,
        key="per_request_input_tokens"
    )
    output_tokens = st.sidebar.number_input(
        "Output Tokens Per Request",
        min_value=1, value=500, step=100,
        key="per_request_output_tokens"
    )
    precision = st.sidebar.selectbox(
        "Open-Source Model Precision",
        options=["FP8", "FP16"],
        index=0,
        help="FP8 uses less memory and is cheaper, FP16 may have better quality",
        key="per_request_precision"
    )

    # Load and prepare data
    paid_apis_df = sheet_name_to_df["Paid APIs - public pricing"].copy()
    opensource_df = sheet_name_to_df["OpenSource-LLMs"].copy()
    gpu_pricing_df = sheet_name_to_df["GPU pricing comparison"].copy()
    performance_df = sheet_name_to_df["Performance comparison"].copy()

    # Prepare performance scores and categories
    lmarena_scores, categories = prepare_performance_scores(performance_df)

    # Prepare paid APIs dataframe
    paid_apis_df = prepare_pricing_dataframe(
        paid_apis_df,
        ["Input Cost ($/1M tokens)", "Output Cost ($/1M tokens)"]
    )

    # Prepare GPU pricing dataframe
    gpu_pricing_df = prepare_pricing_dataframe(
        gpu_pricing_df,
        [
            "On-Demand Price/hr ($)",
            "Spot Price/hr ($)",
            "On-Demand Price/month ($)",
            "Spot Price/month ($)",
            "Regular Provisioning Model (per month)\nNo subscription",
            "Regular Provisioning Model (per month)\n1 year subscription",
            "Regular Provisioning Model (per month)\n3 year subscription",
            "Spot Provisioning Model (per month)",
            "GPU total VRAM"
        ]
    )

    # Calculate costs
    paid_results = calculate_paid_api_costs(
        paid_apis_df, num_requests, input_tokens, output_tokens, lmarena_scores, categories
    )

    opensource_results = calculate_opensource_costs(
        opensource_df, gpu_pricing_df, num_requests,
        input_tokens, output_tokens, lmarena_scores, categories, precision
    )

    # Display Paid API results
    if paid_results:
        st.subheader("💳 Paid API Cost Comparison")
        paid_df = pd.DataFrame(paid_results)
        paid_df = paid_df.sort_values(by="Total Cost ($)", ascending=True).reset_index(drop=True)

        # Apply filters
        paid_df = filter_dataframe(paid_df, key_prefix="paid_api")

        paid_format_dict = {
            "Total Cost ($)": "${:,.4f}",
            "LMArena Score": lambda x: "N/A" if (x == "N/A" or (isinstance(x, float) and pd.isna(x))) else f"{int(x)}"
        }

        st.dataframe(
            paid_df.style.format(paid_format_dict).highlight_min(
                subset=["Total Cost ($)"], color='lightgreen'
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No paid API pricing data available")

    st.divider()

    # Display Open-Source results
    if opensource_results:
        st.subheader("🖥️ Open-Source Self-Hosted Cost Comparison")
        opensource_df = pd.DataFrame(opensource_results)
        opensource_df = opensource_df.sort_values(by="Total Cost ($)", ascending=True).reset_index(drop=True)

        # Reorder columns for better display
        column_order = ["Provider", "Instance", "Model", "Total Cost ($)", "LMArena Score", "Category"]

        # Add optional columns if they exist
        if "Billable Hours" in opensource_df.columns:
            column_order.append("Billable Hours")
        if "Pricing Policy" in opensource_df.columns:
            column_order.append("Pricing Policy")
        if "Batch Size" in opensource_df.columns:
            column_order.append("Batch Size")

        opensource_df = opensource_df[column_order]

        # Apply filters
        opensource_df = filter_dataframe(opensource_df, key_prefix="opensource")

        # Format the dataframe
        opensource_format_dict = {
            "Total Cost ($)": "${:,.4f}",
            "LMArena Score": lambda x: "N/A" if (x == "N/A" or (isinstance(x, float) and pd.isna(x))) else f"{int(x)}"
        }
        if "Billable Hours" in opensource_df.columns:
            opensource_format_dict["Billable Hours"] = "{:.2f}"

        st.dataframe(
            opensource_df.style.format(opensource_format_dict).highlight_min(
                subset=["Total Cost ($)"], color='lightgreen'
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No open-source pricing data available")
