"""Streamlit page implementations for LLM Cost Calculator."""

import pandas as pd
import streamlit as st

from llm_cost_calculator.utils import get_pricing_policies
from llm_cost_calculator.vram_calculator import calculate_min_vram_required
from llm_cost_calculator.data_loader import prepare_performance_scores, prepare_pricing_dataframe
from llm_cost_calculator.cost_calculator import calculate_paid_api_costs, calculate_opensource_costs


def always_on_hosting_page(sheet_name_to_df: dict[str, pd.DataFrame]):
    """Always-on hosting cost comparison page."""
    st.markdown(
        "This page helps you compare the cost of different **open-source LLMs** for always-on hosting. "
        "For comparison, check Google's [Provisioned Throughput Estimator]"
        "(https://console.cloud.google.com/vertex-ai/provisioned-throughput/price-estimate) "
        "for proprietary model pricing."
    )
    st.info(
        get_pricing_policies()
    )

    model_name_to_rows = {
        row["LLM"]: row
        for _, row in sheet_name_to_df["OpenSource-LLMs"].iterrows()
    }

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

    # Get model parameters
    row = model_name_to_rows[model_name]
    min_vram_fp8, min_vram_fp16 = calculate_min_vram_required(
        model_vram_fp8=row["Model VRAM in FP8"],
        model_vram_fp16=row["Model VRAM in FP16"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_layers=row["Number layers"],
        hidden_dim=row["Hidden dimension"],
        batch_size=number_of_requests_in_parallel
    )

    # Select the VRAM based on precision
    min_vram = min_vram_fp8 if precision == "FP8" else min_vram_fp16

    gpu_pricing_df = sheet_name_to_df["GPU pricing comparison"]

    # Prepare GPU pricing dataframe
    price_columns = [
        "On-Demand Price/month ($)",
        "Spot Price/month ($)",
        "Regular Provisioning Model (per month)\nNo subscription",
        "Regular Provisioning Model (per month)\n1 year subscription",
        "Regular Provisioning Model (per month)\n3 year subscription",
        "Spot Provisioning Model (per month)",
        "GPU total VRAM"
    ]
    gpu_pricing_df = prepare_pricing_dataframe(gpu_pricing_df, price_columns)

    valid_gpus = gpu_pricing_df[gpu_pricing_df["GPU total VRAM"] >= min_vram]

    # Display VRAM requirements
    st.subheader(f"Minimum VRAM Requirements ({precision})")
    st.metric("VRAM Required", f"{min_vram:.2f} GB")

    st.divider()

    # Display GPU options
    st.subheader(f"Cheapest GPU Options for {precision}")
    if not valid_gpus.empty:
        results = []
        for col in price_columns[:-1]:  # Exclude GPU total VRAM
            if col in valid_gpus.columns:
                col_data = valid_gpus[[col, "Instance", "Provider", "GPU total VRAM"]].dropna(subset=[col])
                if not col_data.empty:
                    cheapest = col_data.loc[col_data[col].idxmin()]
                    results.append({
                        "Pricing Model": col,
                        "Cost ($/month)": cheapest[col],
                        "Provider": cheapest['Provider'],
                        "Instance": cheapest['Instance'],
                        "VRAM (GB)": cheapest['GPU total VRAM'],
                    })

        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, width="stretch")
        else:
            st.warning("No valid pricing data available")
    else:
        st.warning(f"No GPU found that meets the {precision} VRAM requirement")


def per_request_pricing_page(sheet_name_to_df: dict[str, pd.DataFrame]):
    """Per-request pricing comparison page."""
    st.markdown(
        "Compare the cost of running a specific number of requests across "
        "**paid APIs** and **open-source models**."
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

    # Prepare performance scores
    lmarena_scores = prepare_performance_scores(performance_df)

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
        paid_apis_df, num_requests, input_tokens, output_tokens, lmarena_scores
    )

    opensource_results = calculate_opensource_costs(
        opensource_df, gpu_pricing_df, num_requests,
        input_tokens, output_tokens, lmarena_scores, precision
    )

    # Display Paid API results
    if paid_results:
        st.subheader("💳 Paid API Cost Comparison")
        paid_df = pd.DataFrame(paid_results)
        paid_df = paid_df.sort_values(by="Total Cost ($)", ascending=True).reset_index(drop=True)

        paid_format_dict = {
            "Total Cost ($)": "${:,.4f}",
            "LMArena Score": lambda x: "N/A" if (x == "N/A" or (isinstance(x, float) and pd.isna(x))) else f"{int(x)}"
        }

        st.dataframe(
            paid_df.style.format(paid_format_dict).highlight_min(
                subset=["Total Cost ($)"], color='lightgreen'
            ),
            use_container_width=True
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
        column_order = ["Provider", "Instance", "Model", "Total Cost ($)", "LMArena Score"]

        # Add optional columns if they exist
        if "Billable Hours" in opensource_df.columns:
            column_order.append("Billable Hours")
        if "Pricing Policy" in opensource_df.columns:
            column_order.append("Pricing Policy")
        if "Batch Size" in opensource_df.columns:
            column_order.append("Batch Size")

        opensource_df = opensource_df[column_order]

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
            use_container_width=True
        )
    else:
        st.warning("No open-source pricing data available")
