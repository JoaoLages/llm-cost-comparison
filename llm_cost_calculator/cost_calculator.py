"""Cost calculation utilities for LLM pricing."""

import pandas as pd
from typing import Optional, Literal

from llm_cost_calculator.vram_calculator import calculate_min_vram
from llm_cost_calculator.throughput_estimator import estimate_tokens_per_sec, calculate_execution_time


def calculate_paid_api_costs(
    paid_apis_df: pd.DataFrame,
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    lmarena_scores: dict
) -> list[dict]:
    """
    Calculate costs for paid API models.

    Args:
        paid_apis_df: DataFrame with paid API pricing
        num_requests: Number of requests
        input_tokens: Input tokens per request
        output_tokens: Output tokens per request
        lmarena_scores: Mapping of model names to LMArena scores

    Returns:
        List of result dictionaries
    """
    results = []

    for _, row in paid_apis_df.iterrows():
        if pd.notna(row["Input Cost ($/1M tokens)"]) and pd.notna(row["Output Cost ($/1M tokens)"]):
            input_cost = (input_tokens * num_requests * row["Input Cost ($/1M tokens)"]) / 1_000_000
            output_cost = (output_tokens * num_requests * row["Output Cost ($/1M tokens)"]) / 1_000_000
            total_cost = input_cost + output_cost

            results.append({
                "Provider": row["Provider"],
                "Model": row["Model"],
                "Total Cost ($)": total_cost,
                "LMArena Score": lmarena_scores.get(row["Model"], "N/A")
            })

    return results


def calculate_billable_cost(
    execution_hours: float,
    price_per_hour: float,
    pricing_policy: str
) -> tuple[float, float]:
    """
    Calculate billable cost based on pricing policy constraints.

    Args:
        execution_hours: Actual execution time in hours
        price_per_hour: Price per hour (always in $/hour)
        pricing_policy: Pricing policy type (must match exact names from pricing_columns)

    Returns:
        Tuple of (total_cost, billable_hours)

    Raises:
        ValueError: If pricing_policy is not recognized
    """
    import math

    HOURS_PER_MONTH = 730  # Average hours in a month (365 * 24 / 12)
    HOURS_PER_YEAR = 8760  # 365 * 24

    # Exact policy matching
    # All calculations start from hourly price and convert to appropriate billing unit

    if pricing_policy == "On-Demand":
        # On-Demand: pay only for actual usage, can start/stop anytime
        billable_hours = execution_hours
        total_cost = billable_hours * price_per_hour

    elif pricing_policy == "Spot":
        # Spot: pay for full hours (rounded up), can be interrupted by provider with 2min notice
        billable_hours = math.ceil(execution_hours)
        total_cost = billable_hours * price_per_hour

    elif pricing_policy == "1-Month Subscription":
        # 1-month subscription: pay per month, can abort anytime
        # Convert hourly price to monthly price
        price_per_month = price_per_hour * HOURS_PER_MONTH
        months_needed = math.ceil(execution_hours / HOURS_PER_MONTH)
        billable_hours = months_needed * HOURS_PER_MONTH
        total_cost = price_per_month * months_needed

    elif pricing_policy == "1-Year Subscription":
        # 1-year subscription: must pay for full year(s), can only abort each year
        # Convert hourly price to monthly price, billed yearly
        price_per_month = price_per_hour * HOURS_PER_MONTH
        years_needed = math.ceil(execution_hours / HOURS_PER_YEAR)
        months_billed = years_needed * 12
        billable_hours = years_needed * HOURS_PER_YEAR
        total_cost = price_per_month * months_billed

    elif pricing_policy == "3-Year Subscription":
        # 3-year subscription: must pay for full 3-year period(s), can only abort every 3 years
        # Convert hourly price to monthly price, billed for 3 years
        price_per_month = price_per_hour * HOURS_PER_MONTH
        three_year_periods = math.ceil(execution_hours / (HOURS_PER_YEAR * 3))
        months_billed = three_year_periods * 36
        billable_hours = three_year_periods * HOURS_PER_YEAR * 3
        total_cost = price_per_month * months_billed

    else:
        raise ValueError(
            f"Unknown pricing policy: '{pricing_policy}'. "
            f"Valid policies are: 'On-Demand', 'Spot', '1-Month Subscription', "
            f"'1-Year Subscription', '3-Year Subscription'"
        )

    return total_cost, billable_hours


def find_all_gpu_configs(
    model_row: pd.Series,
    model_name: str,
    gpu_pricing_df: pd.DataFrame,
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    price_column: str,
    pricing_policy: str,
    lmarena_scores: dict,
    precision: Literal["FP8", "FP16"] = "FP8"
) -> list[dict]:
    """
    Find all valid GPU configurations for a given model and pricing policy.

    Args:
        model_row: Row from OpenSource-LLMs dataframe
        model_name: Name of the model
        gpu_pricing_df: DataFrame with GPU pricing
        num_requests: Total number of requests
        input_tokens: Input tokens per request
        output_tokens: Output tokens per request
        price_column: Name of the price column to use
        pricing_policy: Name of the pricing policy (e.g., "On-Demand", "Spot")
        lmarena_scores: Mapping of model names to LMArena scores
        precision: Model precision ("FP8" or "FP16")

    Returns:
        List of dictionaries with all valid configurations
    """
    # Extract model parameters
    if precision == "FP8":
        model_vram = model_row["Model VRAM in FP8"]
        bytes_per_param = 1
    else:  # FP16
        model_vram = model_row["Model VRAM in FP16"]
        bytes_per_param = 2

    num_layers = model_row["Number layers"]
    hidden_dim = model_row["Hidden dimension"]
    num_params = model_row.get("Number of parameters (B)", 7)

    # Estimate throughput
    tokens_per_sec = estimate_tokens_per_sec(num_params)

    # Find all valid GPU configurations
    valid_gpus = gpu_pricing_df[pd.notna(gpu_pricing_df[price_column])].copy()
    gpu_configs = []

    for _, gpu_row in valid_gpus.iterrows():
        gpu_vram = gpu_row["GPU total VRAM"]
        price_per_hour = gpu_row[price_column]

        # Calculate maximum batch size that fits in this GPU
        max_batch_size = 1
        for batch_size in range(1, num_requests + 1):
            vram_needed = calculate_min_vram(
                model_vram, input_tokens, output_tokens,
                num_layers, hidden_dim, bytes_per_param, batch_size
            )
            if vram_needed <= gpu_vram:
                max_batch_size = batch_size
            else:
                break

        # Check if GPU can fit at least 1 sample
        vram_for_one = calculate_min_vram(
            model_vram, input_tokens, output_tokens,
            num_layers, hidden_dim, bytes_per_param, batch_size=1
        )
        if vram_for_one > gpu_vram:
            continue

        # Calculate execution time (with 1 hour buffer)
        execution_hours = calculate_execution_time(
            num_requests, input_tokens, output_tokens,
            tokens_per_sec, max_batch_size, conservative_buffer_hours=1.0
        )

        # Convert all prices to hourly rate first
        HOURLY_PRICE_COLUMNS = {
            "On-Demand Price/hr ($)",
            "Spot Price/hr ($)",
        }
        MONTHLY_PRICE_COLUMNS = {
            "On-Demand Price/month ($)",
            "Spot Price/month ($)",
            "Regular Provisioning Model (per month)\nNo subscription",
            "Regular Provisioning Model (per month)\n1 year subscription",
            "Regular Provisioning Model (per month)\n3 year subscription",
            "Spot Provisioning Model (per month)",
        }

        if price_column in HOURLY_PRICE_COLUMNS:
            hourly_rate = price_per_hour  # Already hourly
        elif price_column in MONTHLY_PRICE_COLUMNS:
            # Convert monthly to hourly (730 hours per month average)
            hourly_rate = price_per_hour / 730
        else:
            raise ValueError(f"Unknown price column: '{price_column}'")

        # Calculate billable cost (function will convert hourly to appropriate billing unit)
        total_cost, billable_hours = calculate_billable_cost(
            execution_hours, hourly_rate, pricing_policy
        )

        # Add this configuration to the list
        gpu_configs.append({
            "Provider": gpu_row["Provider"],
            "Instance": gpu_row["Instance"],
            "Model": model_name,
            "Total Cost ($)": total_cost,
            "LMArena Score": lmarena_scores.get(model_name, "N/A"),
            "Execution Hours": round(execution_hours, 2),
            "Billable Hours": round(billable_hours, 2),
            "Pricing Policy": pricing_policy,
            "Batch Size": max_batch_size
        })

    return gpu_configs


def calculate_opensource_costs(
    opensource_df: pd.DataFrame,
    gpu_pricing_df: pd.DataFrame,
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    lmarena_scores: dict,
    precision: Literal["FP8", "FP16"] = "FP8"
) -> list[dict]:
    """
    Calculate costs for open-source models across different GPU configurations.

    Args:
        opensource_df: DataFrame with open-source LLM specifications
        gpu_pricing_df: DataFrame with GPU pricing
        num_requests: Number of requests
        input_tokens: Input tokens per request
        output_tokens: Output tokens per request
        lmarena_scores: Mapping of model names to LMArena scores
        precision: Model precision ("FP8" or "FP16")

    Returns:
        List of result dictionaries
    """
    # Define pricing columns to evaluate (column_name, policy_name)
    # Note: Some providers give hourly prices, others give monthly prices
    pricing_columns = [
        ("On-Demand Price/hr ($)", "On-Demand"),
        ("Spot Price/hr ($)", "Spot"),
        ("Spot Price/month ($)", "Spot"),  # Same policy, just monthly price format
        ("Regular Provisioning Model (per month)\nNo subscription", "1-Month Subscription"),
        ("Regular Provisioning Model (per month)\n1 year subscription", "1-Year Subscription"),
        ("Regular Provisioning Model (per month)\n3 year subscription", "3-Year Subscription"),
        ("Spot Provisioning Model (per month)", "1-Month Subscription"),  # Spot provisioning = monthly spot
    ]

    # Determine which VRAM column to check
    vram_column = f"Model VRAM in {precision}"

    results = []

    for _, model_row in opensource_df.iterrows():
        model_name = model_row["LLM"]

        # Check if required columns exist
        if (pd.isna(model_row.get(vram_column)) or
            pd.isna(model_row.get("Number layers")) or
            pd.isna(model_row.get("Hidden dimension"))):
            continue

        # Find all configurations for each pricing policy
        for price_col, pricing_policy in pricing_columns:
            if price_col not in gpu_pricing_df.columns:
                continue

            configs = find_all_gpu_configs(
                model_row, model_name, gpu_pricing_df,
                num_requests, input_tokens, output_tokens,
                price_col, pricing_policy, lmarena_scores, precision
            )

            results.extend(configs)

    return results
