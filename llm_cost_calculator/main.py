"""
LLM Cost Calculator - Main Application

A Streamlit application for comparing costs of different LLM hosting options,
including both paid APIs and self-hosted open-source models.
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path to support both direct execution and package import
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_cost_calculator.common import load_spreadsheet
from llm_cost_calculator.pages.always_on_hosting import always_on_hosting_page
from llm_cost_calculator.pages.per_request_pricing import per_request_pricing_page


def main():
    """Main function to run the Streamlit application."""
    load_dotenv()

    st.set_page_config(
        page_title="LLM Cost Calculator",
        page_icon="💰",
        layout="wide"
    )

    st.title("💰 LLM Cost Calculator")

    # Load data
    sheet_name_to_df = load_spreadsheet()

    # Page selection using radio buttons
    page = st.sidebar.radio(
        "Select Page",
        ["Per-Request Pricing", "Always-on Hosting",]
    )

    if page == "Always-on Hosting":
        always_on_hosting_page(sheet_name_to_df)
    elif page == "Per-Request Pricing":
        per_request_pricing_page(sheet_name_to_df)
    else:
        st.error("Invalid page selection.")


if __name__ == "__main__":
    main()
