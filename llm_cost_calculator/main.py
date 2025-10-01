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

from llm_costs_framework.data_loader import load_spreadsheet
from llm_costs_framework.pages import always_on_hosting_page, per_request_pricing_page


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
        ["Always-on Hosting", "Per-Request Pricing"]
    )

    if page == "Always-on Hosting":
        always_on_hosting_page(sheet_name_to_df)
    else:
        per_request_pricing_page(sheet_name_to_df)


if __name__ == "__main__":
    main()
