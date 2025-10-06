"""Data loading utilities shared across pages."""

import pandas as pd
import streamlit as st
from odf import opendocument, table, text


@st.cache_data(ttl=60)
def load_spreadsheet() -> dict[str, pd.DataFrame]:
    """
    Load all sheets from the ODS spreadsheet.

    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    return pd.read_excel(
        "llm_cost_calculator/LLM Costs (September 2025).ods",
        sheet_name=None,
        engine="odf"
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


@st.cache_data(ttl=300)
def extract_hyperlinks_from_ods(
    file_path: str = "llm_cost_calculator/LLM Costs (September 2025).ods",
    sheet_name: str = "Performance comparison"
) -> dict[str, dict[str, str]]:
    """
    Extract hyperlinks from an ODS file.

    Args:
        file_path: Path to the ODS file
        sheet_name: Name of the sheet to extract links from

    Returns:
        Dictionary with two keys:
        - 'headers': dict mapping column names to their hyperlink URLs
        - 'data': dict mapping cell values to their hyperlink URLs
    """
    doc = opendocument.load(file_path)

    # Find the specified sheet
    for sheet in doc.spreadsheet.getElementsByType(table.Table):
        if sheet.getAttribute('name') == sheet_name:
            rows = sheet.getElementsByType(table.TableRow)

            header_links = {}
            data_links = {}

            if len(rows) > 0:
                # Extract header links
                header_row = rows[0]
                header_cells = header_row.getElementsByType(table.TableCell)

                for cell in header_cells:
                    cell_text = ''.join([str(p) for p in cell.getElementsByType(text.P)])
                    links = cell.getElementsByType(text.A)
                    if links and cell_text:
                        link_url = links[0].getAttribute('href')
                        header_links[cell_text] = link_url

                # Extract data links
                for row in rows[1:]:
                    cells = row.getElementsByType(table.TableCell)
                    for cell in cells:
                        cell_text = ''.join([str(p) for p in cell.getElementsByType(text.P)])
                        links = cell.getElementsByType(text.A)
                        if links and cell_text:
                            link_url = links[0].getAttribute('href')
                            data_links[cell_text] = link_url

            return {'headers': header_links, 'data': data_links}

    return {'headers': {}, 'data': {}}
