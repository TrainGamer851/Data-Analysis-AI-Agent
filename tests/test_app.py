
from pathlib import Path

import pandas as pd
from app.app import load_csv


# Locate CSV relative to the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT_ROOT / "data" / "superstore_orders.csv"


def test_csv_exists():
    # Ensure the CSV file exists before running tests
    assert CSV_FILE.exists(), f"CSV file not found at {CSV_FILE}"
    # Ensure the CSV file is a file (not a directory)
    assert CSV_FILE.is_file()


def test_csv_loads():
    # Test loading the CSV file and check the returned values
    df, status, rows, columns, _ = load_csv(str(CSV_FILE))
    # Ensure the DataFrame is not empty and has the expected shape
    assert isinstance(df, pd.DataFrame)
    # Ensure the DataFrame has the expected number of rows and columns
    assert df.shape[0] == 100  # Only the first 100 rows are returned
    assert df.shape[1] == 21  # The CSV has 21 columns
    # Ensure the status message indicates successful loading
    assert status == "CSV file loaded successfully!"
    # Ensure the total rows and columns are correct
    assert rows == "10194"
    # Ensure the total columns are correct
    assert columns == "21"


def test_csv_preview():
    # Test that the preview DataFrame contains the expected columns
    df, _, _, _, _ = load_csv(str(CSV_FILE))
    # Ensure the preview DataFrame has the expected number of rows
    assert len(df) == 100
    # Ensure the preview DataFrame contains the "Sales" column
    assert "Sales" in df.columns
    # Ensure the preview DataFrame contains the "Quantity" column
    assert "Quantity" in df.columns
    # Ensure the preview DataFrame contains the "Discount" column
    assert "Discount" in df.columns
    # Ensure the preview DataFrame contains the "Profit" column
    assert "Profit" in df.columns
    # Ensure the preview DataFrame contains the "Category" column
    assert "Category" in df.columns


def test_no_file_uploaded():
    # Test the behavior when no file is uploaded
    df, status, rows, columns, _ = load_csv(None)
    # Ensure the DataFrame is empty
    assert df.empty
    # Ensure the status message indicates that no file was uploaded
    assert status == "Please upload a CSV file first."
    # Ensure the total rows and columns are empty strings
    assert rows == ""
    # Ensure the total columns are empty strings
    assert columns == ""
