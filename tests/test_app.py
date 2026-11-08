
"""Tests for the Business Data Analysis Agent's Gradio interface."""

from pathlib import Path

import pandas as pd
import pytest

from app.app import (
    load_csv,
    load_csv_and_reset_chat,
    analyze_question,
    send_message,
)

# Locate the sample CSV relative to the project root.
# This works regardless of where pytest is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT_ROOT / "data" / "superstore_orders.csv"


# ---------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------

def test_csv_exists():
    """Verify that the sample Superstore dataset exists."""
    assert CSV_FILE.exists(), f"CSV file not found at {CSV_FILE}"
    assert CSV_FILE.is_file()


def test_csv_loads():
    """Verify successful CSV loading and dataset statistics."""
    preview, status, rows, columns, full_df = load_csv(str(CSV_FILE))

    # The UI displays only the first 100 rows.
    assert isinstance(preview, pd.DataFrame)
    assert preview.shape == (100, 21)

    # The complete dataset remains available in session state.
    assert isinstance(full_df, pd.DataFrame)
    assert full_df.shape == (10194, 21)

    # Check the status and dataset metadata displayed in the UI.
    assert status == "CSV file loaded successfully!"
    assert rows == "10194"
    assert columns == "21"


def test_csv_preview():
    """Verify that the dataset preview contains business metrics."""
    preview, _, _, _, _ = load_csv(str(CSV_FILE))

    assert len(preview) == 100

    expected_columns = {
        "Sales",
        "Quantity",
        "Discount",
        "Profit",
        "Category",
    }

    assert expected_columns.issubset(set(preview.columns))


def test_no_file_uploaded():
    """Verify the UI handles a missing CSV upload gracefully."""
    preview, status, rows, columns, full_df = load_csv(None)

    assert preview.empty
    assert full_df.empty
    assert status == "Please upload a CSV file first."
    assert rows == ""
    assert columns == ""


def test_empty_csv(tmp_path):
    """Verify that an empty CSV is rejected gracefully."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("Sales,Profit\n", encoding="utf-8")

    preview, status, rows, columns, full_df = load_csv(str(csv_path))

    assert preview.empty
    assert full_df.empty
    assert "empty" in status.lower()
    assert rows == ""
    assert columns == ""


def test_invalid_csv_path():
    """Verify that a nonexistent file produces an error message."""
    preview, status, rows, columns, full_df = load_csv(
        str(PROJECT_ROOT / "data" / "missing_file.csv")
    )

    assert preview.empty
    assert full_df.empty
    assert "Error loading CSV:" in status
    assert rows == ""
    assert columns == ""


# ---------------------------------------------------------
# Business analysis interface tests
# ---------------------------------------------------------

def test_question_without_dataset():
    """Verify that users must load a dataset before analysis."""
    response = analyze_question(
        "What is the total sales?",
        pd.DataFrame(),
    )

    assert response == "Please upload a CSV file first."


def test_empty_question():
    """Verify that empty questions are handled gracefully."""
    df = pd.DataFrame({"Sales": [100, 200]})

    response = analyze_question("", df)

    assert response == "Please enter a question about your dataset."


def test_analysis_placeholder():
    """Verify the current placeholder backend receives a question."""
    df = pd.DataFrame({"Sales": [100, 200]})

    response = analyze_question("What is total sales?", df)

    assert "What is total sales?" in response
    assert "AI analysis will be available in Stage 2." in response

    # This test should be updated when the agent backend is connected.


# ---------------------------------------------------------
# Chat history tests
# ---------------------------------------------------------

def test_send_message():
    """Verify that sending a question creates a chat exchange."""
    df = pd.DataFrame({"Sales": [100, 200]})

    history, cleared_input = send_message(
        "What is total sales?",
        [],
        df,
    )

    # The Gradio chatbot uses role/content message dictionaries.
    assert len(history) == 2
    assert history[0] == {
        "role": "user",
        "content": "What is total sales?",
    }
    assert history[1]["role"] == "assistant"
    assert "Question received:" in history[1]["content"]

    # The question input clears after sending.
    assert cleared_input == ""


def test_chat_history_preserved():
    """Verify that follow-up questions preserve previous messages."""
    df = pd.DataFrame({"Sales": [100, 200]})

    history, _ = send_message("What is total sales?", [], df)

    updated_history, _ = send_message(
        "What about profit?",
        history,
        df,
    )

    # Two questions and two assistant responses.
    assert len(updated_history) == 4

    assert updated_history[0]["content"] == "What is total sales?"
    assert updated_history[2]["content"] == "What about profit?"

    assert updated_history[1]["role"] == "assistant"
    assert updated_history[3]["role"] == "assistant"


def test_empty_chat_message():
    """Verify that blank messages do not change chat history."""
    df = pd.DataFrame({"Sales": [100]})

    history, cleared_input = send_message("   ", [], df)

    assert history == []
    assert cleared_input == ""


def test_chat_without_dataset():
    """Verify that the chatbot requests a dataset when none is loaded."""
    history, _ = send_message(
        "Which category has the highest profit?",
        [],
        None,
    )

    assert len(history) == 2
    assert history[1]["role"] == "assistant"
    assert "Please upload a CSV file first." in history[1]["content"]


# ---------------------------------------------------------
# Dataset and conversation reset tests
# ---------------------------------------------------------

def test_load_dataset_resets_chat():
    """Verify that loading a dataset starts a fresh conversation."""
    preview, status, rows, columns, df, history = (
        load_csv_and_reset_chat(str(CSV_FILE))
    )

    assert status == "CSV file loaded successfully!"
    assert preview.shape == (100, 21)
    assert df.shape == (10194, 21)
    assert rows == "10194"
    assert columns == "21"

    # Loading a dataset resets the chat history.
    assert history == []


def test_missing_dataset_resets_chat():
    """Verify that a failed upload also returns an empty chat history."""
    preview, status, rows, columns, df, history = (
        load_csv_and_reset_chat(None)
    )

    assert preview.empty
    assert df.empty
    assert history == []
    assert status == "Please upload a CSV file first."
