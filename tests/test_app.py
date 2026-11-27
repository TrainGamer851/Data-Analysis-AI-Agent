
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
    preview, status, rows, columns, csv_path = load_csv(str(CSV_FILE))

    assert isinstance(preview, pd.DataFrame)
    assert preview.shape == (100, 21)

    assert isinstance(csv_path, str)
    assert Path(csv_path) == CSV_FILE

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
    assert full_df is None
    assert status == "Please upload a CSV file first."
    assert rows == ""
    assert columns == ""


def test_empty_csv(tmp_path):
    """Verify that an empty CSV is rejected gracefully."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("Sales,Profit\n", encoding="utf-8")

    preview, status, rows, columns, full_df = load_csv(str(csv_path))

    assert preview.empty
    assert full_df is None
    assert "empty" in status.lower()
    assert rows == ""
    assert columns == ""


def test_invalid_csv_path():
    """Verify that a nonexistent file produces an error message."""
    preview, status, rows, columns, full_df = load_csv(
        str(PROJECT_ROOT / "data" / "missing_file.csv")
    )

    assert preview.empty
    assert full_df is None
    assert "Error loading CSV:" in status
    assert rows == ""
    assert columns == ""


# ---------------------------------------------------------
# Business analysis interface tests
# ---------------------------------------------------------

def test_question_without_dataset():
    response = analyze_question(
        "What is the total sales?",
        None,
    )

    assert response == "Please upload a CSV file first."

def test_empty_question():
    response = analyze_question("", str(CSV_FILE))

    assert response == "Please enter a question about your dataset."


def test_analysis_backend(monkeypatch):
    def fake_analysis(
        csv_path,
        question,
        explain=True,
        context=None,
    ):
        assert csv_path == str(CSV_FILE)
        assert question == "What is total sales?"
        assert explain is True

        return {
            "answer": "Total sales are $300.00.",
            "table": None,
            "error": None,
        }

    monkeypatch.setattr(
        "app.app.run_analysis",
        fake_analysis,
    )

    response = analyze_question(
        "What is total sales?",
        str(CSV_FILE),
    )

    assert response == "Total sales are $300.00."


# ---------------------------------------------------------
# Chat history tests
# ---------------------------------------------------------

def test_send_message(monkeypatch):
    monkeypatch.setattr(
        "app.app.run_analysis",
        lambda *args, **kwargs: {
            "answer": "Total sales are $300.00.",
            "table": None,
            "error": None,
        },
    )

    history, cleared_input = send_message(
        "What is total sales?",
        [],
        str(CSV_FILE),
    )

    assert len(history) == 2

    assert history[0] == {
        "role": "user",
        "content": "What is total sales?",
    }

    assert history[1] == {
        "role": "assistant",
        "content": "Total sales are $300.00.",
    }

    assert cleared_input == ""


def test_chat_history_preserved(monkeypatch):
    answers = iter([
        "Total sales are $300.00.",
        "Total profit is $50.00.",
    ])

    monkeypatch.setattr(
        "app.app.run_analysis",
        lambda *args, **kwargs: {
            "answer": next(answers),
            "table": None,
            "error": None,
        },
    )

    history, _ = send_message(
        "What is total sales?",
        [],
        str(CSV_FILE),
    )

    updated_history, _ = send_message(
        "What about profit?",
        history,
        str(CSV_FILE),
    )

    assert len(updated_history) == 4

    assert updated_history[0]["content"] == "What is total sales?"
    assert updated_history[1]["content"] == "Total sales are $300.00."
    assert updated_history[2]["content"] == "What about profit?"
    assert updated_history[3]["content"] == "Total profit is $50.00."

    assert updated_history[1]["role"] == "assistant"
    assert updated_history[3]["role"] == "assistant"


def test_empty_chat_message():
    history, cleared_input = send_message(
        "   ",
        [],
        str(CSV_FILE),
    )

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
    preview, status, rows, columns, csv_path, history = (
        load_csv_and_reset_chat(str(CSV_FILE))
    )

    assert status == "CSV file loaded successfully!"
    assert preview.shape == (100, 21)
    assert Path(csv_path) == CSV_FILE
    assert rows == "10194"
    assert columns == "21"
    assert history == []


def test_missing_dataset_resets_chat():
    preview, status, rows, columns, csv_path, history = (
        load_csv_and_reset_chat(None)
    )

    assert preview.empty
    assert csv_path is None
    assert history == []
    assert status == "Please upload a CSV file first."

# ------------------------------------------------------------
# Frontend cancellation tests
# ------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import patch

import app.app as frontend


def test_controlled_message_success():
    request = SimpleNamespace(
        session_hash="test-success"
    )

    with patch.object(
        frontend,
        "analyze_question_with_chart",
        return_value=("Total sales: 300.00", None),
    ):
        updates = list(
            frontend.send_controlled_message(
                "What are total sales?",
                [],
                "orders.csv",
                request,
            )
        )

    history = updates[-1][0]

    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Total sales: 300.00"
    assert request.session_hash not in frontend._active_jobs


def test_controlled_message_cancellation():
    request = SimpleNamespace(
        session_hash="test-cancellation"
    )

    def fake_analysis(
        question,
        csv_path,
        job=None,
        context=None,
    ):
        frontend.cancel_active_analysis(request)

        assert job.cancelled is True

        return "This result should not appear.", None

    with patch.object(
        frontend,
        "analyze_question_with_chart",
        side_effect=fake_analysis,
    ):
        updates = list(
            frontend.send_controlled_message(
                "Analyze business performance",
                [],
                "orders.csv",
                request,
            )
        )

    history = updates[-1][0]

    assert history[-1]["content"] == "Analysis stopped."
    assert request.session_hash not in frontend._active_jobs


def test_cancellation_is_session_specific():
    first_request = SimpleNamespace(
        session_hash="test-session-one"
    )

    second_request = SimpleNamespace(
        session_hash="test-session-two"
    )

    first_job = frontend.AnalysisJob()
    second_job = frontend.AnalysisJob()

    with frontend._jobs_lock:
        frontend._active_jobs[first_request.session_hash] = first_job
        frontend._active_jobs[second_request.session_hash] = second_job

    try:
        frontend.cancel_active_analysis(first_request)

        assert first_job.cancelled is True
        assert second_job.cancelled is False

    finally:
        with frontend._jobs_lock:
            frontend._active_jobs.pop(
                first_request.session_hash, None
            )
            frontend._active_jobs.pop(
                second_request.session_hash, None
            )


def test_cancel_without_active_job():
    request = SimpleNamespace(
        session_hash="test-no-active-job"
    )

    frontend.cancel_active_analysis(request)

    assert request.session_hash not in frontend._active_jobs


def test_controlled_message_rejects_duplicate_job():
    request = SimpleNamespace(
        session_hash="test-duplicate-job"
    )

    existing_job = frontend.AnalysisJob()

    with frontend._jobs_lock:
        frontend._active_jobs[request.session_hash] = existing_job

    try:
        with patch.object(
            frontend,
            "analyze_question",
        ) as mock_analysis:
            updates = list(
                frontend.send_controlled_message(
                    "Another question",
                    [],
                    "orders.csv",
                    request,
                )
            )

        assert updates == []
        mock_analysis.assert_not_called()

        assert (
            frontend._active_jobs[request.session_hash]
            is existing_job
        )

    finally:
        with frontend._jobs_lock:
            frontend._active_jobs.pop(
                request.session_hash, None
            )

def test_analysis_with_chart():
    response = {
        "answer": "Profit by category.",
        "table": [
            {"Category": "Technology", "Profit": 1000},
            {"Category": "Furniture", "Profit": 200},
        ],
        "error": None,
    }

    with patch.object(
        frontend,
        "run_analysis",
        return_value=response,
    ):
        answer, figure = frontend.analyze_question_with_chart(
            "Show profit by category.",
            "orders.csv",
        )

    assert "Profit by category." in answer
    assert "Supporting Data" not in answer
    assert "Technology" in answer
    assert "Furniture" in answer
    assert "$1,000.00" in answer
    assert "$200.00" in answer
    assert figure is not None
    assert figure.data[0].type == "bar"


def test_analysis_without_chart():
    response = {
        "answer": "Total sales: $1,000.00",
        "table": None,
        "error": None,
    }

    with patch.object(
        frontend,
        "run_analysis",
        return_value=response,
    ):
        answer, figure = frontend.analyze_question_with_chart(
            "What are the total sales?",
            "orders.csv",
        )

    assert answer == "Total sales: $1,000.00"
    assert figure is None


def test_chart_failure_preserves_answer():
    response = {
        "answer": "Profit by category.",
        "table": [
            {"Category": "Technology", "Profit": 1000},
            {"Category": "Furniture", "Profit": 200},
        ],
        "error": None,
    }

    with patch.object(
        frontend,
        "run_analysis",
        return_value=response,
    ):
        with patch.object(
            frontend,
            "build_chart",
            side_effect=RuntimeError("Chart failed"),
        ):
            answer, figure = (
                frontend.analyze_question_with_chart(
                    "Show profit by category.",
                    "orders.csv",
                )
            )

    assert "Profit by category." in answer
    assert "Supporting Data" not in answer
    assert "Technology" in answer
    assert "Furniture" in answer
    assert "$1,000.00" in answer
    assert "$200.00" in answer
    assert figure is None

def test_followup_receives_previous_context():
    request = SimpleNamespace(
        session_hash="test-followup"
    )

    history = [
        {
            "role": "user",
            "content": "Which region has the highest profit?",
        },
        {
            "role": "assistant",
            "content": "West has the highest profit.",
        },
    ]

    with patch.object(
        frontend,
        "analyze_question_with_chart",
        return_value=("West exceeds East.", None),
    ) as mock_analysis:

        updates = list(
            frontend.send_controlled_message(
                "How much higher is it than East?",
                history,
                "orders.csv",
                request,
            )
        )

    context = mock_analysis.call_args.kwargs["context"]

    assert len(context) == 2
    assert context[1]["content"] == (
        "West has the highest profit."
    )

    assert updates[-1][0][-1]["content"] == (
        "West exceeds East."
    )


def test_first_question_has_no_context():
    request = SimpleNamespace(
        session_hash="test-no-context"
    )

    with patch.object(
        frontend,
        "analyze_question_with_chart",
        return_value=("Total sales: $300.", None),
    ) as mock_analysis:

        list(
            frontend.send_controlled_message(
                "What are total sales?",
                [],
                "orders.csv",
                request,
            )
        )

    assert mock_analysis.call_args.kwargs["context"] == []


def test_cancelled_turn_not_sent_as_context():
    request = SimpleNamespace(
        session_hash="test-cancelled-context"
    )

    history = [
        {
            "role": "user",
            "content": "Analyze everything.",
        },
        {
            "role": "assistant",
            "content": "Analysis stopped.",
        },
    ]

    with patch.object(
        frontend,
        "analyze_question_with_chart",
        return_value=("Total profit: $100.", None),
    ) as mock_analysis:

        list(
            frontend.send_controlled_message(
                "What is total profit?",
                history,
                "orders.csv",
                request,
            )
        )

    assert mock_analysis.call_args.kwargs["context"] == []