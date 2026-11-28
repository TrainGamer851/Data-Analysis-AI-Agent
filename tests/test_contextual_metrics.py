import pandas as pd
import pytest

from agent.contextual_metrics import (
    try_contextual_analysis,
)


# Create a representative dataset fixture for the analysis tests.
@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "Region": ["West", "East", "West", "East"],
            "Category": [
                "Technology",
                "Furniture",
                "Technology",
                "Furniture",
            ],
            "Order Date": [
                "2023-01-15",
                "2023-06-15",
                "2024-01-15",
                "2024-06-15",
            ],
            "Sales": [100, 200, 300, 400],
            "Profit": [20, 10, 80, 40],
        }
    )


# Create a conversation turn fixture with the requested content.
def turn(question, answer="Completed analysis."):
    return [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]


# Verify that highest profit region.
def test_highest_profit_region(dataset):
    result = try_contextual_analysis(
        dataset,
        "Which region has the highest total profit?",
    )

    assert result["error"] is None
    assert "West" in result["answer"]
    assert "$100.00" in result["answer"]


# Verify that lowest profit region.
def test_lowest_profit_region(dataset):
    result = try_contextual_analysis(
        dataset,
        "Which region has the lowest total profit?",
    )

    assert "East" in result["answer"]
    assert "$50.00" in result["answer"]


# Verify that absolute difference.
def test_absolute_difference(dataset):
    context = turn("Which region has the highest total profit?")

    result = try_contextual_analysis(
        dataset,
        "How much higher is it than East?",
        context,
    )

    assert "$50.00" in result["answer"]
    assert "West" in result["answer"]
    assert "East" in result["answer"]


# Verify that percentage difference.
def test_percentage_difference(dataset):
    context = turn("Which region has the highest total profit?")

    context += turn("How much higher is it than East?")

    result = try_contextual_analysis(
        dataset,
        "What is the percentage difference relative to East?",
        context,
    )

    assert "100.00%" in result["answer"]


# Verify that percentage reuses baseline.
def test_percentage_reuses_baseline(dataset):
    context = turn("Which region has the highest total profit?")

    context += turn("How much higher is it than East?")

    result = try_contextual_analysis(
        dataset,
        "What is the percentage difference?",
        context,
    )

    assert "100.00%" in result["answer"]


# Verify that grouped profit table.
def test_grouped_profit_table(dataset):
    result = try_contextual_analysis(
        dataset,
        "Show total profit by category in a table.",
    )

    assert result["table"] == [
        {
            "Category": "Technology",
            "Total Profit": 100.0,
        },
        {
            "Category": "Furniture",
            "Total Profit": 50.0,
        },
    ]


# Verify that metric switch.
def test_metric_switch(dataset):
    context = turn("Show total profit by category in a table.")

    result = try_contextual_analysis(
        dataset,
        "Now show sales instead.",
        context,
    )

    assert result["table"] == [
        {
            "Category": "Furniture",
            "Total Sales": 600.0,
        },
        {
            "Category": "Technology",
            "Total Sales": 400.0,
        },
    ]


# Verify that contextual highest sales.
def test_contextual_highest_sales(dataset):
    context = turn("Show total profit by category in a table.")

    context += turn("Now show sales instead.")

    result = try_contextual_analysis(
        dataset,
        "Which one has the highest sales?",
        context,
    )

    assert "Furniture" in result["answer"]
    assert "$600.00" in result["answer"]


# Verify that lowest annual sales.
def test_lowest_annual_sales(dataset):
    context = turn("Show annual sales and profit by year.")

    result = try_contextual_analysis(
        dataset,
        "Which of those years had the lowest sales?",
        context,
    )

    assert "2023" in result["answer"]
    assert "$300.00" in result["answer"]


# Verify that profit in referenced year.
def test_profit_in_referenced_year(dataset):
    context = turn("Show annual sales and profit by year.")

    context += turn("Which of those years had the lowest sales?")

    result = try_contextual_analysis(
        dataset,
        "What was the total profit in that year?",
        context,
    )

    assert "2023" in result["answer"]
    assert "$30.00" in result["answer"]


# Verify that reverse ranking.
def test_reverse_ranking(dataset):
    context = turn("Which region has the highest total profit?")

    result = try_contextual_analysis(
        dataset,
        "What about the lowest?",
        context,
    )

    assert "East" in result["answer"]
    assert "$50.00" in result["answer"]


# Verify that top three.
def test_top_three(dataset):
    context = turn("Show total profit by category in a table.")

    result = try_contextual_analysis(
        dataset,
        "Show the top three instead.",
        context,
    )

    assert len(result["table"]) == 2

    assert result["table"][0]["Category"] == ("Technology")


# Verify that zero percentage baseline.
def test_zero_percentage_baseline(dataset):
    dataset.loc[
        dataset["Region"] == "East",
        "Profit",
    ] = 0

    context = turn("Which region has the highest total profit?")

    result = try_contextual_analysis(
        dataset,
        "What is the percentage difference relative to East?",
        context,
    )

    assert "nonpositive" in result["answer"]


# Verify that missing context.
def test_missing_context(dataset):
    result = try_contextual_analysis(
        dataset,
        "Now show sales instead.",
    )

    assert result is None


# Verify that unsupported question uses ai.
def test_unsupported_question_uses_ai(dataset):
    result = try_contextual_analysis(
        dataset,
        "Why are customers purchasing fewer products?",
    )

    assert result is None


# Verify that missing metric uses ai.
def test_missing_metric_uses_ai(dataset):
    dataset = dataset.drop(columns=["Profit"])

    result = try_contextual_analysis(
        dataset,
        "Which region has the highest total profit?",
    )

    assert result is None


# Verify that invalid numeric values use ai.
def test_invalid_numeric_values_use_ai(dataset):
    dataset["Profit"] = ["invalid", "10", "80", "40"]

    result = try_contextual_analysis(
        dataset,
        "Which region has the highest total profit?",
    )

    assert result is None


# Verify that invalid dates use ai.
def test_invalid_dates_use_ai(dataset):
    dataset.loc[0, "Order Date"] = "invalid"

    context = turn("Show annual sales and profit by year.")

    result = try_contextual_analysis(
        dataset,
        "Which of those years had the lowest sales?",
        context,
    )

    assert result is None


# Verify that tied ranking.
def test_tied_ranking(dataset):
    dataset["Profit"] = [25, 25, 25, 25]

    result = try_contextual_analysis(
        dataset,
        "Which region has the highest total profit?",
    )

    assert "West" in result["answer"]
    assert "East" in result["answer"]


# Verify that tied reference requests clarification.
def test_tied_reference_requests_clarification(dataset):
    dataset["Profit"] = [25, 25, 25, 25]

    context = turn("Which region has the highest total profit?")

    result = try_contextual_analysis(
        dataset,
        "How much higher is it than East?",
        context,
    )

    assert "Clarification needed" in result["answer"]


# Verify that failed turn excluded.
def test_failed_turn_excluded(dataset):
    context = turn("Which region has the highest total profit?")

    context += turn(
        "Show total sales by category.",
        "**Analysis Error:** Analysis timed out.",
    )

    result = try_contextual_analysis(
        dataset,
        "How much higher is it than East?",
        context,
    )

    assert "$50.00" in result["answer"]


# Verify that previous answer not used as numeric evidence.
def test_previous_answer_not_used_as_numeric_evidence(dataset):
    context = turn(
        "Which region has the highest total profit?",
        "West generated $999999999.",
    )

    result = try_contextual_analysis(
        dataset,
        "How much higher is it than East?",
        context,
    )

    assert "$50.00" in result["answer"]
    assert "999999999" not in result["answer"]


# Verify that dataset not modified.
def test_dataset_not_modified(dataset):
    original = dataset.copy(deep=True)

    try_contextual_analysis(
        dataset,
        "Show total profit by category.",
    )

    pd.testing.assert_frame_equal(
        dataset,
        original,
    )


# Verify that unrelated question resets intent.
def test_unrelated_question_resets_intent(dataset):
    context = turn("Which region has the highest total profit?")

    context += turn("How many unique customers are there?")

    result = try_contextual_analysis(
        dataset,
        "How much higher is it than East?",
        context,
    )

    assert result is None
