import pandas as pd
import pytest

from agent.business_metrics import try_business_analysis


# Create the business data fixture used by the tests.
@pytest.fixture
def business_data():
    return pd.DataFrame(
        {
            "Sales": [100.0, 200.0, 300.0],
            "Profit": [-10.0, 40.0, 90.0],
            "Category": [
                "Furniture",
                "Office Supplies",
                "Technology",
            ],
            "Region": ["Central", "East", "West"],
            "Sub-Category": ["Tables", "Paper", "Phones"],
            "Customer ID": ["C1", "C2", "C1"],
            "Order ID": ["O1", "O2", "O3"],
        }
    )


# Verify that overall performance.
def test_overall_performance(business_data):
    result = try_business_analysis(
        business_data,
        "Analyze the overall business performance.",
    )

    assert result["error"] is None
    assert "$600.00" in result["answer"]
    assert "$120.00" in result["answer"]
    assert "20.00%" in result["answer"]
    assert "$200.00" in result["answer"]
    assert "Technology" in result["answer"]
    assert "West" in result["answer"]


# Verify that profitability recommendations.
def test_profitability_recommendations(business_data):
    result = try_business_analysis(
        business_data,
        "Based on the dataset, provide three "
        "actionable recommendations to improve profitability.",
    )

    assert result["error"] is None
    assert "Tables" in result["answer"]
    assert "-$10.00" in result["answer"]
    assert "1." in result["answer"]
    assert "2." in result["answer"]
    assert "3." in result["answer"]


# Verify that unsupported question uses fallback.
def test_unsupported_question_uses_fallback(business_data):
    result = try_business_analysis(
        business_data,
        "What are the total sales?",
    )

    assert result is None


# Verify that missing columns use fallback.
def test_missing_columns_use_fallback():
    data = pd.DataFrame({"Quantity": [1, 2, 3]})

    result = try_business_analysis(
        data,
        "Analyze the overall business performance.",
    )

    assert result is None


# Verify that invalid financial data uses fallback.
def test_invalid_financial_data_uses_fallback(business_data):
    business_data["Sales"] = ["invalid", "200", "300"]

    result = try_business_analysis(
        business_data,
        "Analyze the overall business performance.",
    )

    assert result is None


# Verify that zero sales does not crash.
def test_zero_sales_does_not_crash(business_data):
    business_data["Sales"] = 0

    result = try_business_analysis(
        business_data,
        "Analyze the overall business performance.",
    )

    assert result["error"] is None
    assert "$0.00" in result["answer"]


# Verify that filtered question uses fallback.
def test_filtered_question_uses_fallback(business_data):
    result = try_business_analysis(
        business_data,
        "Analyze the overall business performance for West.",
    )

    assert result is None


# Verify that input dataframe is not modified.
def test_input_dataframe_is_not_modified(business_data):
    original = business_data.copy(deep=True)

    try_business_analysis(
        business_data,
        "Analyze the overall business performance.",
    )

    pd.testing.assert_frame_equal(
        business_data,
        original,
    )
