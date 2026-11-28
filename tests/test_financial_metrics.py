import pandas as pd
import pytest

from agent.financial_metrics import (
    try_financial_analysis,
)


# Create a representative dataset fixture for the analysis tests.
@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "Category": [
                "Alpha",
                "Alpha",
                "Beta",
                "Gamma",
            ],
            "Sub-Category": [
                "Product A",
                "Product A",
                "Product B",
                "Product C",
            ],
            "Region": [
                "North",
                "North",
                "South",
                "East",
            ],
            "Sales": [600, 400, 300, 200],
            "Profit": [-60, -40, 60, 20],
        }
    )


# Verify that above average sales negative profit.
def test_above_average_sales_negative_profit(dataset):
    result = try_financial_analysis(
        dataset,
        "Which product subcategories have "
        "above-average sales but negative total profit?",
    )

    assert result["error"] is None
    assert "Product A" in result["answer"]
    assert len(result["table"]) == 1

    row = result["table"][0]

    assert row["Total_Sales"] == 1000
    assert row["Total_Profit"] == -100
    assert row["Profit_Margin_Pct"] == -10
    assert row["Average_Sales_Threshold"] == 500


# Verify that profit margin below threshold.
def test_profit_margin_below_threshold(dataset):
    result = try_financial_analysis(
        dataset,
        "Which product categories have " "a profit margin below 5%?",
    )

    assert result["table"] == [
        {
            "Category": "Alpha",
            "Total_Sales": 1000.0,
            "Total_Profit": -100.0,
            "Profit_Margin_Pct": -10.0,
        }
    ]


# Verify that sales above threshold.
def test_sales_above_threshold(dataset):
    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $500?",
    )

    assert result["table"] == [
        {
            "Region": "North",
            "Total_Sales": 1000.0,
        }
    ]


# Verify that currency threshold with commas.
def test_currency_threshold_with_commas(dataset):
    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $1,000?",
    )

    assert result["error"] is None
    assert result["table"] is None


# Verify that profit below negative threshold.
def test_profit_below_negative_threshold(dataset):
    result = try_financial_analysis(
        dataset,
        "Which categories have total profit below -$50?",
    )

    assert result["table"][0]["Category"] == "Alpha"
    assert result["table"][0]["Total_Profit"] == -100


# Verify that margin threshold is strict.
def test_margin_threshold_is_strict(dataset):
    result = try_financial_analysis(
        dataset,
        "Which categories have profit margins below 10%?",
    )

    names = [row["Category"] for row in result["table"]]

    assert names == ["Alpha"]


# Verify that zero sales margin is unavailable.
def test_zero_sales_margin_is_unavailable(dataset):
    dataset.loc[
        dataset["Category"] == "Gamma",
        "Sales",
    ] = 0

    result = try_financial_analysis(
        dataset,
        "Which categories have profit margins below 5%?",
    )

    assert "unavailable" in result["answer"]

    assert all(row["Category"] != "Gamma" for row in result["table"])


# Verify that no matching groups.
def test_no_matching_groups(dataset):
    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $500,000?",
    )

    assert result["error"] is None
    assert result["table"] is None
    assert "No regions" in result["answer"]


# Verify that no above average loss matches.
def test_no_above_average_loss_matches(dataset):
    dataset["Profit"] = 100

    result = try_financial_analysis(
        dataset,
        "Which product subcategories have "
        "above-average sales but negative total profit?",
    )

    assert result["table"] is None
    assert "No groups" in result["answer"]


# Verify that missing required column uses fallback.
def test_missing_required_column_uses_fallback(dataset):
    dataset = dataset.drop(columns=["Profit"])

    result = try_financial_analysis(
        dataset,
        "Which categories have profit margins below 5%?",
    )

    assert result is None


# Verify that invalid numeric data uses fallback.
def test_invalid_numeric_data_uses_fallback(dataset):
    dataset["Sales"] = [
        "invalid",
        "400",
        "300",
        "200",
    ]

    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $500?",
    )

    assert result is None


# Verify that missing group values use fallback.
def test_missing_group_values_use_fallback(dataset):
    dataset.loc[0, "Region"] = None

    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $500?",
    )

    assert result is None


# Verify that unsupported question uses ai.
def test_unsupported_question_uses_ai(dataset):
    result = try_financial_analysis(
        dataset,
        "Why are customers purchasing fewer products?",
    )

    assert result is None


# Verify that dataset not modified.
def test_dataset_not_modified(dataset):
    original = dataset.copy(deep=True)

    try_financial_analysis(
        dataset,
        "Which categories have profit margins below 5%?",
    )

    pd.testing.assert_frame_equal(
        dataset,
        original,
    )


# Verify that filtering uses unrounded values.
def test_filtering_uses_unrounded_values():
    dataset = pd.DataFrame(
        {
            "Region": ["A", "B"],
            "Sales": [500.004, 500.0],
        }
    )

    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $500?",
    )

    assert len(result["table"]) == 1
    assert result["table"][0]["Region"] == "A"


# Verify that sales filter does not require profit.
def test_sales_filter_does_not_require_profit(dataset):
    dataset = dataset.drop(columns=["Profit"])

    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $500?",
    )

    assert result["table"][0]["Region"] == "North"


# Verify that custom group values not hardcoded.
def test_custom_group_values_not_hardcoded():
    dataset = pd.DataFrame(
        {
            "Segment": ["Retail", "Wholesale"],
            "Sales": [100, 200],
            "Profit": [-20, 50],
        }
    )

    result = try_financial_analysis(
        dataset,
        "Which segments have total profit below $0?",
    )

    assert result["table"][0]["Segment"] == "Retail"


# Verify that result display limit.
def test_result_display_limit():
    dataset = pd.DataFrame(
        {
            "Region": [f"Region {number}" for number in range(205)],
            "Sales": [100] * 205,
        }
    )

    result = try_financial_analysis(
        dataset,
        "Which regions have total sales above $0?",
    )

    assert len(result["table"]) == 200
    assert "205" in result["answer"]
    assert "Showing the first 200" in result["answer"]


# Create a dataset fixture containing discount values.
@pytest.fixture
def discount_dataset():
    return pd.DataFrame(
        {
            "Category": ["Alpha", "Alpha", "Beta", "Beta"],
            "Sales": [100, 200, 300, 400],
            "Profit": [30, -10, 20, 60],
            "Discount": [0.1, 0.3, 0.05, 0.15],
        }
    )


# Verify that discount relationship.
def test_discount_relationship(discount_dataset):
    result = try_financial_analysis(
        discount_dataset,
        "Analyze the relationship between discount "
        "and profitability across product categories.",
    )

    assert result["error"] is None

    rows = {row["Category"]: row for row in result["table"]}

    assert rows["Alpha"]["Total_Sales"] == 300
    assert rows["Alpha"]["Total_Profit"] == 20
    assert rows["Alpha"]["Profit_Margin_Pct"] == 6.67
    assert rows["Alpha"]["Average_Discount_Pct"] == 20
    assert rows["Alpha"]["Discount_Profit_Correlation"] == -1

    assert rows["Beta"]["Average_Discount_Pct"] == 10
    assert rows["Beta"]["Discount_Profit_Correlation"] == 1

    assert "Correlation does not establish causation" in (result["answer"])


# Verify that average discount without profit.
def test_average_discount_without_profit(discount_dataset):
    dataset = discount_dataset.drop(columns=["Sales", "Profit"])

    result = try_financial_analysis(
        dataset,
        "Show average discount by category.",
    )

    assert result["table"] == [
        {
            "Category": "Alpha",
            "Average_Discount_Pct": 20.0,
        },
        {
            "Category": "Beta",
            "Average_Discount_Pct": 10.0,
        },
    ]


# Verify that percentage discount values.
def test_percentage_discount_values(discount_dataset):
    discount_dataset["Discount"] = [10, 30, 5, 15]

    result = try_financial_analysis(
        discount_dataset,
        "Show average discount by category.",
    )

    assert result["table"][0]["Average_Discount_Pct"] == 20


# Verify that explicit percentage discount not multiplied.
def test_explicit_percentage_discount_not_multiplied(
    discount_dataset,
):
    dataset = discount_dataset.rename(
        columns={
            "Discount": "Discount_Pct",
        }
    )

    result = try_financial_analysis(
        dataset,
        "Show average discount by category.",
    )

    assert result["table"][0]["Average_Discount_Pct"] == 0.2


# Verify that fractional discount assumption disclosed.
def test_fractional_discount_assumption_disclosed(
    discount_dataset,
):
    result = try_financial_analysis(
        discount_dataset,
        "Show average discount by category.",
    )

    assert "fractional discounts" in result["answer"]


# Verify that constant discount correlation unavailable.
def test_constant_discount_correlation_unavailable(
    discount_dataset,
):
    discount_dataset["Discount"] = 0.2

    result = try_financial_analysis(
        discount_dataset,
        "Analyze the relationship between discount "
        "and profitability across categories.",
    )

    assert all(row["Discount_Profit_Correlation"] is None for row in result["table"])

    assert "unavailable" in result["answer"].lower()


# Verify that constant profit correlation unavailable.
def test_constant_profit_correlation_unavailable(
    discount_dataset,
):
    discount_dataset["Profit"] = 10

    result = try_financial_analysis(
        discount_dataset,
        "Analyze the relationship between discount "
        "and profitability across categories.",
    )

    assert all(row["Discount_Profit_Correlation"] is None for row in result["table"])


# Verify that missing discount uses valid observations.
def test_missing_discount_uses_valid_observations(
    discount_dataset,
):
    discount_dataset.loc[0, "Discount"] = float("nan")

    result = try_financial_analysis(
        discount_dataset,
        "Analyze the relationship between discount "
        "and profitability across categories.",
    )

    alpha = result["table"][0]

    assert alpha["Average_Discount_Pct"] == 30
    assert alpha["Discount_Profit_Correlation"] is None
    assert alpha["Total_Sales"] == 300


# Verify that invalid discount uses fallback.
@pytest.mark.parametrize(
    "discounts",
    [
        [-0.1, 0.2, 0.3, 0.4],
        [101, 20, 30, 40],
        [0.1, 20, 0.3, 40],
        ["invalid", 0.2, 0.3, 0.4],
        [float("inf"), 0.2, 0.3, 0.4],
    ],
)
def test_invalid_discount_uses_fallback(
    discount_dataset,
    discounts,
):
    discount_dataset["Discount"] = discounts

    result = try_financial_analysis(
        discount_dataset,
        "Show average discount by category.",
    )

    assert result is None


# Verify that discount analysis preserves dataset.
def test_discount_analysis_preserves_dataset(
    discount_dataset,
):
    original = discount_dataset.copy(deep=True)

    try_financial_analysis(
        discount_dataset,
        "Analyze the relationship between discount "
        "and profitability across categories.",
    )

    pd.testing.assert_frame_equal(
        discount_dataset,
        original,
    )


# Verify that discount analysis supports custom segments.
def test_discount_analysis_supports_custom_segments(
    discount_dataset,
):
    dataset = discount_dataset.rename(
        columns={
            "Category": "Segment",
        }
    )

    result = try_financial_analysis(
        dataset,
        "Show average discount by segment.",
    )

    assert result["table"][0]["Segment"] == "Alpha"
    assert result["table"][0]["Average_Discount_Pct"] == 20


# Verify that discount recommendation uses fallback.
def test_discount_recommendation_uses_fallback(
    discount_dataset,
):
    result = try_financial_analysis(
        discount_dataset,
        "Recommend discount strategies by category.",
    )

    assert result is None


# Verify that discount extrema preserve ties.
def test_discount_extrema_preserve_ties(
    discount_dataset,
):
    discount_dataset["Discount"] = 0.2

    result = try_financial_analysis(
        discount_dataset,
        "Show average discount by category.",
    )

    assert "**Alpha**, **Beta**" in result["answer"]


# Verify that discount correlation precision.
def test_discount_correlation_precision():
    dataset = pd.DataFrame(
        {
            "Category": ["Alpha"] * 4,
            "Sales": [100, 200, 300, 400],
            "Profit": [10, 30, 20, 15],
            "Discount": [0.1, 0.2, 0.3, 0.4],
        }
    )

    result = try_financial_analysis(
        dataset,
        "Analyze the relationship between discount "
        "and profitability across categories.",
    )

    expected = round(
        dataset["Discount"].corr(dataset["Profit"]),
        4,
    )

    assert result["table"][0]["Discount_Profit_Correlation"] == expected


# Verify that discount zero sales margin unavailable.
def test_discount_zero_sales_margin_unavailable(
    discount_dataset,
):
    discount_dataset["Sales"] = 0

    result = try_financial_analysis(
        discount_dataset,
        "Analyze the relationship between discount "
        "and profitability across categories.",
    )

    assert all(row["Profit_Margin_Pct"] is None for row in result["table"])


# Verify that ambiguous discount columns use fallback.
def test_ambiguous_discount_columns_use_fallback(
    discount_dataset,
):
    discount_dataset["Discount_Pct"] = 20

    result = try_financial_analysis(
        discount_dataset,
        "Show average discount by category.",
    )

    assert result is None
