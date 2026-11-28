import pandas as pd
import pytest
from agent.annual_metrics import try_annual_analysis


# Verify that two digit year parsing.
def test_two_digit_year_parsing():
    dates = pd.Series(
        [
            "1/3/23",
            "12/31/24",
            "6/15/25",
            "10/20/26",
        ]
    )

    parsed = pd.to_datetime(
        dates,
        format="%m/%d/%y",
        errors="coerce",
    )

    assert parsed.isna().sum() == 0

    assert parsed.dt.year.tolist() == [2023, 2024, 2025, 2026]


# Verify that mixed date formats.
def test_mixed_date_formats():
    dates = pd.Series(
        [
            "1/3/2023",
            "2024-06-15",
            "12/31/2025",
        ]
    )

    parsed = pd.to_datetime(
        dates,
        format="mixed",
        dayfirst=False,
        errors="coerce",
    )

    assert parsed.isna().sum() == 0

    assert parsed.dt.year.tolist() == [2023, 2024, 2025]


# Verify that invalid dates are detected.
def test_invalid_dates_are_detected():
    dates = pd.Series(
        [
            "1/3/23",
            "invalid",
            "2/30/24",
            None,
        ]
    )

    parsed = pd.to_datetime(
        dates,
        format="%m/%d/%y",
        errors="coerce",
    )

    assert parsed.isna().sum() == 3


# Verify that annual totals reconcile.
def test_annual_totals_reconcile():
    df = pd.DataFrame(
        {
            "Order Date": [
                "1/3/23",
                "12/31/23",
                "1/1/24",
                "12/31/24",
            ],
            "Sales": [100.0, 200.0, 300.0, 400.0],
            "Profit": [10.0, 20.0, 30.0, 40.0],
        }
    )

    dates = pd.to_datetime(
        df["Order Date"],
        format="%m/%d/%y",
        errors="coerce",
    )

    assert dates.notna().all()

    annual = df.groupby(dates.dt.year)[["Sales", "Profit"]].sum()

    assert annual.loc[2023, "Sales"] == 300
    assert annual.loc[2024, "Sales"] == 700

    assert annual["Sales"].sum() == pytest.approx(df["Sales"].sum())

    assert annual["Profit"].sum() == pytest.approx(df["Profit"].sum())


# Verify that annual growth calculation.
def test_annual_growth_calculation():
    sales = pd.Series(
        [100.0, 150.0, 120.0],
        index=[2023, 2024, 2025],
    )

    growth = sales.pct_change() * 100

    assert pd.isna(growth.loc[2023])
    assert growth.loc[2024] == pytest.approx(50.0)
    assert growth.loc[2025] == pytest.approx(-20.0)


from agent.annual_metrics import try_annual_analysis


# Create the annual data fixture used by the tests.
@pytest.fixture
def annual_data():
    return pd.DataFrame(
        {
            "Order Date": [
                "1/15/23",
                "12/31/23",
                "1/15/24",
                "12/31/24",
            ],
            "Sales": [100.0, 200.0, 300.0, 400.0],
            "Profit": [10.0, 20.0, 30.0, 40.0],
        }
    )


# Verify that fast annual analysis.
def test_fast_annual_analysis(annual_data):
    result = try_annual_analysis(
        annual_data,
        "Show annual sales and profit by year.",
    )

    assert result["error"] is None

    assert result["table"] == [
        {
            "Year": 2023,
            "Total Sales": 300.0,
            "Total Profit": 30.0,
        },
        {
            "Year": 2024,
            "Total Sales": 700.0,
            "Total Profit": 70.0,
        },
    ]


# Verify that fast annual growth.
def test_fast_annual_growth(annual_data):
    result = try_annual_analysis(
        annual_data,
        "Calculate year-over-year sales growth.",
    )

    assert result["table"][0]["Sales Growth %"] is None

    assert result["table"][1]["Sales Growth %"] == (pytest.approx(133.33))


# Verify that fast date validation.
def test_fast_date_validation(annual_data):
    result = try_annual_analysis(
        annual_data,
        "Are there any invalid or missing order dates?",
    )

    assert "No invalid or missing" in result["answer"]


# Verify that fast highest profit year.
def test_fast_highest_profit_year(annual_data):
    result = try_annual_analysis(
        annual_data,
        "Which year had the highest total profit?",
    )

    assert "2024" in result["answer"]
    assert "$70.00" in result["answer"]


# Verify that invalid dates prevent incomplete totals.
def test_invalid_dates_prevent_incomplete_totals(annual_data):
    annual_data.loc[0, "Order Date"] = "2/30/23"

    result = try_annual_analysis(
        annual_data,
        "Show annual sales and profit by year.",
    )

    assert result["table"] is None
    assert "1 records" in result["answer"]


# Verify that zero previous sales.
def test_zero_previous_sales(annual_data):
    annual_data.loc[:1, "Sales"] = 0

    result = try_annual_analysis(
        annual_data,
        "Calculate year-over-year sales growth.",
    )

    assert result["table"][1]["Sales Growth %"] is None


# Verify that missing previous year.
def test_missing_previous_year(annual_data):
    annual_data["Order Date"] = [
        "1/15/23",
        "12/31/23",
        "1/15/25",
        "12/31/25",
    ]

    result = try_annual_analysis(
        annual_data,
        "Calculate year-over-year sales growth.",
    )

    assert result["table"][1]["Sales Growth %"] is None


# Verify that ambiguous dates use fallback.
def test_ambiguous_dates_use_fallback(annual_data):
    annual_data["Order Date"] = [
        "1/2/23",
        "2/3/23",
        "3/4/24",
        "4/5/24",
    ]

    result = try_annual_analysis(
        annual_data,
        "Show annual sales and profit by year.",
    )

    assert result is None


# Verify that unsupported annual question.
def test_unsupported_annual_question(annual_data):
    result = try_annual_analysis(
        annual_data,
        "Show annual sales for the West region.",
    )

    assert result is None


# Verify that annual analysis preserves dataframe.
def test_annual_analysis_preserves_dataframe(annual_data):
    original = annual_data.copy(deep=True)

    try_annual_analysis(
        annual_data,
        "Show annual sales and profit by year.",
    )

    pd.testing.assert_frame_equal(
        annual_data,
        original,
    )
