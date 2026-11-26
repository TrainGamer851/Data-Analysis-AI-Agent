import pandas as pd
import pytest

from agent.dataset_schema import profile_dataset

from agent.data_integrity import (
    validate_analysis_integrity,
    explicit_date_convention,
    explicitly_allows_exclusions,
    explicitly_defines_discount_scale,
)


def validate(df, question):
    return validate_analysis_integrity(
        question,
        profile_dataset(df),
    )


def test_invalid_revenue_requires_clarification():
    df = pd.DataFrame({
        "Revenue_USD": [
            "5000",
            "unknown",
            "3000",
            None,
            "2000",
            "inf",
        ],
    })

    result = validate(
        df,
        "Calculate total revenue.",
    )

    assert result is not None
    assert result["error"] is None
    assert result["table"] is None
    assert "3 row(s)" in result["answer"]
    assert "3 valid numeric row(s)" in result["answer"]


def test_missing_numeric_values_require_clarification():
    df = pd.DataFrame({
        "Revenue": [1000, None, 2000],
    })

    result = validate(
        df,
        "Calculate total revenue.",
    )

    assert result is not None
    assert "1 missing" in result["answer"]


def test_nonfinite_numeric_values_require_clarification():
    df = pd.DataFrame({
        "Revenue": [1000, float("inf"), 2000],
    })

    result = validate(
        df,
        "Calculate total revenue.",
    )

    assert result is not None
    assert "1 invalid" in result["answer"]


def test_clean_numeric_values_are_allowed():
    df = pd.DataFrame({
        "Revenue": [1000, 2000, 3000],
    })

    assert validate(
        df,
        "Calculate total revenue.",
    ) is None


def test_formatted_currency_values_are_allowed():
    df = pd.DataFrame({
        "Revenue_USD": ["$1,000", "$2,000"],
    })

    assert validate(
        df,
        "Calculate total revenue in USD.",
    ) is None


def test_all_missing_revenue_is_rejected():
    df = pd.DataFrame({
        "Revenue": [None, None],
    })

    result = validate(
        df,
        "Calculate total revenue.",
    )

    assert result is not None
    assert "usable numeric values" in result["answer"]


def test_unrelated_invalid_revenue_does_not_block_quantity():
    df = pd.DataFrame({
        "Revenue": ["1000", "unknown"],
        "Quantity": [10, 20],
    })

    assert validate(
        df,
        "Calculate total quantity.",
    ) is None


def test_explicit_valid_values_only_is_allowed():
    df = pd.DataFrame({
        "Revenue": ["1000", "unknown", "2000"],
    })

    assert validate(
        df,
        "Calculate the sum of only the valid finite revenue "
        "values and report how many rows were excluded.",
    ) is None


def test_ambiguous_dates_require_clarification():
    df = pd.DataFrame({
        "Order_Date": [
            "01/02/2026",
            "03/04/2026",
        ],
        "Revenue": [1000, 2000],
    })

    result = validate(
        df,
        "Calculate monthly revenue.",
    )

    assert result is not None
    assert "MM/DD/YYYY" in result["answer"]
    assert "DD/MM/YYYY" in result["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "Calculate monthly revenue using MM/DD/YYYY.",
        "Calculate monthly revenue using DD/MM/YYYY.",
    ],
)
def test_explicit_date_format_is_allowed(question):
    df = pd.DataFrame({
        "Order_Date": [
            "01/02/2026",
            "03/04/2026",
        ],
        "Revenue": [1000, 2000],
    })

    assert validate(df, question) is None


def test_conflicting_dates_require_clarification():
    df = pd.DataFrame({
        "Order_Date": [
            "01/25/2026",
            "25/01/2026",
        ],
        "Revenue": [1000, 2000],
    })

    result = validate(
        df,
        "Calculate monthly revenue.",
    )

    assert result is not None
    assert "conflicting date" in result["answer"]


def test_unrelated_date_ambiguity_is_allowed():
    df = pd.DataFrame({
        "Order_Date": [
            "01/02/2026",
            "03/04/2026",
        ],
        "Revenue": [1000, 2000],
    })

    assert validate(
        df,
        "Calculate total revenue regardless of date.",
    ) is None


def test_mixed_discount_scale_requires_clarification():
    df = pd.DataFrame({
        "Discount": [0.15, 20, 0.10, 30],
    })

    result = validate(
        df,
        "Calculate average discount percentage.",
    )

    assert result is not None
    assert "discount units" in result["answer"]


def test_explicit_discount_conversion_is_allowed():
    df = pd.DataFrame({
        "Discount": [0.15, 20, 0.10, 30],
    })

    assert validate(
        df,
        "Calculate average discount percentage. "
        "Convert fractional values to percentages "
        "and treat values above 1 as percentage points.",
    ) is None


def test_fractional_discount_is_allowed():
    df = pd.DataFrame({
        "Discount": [0.15, 0.20, 0.30],
    })

    assert validate(
        df,
        "Calculate average discount percentage.",
    ) is None


def test_explicit_percentage_column_is_allowed():
    df = pd.DataFrame({
        "Discount_Pct": [0.5, 10, 20],
    })

    assert validate(
        df,
        "Calculate average discount percentage.",
    ) is None


def test_unrelated_discount_problem_is_allowed():
    df = pd.DataFrame({
        "Discount": [0.15, 20],
        "Revenue": [1000, 2000],
    })

    assert validate(
        df,
        "Calculate total revenue.",
    ) is None


def test_unfamiliar_numeric_column_is_allowed():
    df = pd.DataFrame({
        "Billed_Value_USD": [1000, 2000],
    })

    assert validate(
        df,
        "Calculate total billed value.",
    ) is None


def test_explicit_column_takes_priority():
    df = pd.DataFrame({
        "Sales": ["1000", "unknown"],
        "Revenue": [1000, 2000],
    })

    assert validate(
        df,
        "Calculate total revenue.",
    ) is None

    result = validate(
        df,
        "Calculate total sales.",
    )

    assert result is not None
    assert "'Sales'" in result["answer"]


def test_profile_unavailable_preserves_fallback():
    assert validate_analysis_integrity(
        "Calculate total revenue.",
        None,
    ) is None


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({
        "Revenue": ["1000", "unknown"],
    })

    original = df.copy(deep=True)

    validate(
        df,
        "Calculate total revenue.",
    )

    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Use MM/DD/YYYY.", "month_first"),
        ("Use DD/MM/YYYY.", "day_first"),
        ("Use month first.", "month_first"),
        ("Use day first.", "day_first"),
        ("Calculate monthly revenue.", None),
        ("Use MM/DD/YYYY or DD/MM/YYYY.", None),
    ],
)
def test_date_convention_recognition(question, expected):
    assert explicit_date_convention(question) == expected


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Calculate only valid values.", True),
        ("Calculate valid finite revenue.", True),
        ("Exclude invalid values.", True),
        ("Calculate total revenue.", False),
    ],
)
def test_exclusion_permission(question, expected):
    assert explicitly_allows_exclusions(question) == expected


def test_discount_conversion_requires_explicit_rule():
    assert not explicitly_defines_discount_scale(
        "Calculate average discount percentage."
    )

    assert explicitly_defines_discount_scale(
        "Convert fractional values to percentages "
        "and treat values above 1 as percentage points."
    )
