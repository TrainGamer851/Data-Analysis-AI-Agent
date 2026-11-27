import pandas as pd
import pytest

from agent.data_quality import (
    inspect_numeric_values,
    inspect_date_convention,
    inspect_column_quality,
)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([100, 200], "numeric"),
        (["100", "200"], "numeric_text"),
        (["1,000", "2,000"], "formatted_numeric_text"),
        (["$100", "$200"], "formatted_numeric_text"),
        (["100", "unknown"], "partially_numeric_text"),
        (["Email", "Search"], "not_numeric"),
        ([True, False], "boolean"),
        ([None, None], "empty"),
        ([100, float("inf")], "numeric_with_invalid_values"),
        (["(100)", "(200)"], "formatted_numeric_text"),
        (["10%", "20%"], "formatted_numeric_text"),
        (["USD 100", "USD 200"], "formatted_numeric_text"),
        (["1e3", "2e3"], "numeric_text"),
    ],
)
def test_numeric_compatibility(values, expected):
    quality = inspect_numeric_values(
        pd.Series(values)
    )

    assert quality["numeric_compatibility"] == expected


def test_partial_numeric_counts():
    quality = inspect_numeric_values(
        pd.Series(["100", "unknown", "200", None])
    )

    assert quality["non_missing_count"] == 3
    assert quality["numeric_count"] == 2
    assert quality["invalid_numeric_count"] == 1


def test_nonfinite_values():
    quality = inspect_numeric_values(
        pd.Series([100, float("inf"), float("-inf")])
    )

    assert quality["numeric_count"] == 1
    assert quality["nonfinite_count"] == 2
    assert quality["invalid_numeric_count"] == 2


def test_invalid_thousands_separator():
    quality = inspect_numeric_values(
        pd.Series(["1,00", "100"])
    )

    assert quality["numeric_count"] == 1
    assert quality["invalid_numeric_count"] == 1


def test_dollar_symbol_not_assumed_usd():
    quality = inspect_numeric_values(
        pd.Series(["$100", "$200"])
    )

    assert quality["currency_hints"] == [
        "dollar_symbol"
    ]


def test_multiple_currency_markers():
    quality = inspect_column_quality(
        pd.Series(["$100", "€200"]),
        business_concepts=["revenue"],
    )

    assert "multiple_currency_markers" in (
        quality["warnings"]
    )


def test_currency_header_conflict():
    quality = inspect_column_quality(
        pd.Series(["EUR 100", "EUR 200"]),
        business_concepts=["revenue"],
        unit_hint="USD",
    )

    assert "currency_header_value_conflict" in (
        quality["warnings"]
    )


def test_mixed_percentage_notation():
    quality = inspect_column_quality(
        pd.Series(["10%", "0.20"]),
        business_concepts=["discount"],
    )

    assert "mixed_percentage_notation" in (
        quality["warnings"]
    )


def test_numeric_text_warning():
    quality = inspect_column_quality(
        pd.Series(["100", "200"]),
        business_concepts=["revenue"],
    )

    assert "numeric_conversion_required" in (
        quality["warnings"]
    )


def test_all_values_missing_warning():
    quality = inspect_column_quality(
        pd.Series([None, None]),
        business_concepts=["profit"],
    )

    assert "all_values_missing" in quality["warnings"]


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            ["2026-01-01", "2026-02-01"],
            "iso_date_candidate",
        ),
        (
            ["01/02/2026", "03/04/2026"],
            "ambiguous_date_convention",
        ),
        (
            ["01/25/2026", "02/15/2026"],
            "month_first_candidate",
        ),
        (
            ["25/01/2026", "15/02/2026"],
            "day_first_candidate",
        ),
        (
            ["25/01/2026", "02/15/2026"],
            "conflicting_date_conventions",
        ),
        (
            ["01/25/2026", "unknown"],
            "mixed_or_invalid_date_format",
        ),
        (
            ["32/01/2026"],
            "mixed_or_invalid_date_format",
        ),
        (
            ["unknown", "invalid"],
            "unrecognized_date_format",
        ),
        (
            [None, None],
            "empty",
        ),
    ],
)
def test_date_conventions(values, expected):
    assert inspect_date_convention(
        pd.Series(values)
    ) == expected


def test_ambiguous_date_warning():
    quality = inspect_column_quality(
        pd.Series(["01/02/2026", "03/04/2026"]),
        business_concepts=["order_date"],
    )

    assert "ambiguous_date_convention" in (
        quality["warnings"]
    )


def test_mixed_discount_scale():
    quality = inspect_column_quality(
        pd.Series([0.15, 20]),
        business_concepts=["discount"],
    )

    assert "potentially_mixed_discount_scale" in (
        quality["warnings"]
    )


def test_explicit_discount_percentage():
    quality = inspect_column_quality(
        pd.Series([0.15, 20]),
        business_concepts=["discount"],
        unit_hint="percentage",
    )

    assert "potentially_mixed_discount_scale" not in (
        quality["warnings"]
    )


def test_invalid_discount_ratio():
    quality = inspect_column_quality(
        pd.Series([0.15, 20]),
        business_concepts=["discount"],
        unit_hint="ratio",
    )

    assert "discount_ratio_above_one" in (
        quality["warnings"]
    )


def test_discount_outside_expected_range():
    quality = inspect_column_quality(
        pd.Series([-10, 120]),
        business_concepts=["discount"],
    )

    assert "discount_outside_expected_range" in (
        quality["warnings"]
    )


def test_original_series_not_modified():
    series = pd.Series([
        "$1,000",
        "unknown",
        "(200)",
    ])

    original = series.copy(deep=True)

    inspect_column_quality(
        series,
        business_concepts=["revenue"],
    )

    pd.testing.assert_series_equal(
        series,
        original,
    )
