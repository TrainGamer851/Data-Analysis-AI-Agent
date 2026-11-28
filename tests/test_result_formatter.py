import pytest

from agent.currency_units import (
    get_explicit_currency,
    format_currency,
)

from agent.result_formatter import (
    format_named_value,
    summarize_single_metric,
    summarize_scalar_correlation,
    extract_single_correlation,
    summarize_correlation_interpretation,
    summarize_multiple_metrics,
    validate_single_metric_explanation,
    validate_multiple_metric_explanation,
)


# Verify that explicit currency detection.
@pytest.mark.parametrize(
    "name, expected",
    [
        ("Total_Revenue_USD", "USD"),
        ("Total_Valid_Revenue_USD", "USD"),
        ("Revenue_EUR", "EUR"),
        ("Net_Profit_GBP", "GBP"),
        ("Revenue_CAD", "CAD"),
        ("Revenue_AUD", "AUD"),
        ("Revenue_JPY", "JPY"),
        ("Revenue_INR", "INR"),
        ("Revenue_CNY", "CNY"),
        ("revenue_usd", "USD"),
        ("Revenue USD", "USD"),
        ("Revenue", None),
        ("Quantity", None),
        ("USD", None),
        ("Revenue_USD_EUR", None),
        ("Revenue_USDollar", None),
        ("Revenue_USD_Pct", None),
    ],
)
def test_explicit_currency_detection(name, expected):
    assert get_explicit_currency(name) == expected


# Verify that currency formatting.
@pytest.mark.parametrize(
    "value, currency, expected",
    [
        (10000, "USD", "$10,000.00"),
        (10000.5, "USD", "$10,000.50"),
        (-1250.5, "USD", "-$1,250.50"),
        (2500.75, "EUR", "€2,500.75"),
        (-1250.5, "GBP", "-£1,250.50"),
        (3000, "CAD", "CAD 3,000.00"),
        (4000, "AUD", "AUD 4,000.00"),
        (0, "USD", "$0.00"),
        (float("nan"), "USD", "Unavailable"),
        (float("inf"), "EUR", "Unavailable"),
        (float("-inf"), "GBP", "Unavailable"),
        (True, "USD", "True"),
    ],
)
def test_currency_formatting(value, currency, expected):
    assert format_currency(value, currency) == expected


# Verify that unsupported currency.
def test_unsupported_currency():
    with pytest.raises(ValueError):
        format_currency(100, "UNKNOWN")


# Verify that named value currency formatting.
@pytest.mark.parametrize(
    "name, value, expected",
    [
        (
            "Total_Revenue_USD",
            10000.0,
            "$10,000.00",
        ),
        (
            "Total_Valid_Revenue_USD",
            10000.5,
            "$10,000.50",
        ),
        (
            "Revenue_EUR",
            8000.0,
            "€8,000.00",
        ),
        (
            "Revenue_GBP",
            6000.0,
            "£6,000.00",
        ),
        (
            "Revenue_CAD",
            3000.0,
            "CAD 3,000.00",
        ),
        (
            "Revenue_AUD",
            4000.0,
            "AUD 4,000.00",
        ),
        (
            "Revenue_JPY",
            150000,
            "JPY 150,000.00",
        ),
        (
            "Net_Profit_USD",
            -500.0,
            "-$500.00",
        ),
        (
            "Net_Profit_EUR",
            -1250.75,
            "-€1,250.75",
        ),
        (
            "Billed_Value_USD",
            42000,
            "$42,000.00",
        ),
        (
            "Direct_Expenditure_USD",
            31000,
            "$31,000.00",
        ),
        (
            "Total_Net_Profit",
            4250.0,
            "$4,250.00",
        ),
        (
            "Gross_Profit",
            5000,
            "$5,000.00",
        ),
        (
            "Operating_Expenses",
            2500,
            "$2,500.00",
        ),
        (
            "Total_Revenue",
            10000,
            "$10,000.00",
        ),
        (
            "Excluded_Rows",
            3,
            "3",
        ),
        (
            "Total_Quantity",
            210,
            "210",
        ),
        (
            "ROI_Pct",
            182.76,
            "182.76%",
        ),
        (
            "Net_Profit_Margin_Pct",
            12.14,
            "12.14%",
        ),
        (
            "Discount_Profit_Correlation",
            -0.4782,
            "-0.4782",
        ),
        (
            "Revenue_USD",
            None,
            "Unavailable",
        ),
        (
            "Revenue_EUR",
            float("inf"),
            "Unavailable",
        ),
        (
            "Revenue_GBP",
            float("nan"),
            "Unavailable",
        ),
        (
            "Revenue_USD_EUR",
            1000,
            "1,000",
        ),
        (
            "Revenue_USD_Pct",
            12.5,
            "12.50%",
        ),
        (
            "Revenue_USD",
            True,
            "True",
        ),
    ],
)
def test_named_value_currency_formatting(
    name,
    value,
    expected,
):
    assert (
        format_named_value(
            name,
            value,
        )
        == expected
    )


# Verify that single metric summary usd.
def test_single_metric_summary_usd():
    answer = summarize_single_metric(
        {
            "Total_Revenue_USD": 10000,
        }
    )

    assert answer == ("Total Revenue USD is **$10,000.00**.")


# Verify that single metric summary eur.
def test_single_metric_summary_eur():
    answer = summarize_single_metric(
        {
            "Total_Revenue_EUR": 8000,
        }
    )

    assert answer == ("Total Revenue EUR is **€8,000.00**.")


# Verify that single metric summary negative gbp.
def test_single_metric_summary_negative_gbp():
    answer = summarize_single_metric(
        {
            "Net_Profit_GBP": -500,
        }
    )

    assert answer == ("Net Profit GBP is **-£500.00**.")


# Verify that single metric percentage unchanged.
def test_single_metric_percentage_unchanged():
    answer = summarize_single_metric(
        {
            "ROI_Pct": 182.76,
        }
    )

    assert answer == ("ROI % is **182.76%**.")


# Verify that currency formatting preserves original result.
def test_currency_formatting_preserves_original_result():
    result = {
        "Total_Revenue_USD": 10000.5678,
    }

    original = result.copy()

    summarize_single_metric(result)

    assert result == original


# --------------------------------------------------
# Explicit currency explanation validation
# --------------------------------------------------


# Verify that explicit currency explanation.
@pytest.mark.parametrize(
    "name, value, explanation, expected",
    [
        (
            "Revenue_USD",
            10000,
            "Revenue is $10,000.00.",
            True,
        ),
        (
            "Revenue_EUR",
            8000,
            "Revenue is €8,000.00.",
            True,
        ),
        (
            "Revenue_GBP",
            6000,
            "Revenue is £6,000.00.",
            True,
        ),
        (
            "Revenue_CAD",
            3000,
            "Revenue is CAD 3,000.00.",
            True,
        ),
        (
            "Revenue_AUD",
            4000,
            "Revenue is AUD 4,000.00.",
            True,
        ),
        (
            "Revenue_EUR",
            8000,
            "Revenue is $8,000.00.",
            False,
        ),
        (
            "Revenue_GBP",
            6000,
            "Revenue is €6,000.00.",
            False,
        ),
        (
            "Revenue_USD",
            10000,
            "Profit is $10,000.00.",
            False,
        ),
        (
            "Revenue_USD",
            10000,
            "Revenue is $12,000.00.",
            False,
        ),
        (
            "Revenue_USD",
            10000,
            "Revenue is -$10,000.00.",
            False,
        ),
        (
            "Revenue_USD",
            10000,
            "Revenue is not $10,000.00.",
            False,
        ),
        (
            "Net_Profit_GBP",
            -500,
            "Net Profit is -£500.00.",
            True,
        ),
        (
            "Revenue_USD",
            10000.5678,
            "Revenue is $10,000.57.",
            True,
        ),
        (
            "Revenue_USD",
            10000,
            "Revenue is 10,000.00.",
            False,
        ),
    ],
)
def test_explicit_currency_explanation(
    name,
    value,
    explanation,
    expected,
):
    assert (
        validate_single_metric_explanation(
            explanation,
            {name: value},
        )
        is expected
    )


# --------------------------------------------------
# Multi-metric summaries
# --------------------------------------------------


# Verify that valid revenue exclusion summary.
def test_valid_revenue_exclusion_summary():
    result = {
        "Total_Valid_Revenue_USD": 10000,
        "Excluded_Rows": 3,
    }

    assert summarize_multiple_metrics(result) == (
        "The total valid revenue is " "**$10,000.00**, excluding **3 rows**."
    )


# Verify that single excluded row summary.
def test_single_excluded_row_summary():
    result = {
        "Total_Valid_Revenue_EUR": 8000,
        "Excluded_Rows": 1,
    }

    assert summarize_multiple_metrics(result) == (
        "The total valid revenue is " "**€8,000.00**, excluding **1 row**."
    )


# Verify that independent metric summary.
def test_independent_metric_summary():
    result = {
        "Total_Revenue_USD": 10000,
        "Net_Profit_USD": 2000,
        "Profit_Margin_Pct": 20,
    }

    answer = summarize_multiple_metrics(result)

    assert "Total Revenue USD is **$10,000.00**." in answer
    assert "Net Profit USD is **$2,000.00**." in answer
    assert "Profit Margin % is **20.00%**." in answer


# Verify that unsupported multi metric summary.
@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"Revenue_USD": 1000},
        {"Revenue_USD": 1000, "Profit": None},
        {"Revenue_USD": 1000, "Profit": float("inf")},
        {"Revenue_USD": 1000, "Profit": float("nan")},
        {"Revenue_USD": 1000, "Valid": True},
        {"Revenue_USD": 1000, "Region": "West"},
        {"Revenue_USD": 1000, "Details": [1, 2]},
    ],
)
def test_unsupported_multi_metric_summary(result):
    assert summarize_multiple_metrics(result) is None


# Verify that multi metric explanation validation.
@pytest.mark.parametrize(
    "explanation, expected",
    [
        (
            "The total valid revenue is " "$10,000.00, excluding 3 rows.",
            True,
        ),
        (
            "The total valid revenue is " "**$10,000.00**, excluding **3 rows**.",
            True,
        ),
        (
            "The total valid revenue is " "$12,000.00, excluding 3 rows.",
            False,
        ),
        (
            "The total valid revenue is " "$10,000.00, excluding 5 rows.",
            False,
        ),
        (
            "The total valid revenue is " "€10,000.00, excluding 3 rows.",
            False,
        ),
        (
            "The total valid revenue is " "-$10,000.00, excluding 3 rows.",
            False,
        ),
        (
            "The total valid revenue is $10,000.00.",
            False,
        ),
        (
            "The total valid profit is " "$10,000.00, excluding 3 rows.",
            False,
        ),
        (
            "The total valid revenue is "
            "$10,000.00, excluding 3 rows. "
            "Revenue will increase next year.",
            False,
        ),
        (
            "The total valid revenue is not " "$10,000.00, excluding 3 rows.",
            False,
        ),
    ],
)
def test_multi_metric_explanation_validation(
    explanation,
    expected,
):
    result = {
        "Total_Valid_Revenue_USD": 10000,
        "Excluded_Rows": 3,
    }

    assert (
        validate_multiple_metric_explanation(
            explanation,
            result,
        )
        is expected
    )


# Verify that multi metric swapped values rejected.
def test_multi_metric_swapped_values_rejected():
    result = {
        "Total_Revenue_USD": 10000,
        "Net_Profit_USD": 2000,
    }

    explanation = "Total Revenue USD is $2,000.00. " "Net Profit USD is $10,000.00."

    assert not validate_multiple_metric_explanation(
        explanation,
        result,
    )


# Verify that multi metric summary preserves original values.
def test_multi_metric_summary_preserves_original_values():
    result = {
        "Total_Revenue_USD": 10000.5678,
        "Net_Profit_USD": 2000.1234,
    }

    original = result.copy()

    summarize_multiple_metrics(result)

    assert result == original


# Verify that multi metric correlation disclaimer.
def test_multi_metric_correlation_disclaimer():
    result = {
        "Total_Revenue_USD": 10000,
        "Discount_Profit_Correlation": -0.4782,
    }

    answer = summarize_multiple_metrics(result)

    assert "-0.4782" in answer
    assert "Correlation does not establish causation." in answer


# --------------------------------------------------
# Consistent percentage and correlation presentation
# --------------------------------------------------


# Verify that single metric percentage precision.
@pytest.mark.parametrize(
    "value, expected",
    [
        (20, "20.00%"),
        (12.5, "12.50%"),
        (12.3456, "12.35%"),
        (-5, "-5.00%"),
        (0, "0.00%"),
        (100, "100.00%"),
    ],
)
def test_single_metric_percentage_precision(
    value,
    expected,
):
    result = {
        "Profit_Margin_Pct": value,
    }

    answer = summarize_single_metric(result)

    assert answer == (f"Profit Margin % is **{expected}**.")


# Verify that single metric correlation disclaimer.
@pytest.mark.parametrize(
    "value, expected",
    [
        (-0.478234, "-0.4782"),
        (0.823456, "0.8235"),
        (0, "0.0000"),
        (1, "1.0000"),
        (-1, "-1.0000"),
    ],
)
def test_single_metric_correlation_disclaimer(
    value,
    expected,
):
    result = {
        "Discount_Profit_Correlation": value,
    }

    answer = summarize_single_metric(result)

    assert answer == (
        f"Discount Profit Correlation is "
        f"**{expected}**.\n\n"
        "Correlation does not establish causation."
    )


# Verify that single metric correlation preserves value.
def test_single_metric_correlation_preserves_value():
    result = {
        "Discount_Profit_Correlation": -0.47823456,
    }

    original = result.copy()

    summarize_single_metric(result)

    assert result == original


# Verify that non correlation has no disclaimer.
def test_non_correlation_has_no_disclaimer():
    answer = summarize_single_metric(
        {
            "Total_Revenue_USD": 10000,
        }
    )

    assert "Correlation does not establish causation." not in answer


# --------------------------------------------------
# Scalar correlation presentation
# --------------------------------------------------


# Verify that scalar correlation precision.
@pytest.mark.parametrize(
    "value, expected",
    [
        (0.048992244529133366, "0.0490"),
        (-0.47823456, "-0.4782"),
        (0.82345678, "0.8235"),
        (0, "0.0000"),
        (1, "1.0000"),
        (-1, "-1.0000"),
    ],
)
def test_scalar_correlation_precision(
    value,
    expected,
):
    answer = summarize_scalar_correlation(
        "Calculate the Pearson correlation " "between Discount_Pct and Net_Profit_USD.",
        value,
    )

    assert answer == (
        "The Pearson correlation coefficient is "
        f"**{expected}**.\n\n"
        "Correlation does not establish causation."
    )


# Verify that scalar correlation method.
@pytest.mark.parametrize(
    "question, label",
    [
        (
            "Calculate the Pearson correlation.",
            "The Pearson correlation coefficient",
        ),
        (
            "Compute the Spearman correlation.",
            "The Spearman correlation coefficient",
        ),
        (
            "Find the Kendall correlation.",
            "The Kendall correlation coefficient",
        ),
        (
            "What is the correlation coefficient?",
            "The correlation coefficient",
        ),
        (
            "Please calculate the correlation.",
            "The correlation coefficient",
        ),
        (
            "CALCULATE THE PEARSON CORRELATION.",
            "The Pearson correlation coefficient",
        ),
    ],
)
def test_scalar_correlation_method(
    question,
    label,
):
    answer = summarize_scalar_correlation(
        question,
        0.512345,
    )

    assert answer == (
        f"{label} is **0.5123**.\n\n" "Correlation does not establish causation."
    )


# Verify that invalid scalar correlation.
@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "0.5",
        float("nan"),
        float("inf"),
        float("-inf"),
        1.1,
        -1.1,
        {"Correlation": 0.5},
        [0.5],
    ],
)
def test_invalid_scalar_correlation(value):
    assert (
        summarize_scalar_correlation(
            "Calculate the Pearson correlation.",
            value,
        )
        is None
    )


# Verify that unsupported scalar correlation question.
@pytest.mark.parametrize(
    "question",
    [
        "Calculate the total revenue.",
        "Calculate the average discount.",
        "Calculate the overall profit margin.",
        "Calculate the correlation matrix.",
        "Calculate the correlation p-value.",
        "Calculate the correlation significance.",
        "Calculate the correlation by region.",
        "Calculate the correlation by category.",
        "Calculate the correlation by department.",
        "Calculate the correlation and also " "the total revenue.",
        "Calculate the correlation as well as " "the average profit.",
        "Find the strongest correlation.",
        "What is the number of correlations?",
        "Does correlation imply causation?",
        "",
        None,
    ],
)
def test_unsupported_scalar_correlation_question(
    question,
):
    assert (
        summarize_scalar_correlation(
            question,
            0.5,
        )
        is None
    )


# Verify that scalar correlation preserves precision.
def test_scalar_correlation_preserves_precision():
    value = 0.048992244529133366

    answer = summarize_scalar_correlation(
        "Calculate the Pearson correlation.",
        value,
    )

    assert "**0.0490**" in answer
    assert value == 0.048992244529133366
    assert answer.count("Correlation does not establish causation.") == 1


# --------------------------------------------------
# Statistical explanation integrity
# --------------------------------------------------

CORRELATION_QUESTION = (
    "Calculate the Pearson correlation " "between Sales and Profit and explain it."
)


# Verify that extract scalar correlation.
def test_extract_scalar_correlation():
    value = 0.048992244529133366

    assert (
        extract_single_correlation(
            CORRELATION_QUESTION,
            value,
        )
        == value
    )


# Verify that extract named correlation.
def test_extract_named_correlation():
    result = {
        "Sales_Profit_Pearson_Correlation": 0.5,
    }

    assert (
        extract_single_correlation(
            CORRELATION_QUESTION,
            result,
        )
        == 0.5
    )


# Verify that discard unverified correlation narrative.
@pytest.mark.parametrize(
    "field",
    [
        "Interpretation",
        "Explanation",
        "Recommendation",
        "Conclusion",
        "Business_Interpretation",
        "Business_Implications",
        "Summary",
        "Insights",
    ],
)
def test_discard_unverified_correlation_narrative(field):
    result = {
        "Sales_Profit_Pearson_Correlation": 0.049,
        field: "Increasing sales causes higher profit.",
    }

    original = result.copy()

    answer = summarize_correlation_interpretation(
        CORRELATION_QUESTION,
        result,
    )

    assert "**0.0490**" in answer
    assert "causes higher profit" not in answer
    assert result == original


# Verify that unsupported correlation evidence.
@pytest.mark.parametrize(
    "result",
    [
        None,
        True,
        "0.5",
        [],
        {},
        float("nan"),
        float("inf"),
        -1.1,
        1.1,
        {"Revenue_USD": 0.5},
        {"Correlation": True},
        {"Correlation": float("nan")},
        {"Correlation": 1.1},
        {"Correlation_Count": 1},
        {"Correlation_P_Value": 0.05},
        {"Spearman_Correlation": 0.5},
        {
            "Pearson_Correlation": 0.5,
            "Spearman_Correlation": 0.4,
        },
        {
            "Pearson_Correlation": 0.5,
            "Excluded_Rows": 3,
        },
        {
            "Pearson_Correlation": 0.5,
            "Exclusion_Reason": "Missing values",
        },
        {
            "Pearson_Correlation": 0.5,
            "Revenue_USD": 10000,
        },
    ],
)
def test_unsupported_correlation_evidence(result):
    assert (
        extract_single_correlation(
            CORRELATION_QUESTION,
            result,
        )
        is None
    )


# Verify that correlation interpretation strength.
@pytest.mark.parametrize(
    "value, expected",
    [
        (0.049, "very weak positive"),
        (-0.049, "very weak negative"),
        (0.1, "weak positive"),
        (-0.2, "weak negative"),
        (0.3, "moderate positive"),
        (-0.4, "moderate negative"),
        (0.5, "strong positive"),
        (-0.6, "strong negative"),
        (0.7, "very strong positive"),
        (-0.9, "very strong negative"),
        (1.0, "very strong positive"),
        (-1.0, "very strong negative"),
    ],
)
def test_correlation_interpretation_strength(
    value,
    expected,
):
    answer = summarize_correlation_interpretation(
        CORRELATION_QUESTION,
        value,
    )

    assert expected in answer
    assert f"**{value:.4f}**" in answer

    assert answer.count("Correlation does not establish causation.") == 1


# Verify that method specific correlation interpretation.
@pytest.mark.parametrize(
    "method, expected",
    [
        ("Pearson", "linear association"),
        ("Spearman", "monotonic association"),
        ("Kendall", "rank association"),
    ],
)
def test_method_specific_correlation_interpretation(
    method,
    expected,
):
    answer = summarize_correlation_interpretation(
        f"Calculate the {method} correlation "
        "between Sales and Profit and explain it.",
        0.5,
    )

    assert expected in answer
    assert f"The {method} correlation" in answer


# Verify that zero correlation interpretation.
def test_zero_correlation_interpretation():
    answer = summarize_correlation_interpretation(
        CORRELATION_QUESTION,
        0,
    )

    assert "**0.0000**" in answer
    assert "no linear association" in answer


# Verify that correlation interpretation preserves precision.
def test_correlation_interpretation_preserves_precision():
    value = 0.048992244529133366

    answer = summarize_correlation_interpretation(
        CORRELATION_QUESTION,
        value,
    )

    assert "**0.0490**" in answer
    assert value == 0.048992244529133366


# Verify that correlation interpretation unrelated question.
def test_correlation_interpretation_unrelated_question():
    assert (
        summarize_correlation_interpretation(
            "Calculate the ratio of Sales to Profit.",
            0.5,
        )
        is None
    )
