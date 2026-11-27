import pytest

from app.chart_builder import (
    build_chart,
    get_chart_currency,
)

def test_category_profit_bar_chart():
    table = [
        {"Category": "Technology", "Profit": 1000},
        {"Category": "Furniture", "Profit": 200},
    ]

    figure = build_chart(table)

    assert figure is not None
    assert figure.data[0].type == "bar"


def test_annual_sales_line_chart():
    table = [
        {"Year": 2023, "Sales": 1000},
        {"Year": 2024, "Sales": 1500},
    ]

    figure = build_chart(table)

    assert figure is not None
    assert figure.data[0].type == "scatter"


def test_growth_bar_chart():
    table = [
        {"Year": 2023, "Sales Growth %": None},
        {"Year": 2024, "Sales Growth %": 25.0},
        {"Year": 2025, "Sales Growth %": -10.0},
    ]

    figure = build_chart(table)

    assert figure is not None
    assert figure.data[0].type == "bar"


def test_empty_table():
    assert build_chart([]) is None


def test_missing_table():
    assert build_chart(None) is None


def test_single_row_table():
    table = [
        {"Category": "Technology", "Profit": 1000},
    ]

    assert build_chart(table) is None


def test_text_only_table():
    table = [
        {"Category": "Technology", "Status": "Good"},
        {"Category": "Furniture", "Status": "Poor"},
    ]

    assert build_chart(table) is None


def test_multiple_financial_metrics():
    table = [
        {
            "Category": "Technology",
            "Sales": 1000,
            "Profit": 200,
        },
        {
            "Category": "Furniture",
            "Sales": 800,
            "Profit": 50,
        },
    ]

    figure = build_chart(table)

    assert len(figure.data) == 2


def test_incompatible_units_not_combined():
    table = [
        {
            "Category": "Technology",
            "Sales": 1000,
            "Profit Margin %": 20,
        },
        {
            "Category": "Furniture",
            "Sales": 800,
            "Profit Margin %": 10,
        },
    ]

    figure = build_chart(table)

    assert len(figure.data) == 1


def test_years_sorted_chronologically():
    table = [
        {"Year": 2025, "Sales": 1500},
        {"Year": 2023, "Sales": 1000},
        {"Year": 2024, "Sales": 1200},
    ]

    figure = build_chart(table)

    assert list(figure.data[0].x) == [
        "2023",
        "2024",
        "2025",
    ]


def test_chart_preserves_calculated_values():
    table = [
        {"Region": "West", "Profit": 110798.817},
        {"Region": "East", "Profit": 94883.2603},
    ]

    figure = build_chart(table)

    assert list(figure.data[0].y) == [
        110798.817,
        94883.2603,
    ]


def test_duplicate_categories_not_charted():
    table = [
        {"Category": "Technology", "Profit": 100},
        {"Category": "Technology", "Profit": 200},
    ]

    assert build_chart(table) is None

def test_index_column_ignored():
    table = [
        {"index": 3, "Region": "West", "Profit": 1000},
        {"index": 1, "Region": "East", "Profit": 800},
    ]

    figure = build_chart(table)

    assert figure is not None
    assert len(figure.data) == 1
    assert figure.data[0].name == "Profit"
    assert list(figure.data[0].y) == [1000, 800]


def test_growth_prioritized_over_sales():
    table = [
        {
            "Year": 2023,
            "Total Sales": 1000,
            "Sales Growth %": None,
        },
        {
            "Year": 2024,
            "Total Sales": 1500,
            "Sales Growth %": 50.0,
        },
    ]

    figure = build_chart(table)

    assert figure is not None
    assert figure.data[0].type == "bar"
    assert figure.data[0].name == "Sales Growth %"
    assert figure.data[0].y[1] == 50.0
    assert figure.layout.yaxis.ticksuffix == "%"


def test_negative_growth_color():
    table = [
        {"Year": 2024, "Sales Growth %": -4.26},
        {"Year": 2025, "Sales Growth %": 29.80},
    ]

    figure = build_chart(table)

    assert list(figure.data[0].marker.color) == [
        "#EF4444",
        "#3B82F6",
    ]


def test_dark_chart_background():
    table = [
        {"Region": "West", "Profit": 1000},
        {"Region": "East", "Profit": 800},
    ]

    figure = build_chart(table)

    assert figure.layout.paper_bgcolor == "#111827"
    assert figure.layout.plot_bgcolor == "#111827"


def test_index_only_is_not_charted():
    table = [
        {"index": 0, "Region": "West"},
        {"index": 1, "Region": "East"},
    ]

    assert build_chart(table) is None


# --------------------------------------------------
# Currency-aware chart formatting
# --------------------------------------------------

@pytest.mark.parametrize(
    "column_type, expected",
    [
        ("money", "USD"),
        ("money:USD", "USD"),
        ("money:EUR", "EUR"),
        ("money:GBP", "GBP"),
        ("money:CAD", "CAD"),
        ("money:AUD", "AUD"),
        ("money:JPY", "JPY"),
        ("money:INR", "INR"),
        ("money:CNY", "CNY"),
        ("money:UNKNOWN", None),
        ("percentage", None),
        ("count", None),
        ("number", None),
        ("identifier", None),
    ],
)
def test_chart_currency_resolution(
    column_type,
    expected,
):
    assert get_chart_currency(
        column_type
    ) == expected


@pytest.mark.parametrize(
    "currency, prefix",
    [
        ("USD", "$"),
        ("EUR", "€"),
        ("GBP", "£"),
        ("CAD", "CAD "),
        ("AUD", "AUD "),
        ("JPY", "JPY "),
        ("INR", "INR "),
        ("CNY", "CNY "),
    ],
)
def test_explicit_currency_chart(
    currency,
    prefix,
):
    column = f"Revenue_{currency}"

    table = [
        {
            "Region": "West",
            column: 10000.5678,
        },
        {
            "Region": "East",
            column: 8000.1234,
        },
    ]

    figure = build_chart(table)

    assert figure is not None
    assert len(figure.data) == 1

    assert figure.layout.yaxis.title.text == (
        f"Amount ({currency})"
    )

    assert figure.layout.yaxis.tickprefix == prefix

    assert figure.layout.yaxis.tickformat == ",.2f"

    assert list(figure.data[0].customdata) == [
        f"{prefix}10,000.57",
        f"{prefix}8,000.12",
    ]

    assert "%{customdata}" in (
        figure.data[0].hovertemplate
    )

    # Preserve original calculated values.
    assert list(figure.data[0].y) == [
        10000.5678,
        8000.1234,
    ]


def test_negative_currency_hover():
    table = [
        {
            "Region": "West",
            "Net_Profit_EUR": -1250.7543,
        },
        {
            "Region": "East",
            "Net_Profit_EUR": 800.1234,
        },
    ]

    figure = build_chart(table)

    assert list(figure.data[0].customdata) == [
        "-€1,250.75",
        "€800.12",
    ]

    assert list(figure.data[0].y) == [
        -1250.7543,
        800.1234,
    ]


def test_same_currency_metrics_combined():
    table = [
        {
            "Region": "West",
            "Revenue_EUR": 10000,
            "Net_Profit_EUR": 2000,
        },
        {
            "Region": "East",
            "Revenue_EUR": 8000,
            "Net_Profit_EUR": 1500,
        },
    ]

    figure = build_chart(table)

    assert len(figure.data) == 2

    assert figure.layout.yaxis.title.text == (
        "Amount (EUR)"
    )


def test_different_currencies_not_combined():
    table = [
        {
            "Region": "West",
            "Revenue_USD": 10000,
            "Revenue_EUR": 8000,
        },
        {
            "Region": "East",
            "Revenue_USD": 5000,
            "Revenue_EUR": 4000,
        },
    ]

    figure = build_chart(table)

    assert len(figure.data) == 1

    assert figure.layout.yaxis.title.text == (
        "Amount (USD)"
    )

    assert list(figure.data[0].y) == [
        10000,
        5000,
    ]


def test_legacy_money_chart_unchanged():
    table = [
        {"Region": "West", "Sales": 1000},
        {"Region": "East", "Sales": 800},
    ]

    figure = build_chart(table)

    assert figure.layout.yaxis.title.text == (
        "Amount (USD)"
    )

    assert figure.layout.yaxis.tickprefix == "$"

    assert list(figure.data[0].customdata) == [
        "$1,000.00",
        "$800.00",
    ]


def test_currency_annual_line_chart():
    table = [
        {"Year": 2025, "Revenue_GBP": 15000},
        {"Year": 2023, "Revenue_GBP": 10000},
        {"Year": 2024, "Revenue_GBP": 12000},
    ]

    figure = build_chart(table)

    assert figure.data[0].type == "scatter"

    assert list(figure.data[0].x) == [
        "2023",
        "2024",
        "2025",
    ]

    assert list(figure.data[0].customdata) == [
        "£10,000.00",
        "£12,000.00",
        "£15,000.00",
    ]

    assert figure.layout.yaxis.tickprefix == "£"


def test_currency_missing_value_hover():
    table = [
        {"Region": "West", "Revenue_EUR": 1000},
        {"Region": "East", "Revenue_EUR": None},
    ]

    figure = build_chart(table)

    assert figure is not None

    assert list(figure.data[0].customdata) == [
        "€1,000.00",
        "Unavailable",
    ]


def test_currency_and_percentage_not_combined():
    table = [
        {
            "Region": "West",
            "Revenue_EUR": 10000,
            "Profit_Margin_Pct": 20,
        },
        {
            "Region": "East",
            "Revenue_EUR": 8000,
            "Profit_Margin_Pct": 15,
        },
    ]

    figure = build_chart(table)

    assert len(figure.data) == 1

    assert figure.layout.yaxis.title.text == (
        "Amount (EUR)"
    )


def test_currency_growth_priority_preserved():
    table = [
        {
            "Year": 2023,
            "Revenue_EUR": 10000,
            "Revenue_Growth_Pct": None,
        },
        {
            "Year": 2024,
            "Revenue_EUR": 12000,
            "Revenue_Growth_Pct": 20,
        },
    ]

    figure = build_chart(table)

    assert len(figure.data) == 1
    assert figure.data[0].type == "bar"

    assert figure.layout.yaxis.title.text == (
        "Percentage"
    )

    assert figure.layout.yaxis.ticksuffix == "%"


def test_currency_chart_preserves_input():
    table = [
        {
            "Region": "West",
            "Revenue_EUR": 10000.5678,
        },
        {
            "Region": "East",
            "Revenue_EUR": 8000.1234,
        },
    ]

    original = [
        row.copy()
        for row in table
    ]

    build_chart(table)

    assert table == original


# --------------------------------------------------
# Generalized categorical chart dimensions
# --------------------------------------------------


@pytest.mark.parametrize(
    "dimension",
    [
        "Department",
        "Merchandise_Family",
        "Marketing_Channel",
        "Employee_Role",
        "Supplier",
        "Warehouse",
        "Business_Unit",
        "Customer_Type",
    ],
)
def test_inferred_chart_dimension(dimension):
    table = [
        {
            dimension: "Group A",
            "Revenue_USD": 10000,
        },
        {
            dimension: "Group B",
            "Revenue_USD": 8000,
        },
    ]

    figure = build_chart(table)

    assert figure is not None
    assert figure.data[0].type == "bar"

    assert list(figure.data[0].x) == [
        "Group A",
        "Group B",
    ]

    assert list(figure.data[0].y) == [
        10000,
        8000,
    ]

    assert figure.layout.yaxis.tickprefix == "$"


def test_department_financial_chart():
    table = [
        {
            "Department": "Engineering",
            "Total_Revenue_USD": 12500,
            "Total_Net_Profit_USD": 2500,
        },
        {
            "Department": "Marketing",
            "Total_Revenue_USD": 8000,
            "Total_Net_Profit_USD": -500,
        },
        {
            "Department": "Finance",
            "Total_Revenue_USD": 10500,
            "Total_Net_Profit_USD": 1500,
        },
        {
            "Department": "Operations",
            "Total_Revenue_USD": 4000,
            "Total_Net_Profit_USD": 750,
        },
    ]

    figure = build_chart(table)

    assert figure is not None
    assert len(figure.data) == 2

    assert list(figure.data[1].y) == [
        2500,
        -500,
        1500,
        750,
    ]

    assert figure.layout.yaxis.title.text == (
        "Amount (USD)"
    )


def test_unfamiliar_business_dimension_chart():
    table = [
        {
            "Merchandise_Family": "Electronics",
            "Billed_Value_USD": 20000,
        },
        {
            "Merchandise_Family": "Furniture",
            "Billed_Value_USD": 12000,
        },
        {
            "Merchandise_Family": "Clothing",
            "Billed_Value_USD": 10000,
        },
    ]

    figure = build_chart(table)

    assert figure is not None

    assert list(figure.data[0].y) == [
        20000,
        12000,
        10000,
    ]


@pytest.mark.parametrize(
    "dimension",
    [
        "Order_ID",
        "Employee_ID",
        "Product_SKU",
        "Customer_Email",
        "Transaction_UUID",
        "Description",
        "Notes",
    ],
)
def test_identifier_or_free_text_not_inferred(dimension):
    table = [
        {
            dimension: "Value A",
            "Revenue_USD": 1000,
        },
        {
            dimension: "Value B",
            "Revenue_USD": 2000,
        },
    ]

    assert build_chart(table) is None


def test_multiple_inferred_dimensions_rejected():
    table = [
        {
            "Department": "Engineering",
            "Warehouse": "North",
            "Revenue_USD": 1000,
        },
        {
            "Department": "Finance",
            "Warehouse": "South",
            "Revenue_USD": 2000,
        },
    ]

    assert build_chart(table) is None


def test_known_and_inferred_dimensions_rejected():
    table = [
        {
            "Region": "West",
            "Department": "Engineering",
            "Revenue_USD": 1000,
        },
        {
            "Region": "East",
            "Department": "Finance",
            "Revenue_USD": 2000,
        },
    ]

    assert build_chart(table) is None


def test_duplicate_inferred_dimension_rejected():
    table = [
        {
            "Department": "Finance",
            "Revenue_USD": 1000,
        },
        {
            "Department": "Finance",
            "Revenue_USD": 2000,
        },
    ]

    assert build_chart(table) is None


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        "",
        "   ",
        "A" * 81,
    ],
)
def test_invalid_inferred_dimension_rejected(
    invalid_value,
):
    table = [
        {
            "Department": "Finance",
            "Revenue_USD": 1000,
        },
        {
            "Department": invalid_value,
            "Revenue_USD": 2000,
        },
    ]

    assert build_chart(table) is None


def test_inferred_dimension_preserves_input():
    table = [
        {
            "Department": "Engineering",
            "Revenue_EUR": 1000.5678,
        },
        {
            "Department": "Finance",
            "Revenue_EUR": 2000.1234,
        },
    ]

    original = [
        row.copy()
        for row in table
    ]

    figure = build_chart(table)

    assert figure is not None
    assert table == original

    assert list(figure.data[0].y) == [
        1000.5678,
        2000.1234,
    ]