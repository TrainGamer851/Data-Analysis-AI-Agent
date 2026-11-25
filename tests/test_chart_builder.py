from app.chart_builder import build_chart


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