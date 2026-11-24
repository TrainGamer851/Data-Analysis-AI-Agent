import pandas as pd

from app.table_formatter import (
    format_business_table,
    business_table_to_markdown,
)


def test_money_formatting():
    df = pd.DataFrame({
        "Total Sales": [2326534.3543],
        "Total Profit": [292296.8146],
    })

    result = format_business_table(df)

    assert result["Total Sales"].iloc[0] == "$2,326,534.35"
    assert result["Total Profit"].iloc[0] == "$292,296.81"


def test_negative_money():
    df = pd.DataFrame({
        "Profit": [-17753.21],
    })

    result = format_business_table(df)

    assert result["Profit"].iloc[0] == "-$17,753.21"


def test_zero_money():
    df = pd.DataFrame({
        "Sales": [0],
    })

    result = format_business_table(df)

    assert result["Sales"].iloc[0] == "$0.00"


def test_percentage_formatting():
    df = pd.DataFrame({
        "Profit Margin %": [12.5636148],
        "Average Discount Pct": [15.53845],
    })

    result = format_business_table(df)

    assert result["Profit Margin %"].iloc[0] == "12.56%"
    assert result["Average Discount Pct"].iloc[0] == "15.54%"


def test_percentage_not_multiplied():
    df = pd.DataFrame({
        "Profit Margin %": [0.5],
    })

    result = format_business_table(df)

    assert result["Profit Margin %"].iloc[0] == "0.50%"


def test_count_formatting():
    df = pd.DataFrame({
        "Unique Orders": [5111],
        "Unique Customers": [804],
    })

    result = format_business_table(df)

    assert result["Unique Orders"].iloc[0] == "5,111"
    assert result["Unique Customers"].iloc[0] == "804"


def test_year_and_identifier_formatting():
    df = pd.DataFrame({
        "Year": [2026],
        "Postal Code": [92501],
        "Order ID": ["00123"],
    })

    result = format_business_table(df)

    assert result["Year"].iloc[0] == "2026"
    assert result["Postal Code"].iloc[0] == "92501"
    assert result["Order ID"].iloc[0] == "00123"


def test_average_quantity_formatting():
    df = pd.DataFrame({
        "Average Quantity": [3.756],
    })

    result = format_business_table(df)

    assert result["Average Quantity"].iloc[0] == "3.76"


def test_missing_values():
    df = pd.DataFrame({
        "Profit": [float("nan")],
    })

    result = format_business_table(df)

    assert result["Profit"].iloc[0] == "—"


def test_text_values_preserved():
    df = pd.DataFrame({
        "Category": ["Technology", "Furniture"],
    })

    result = format_business_table(df)

    assert result["Category"].tolist() == [
        "Technology",
        "Furniture",
    ]


def test_original_dataframe_not_modified():
    df = pd.DataFrame({
        "Sales": [1234.567],
        "Quantity": [10],
    })

    original = df.copy(deep=True)

    format_business_table(df)

    pd.testing.assert_frame_equal(df, original)


def test_markdown_preserves_formatting():
    df = pd.DataFrame({
        "Year": [2026],
        "Total Profit": [292296.81],
        "Profit Margin %": [12.56],
    })

    markdown = business_table_to_markdown(df)

    assert "2026" in markdown
    assert "$292,296.81" in markdown
    assert "12.56%" in markdown
    assert "2,026" not in markdown

def test_order_year_without_commas():
    df = pd.DataFrame({
        "Order Year": [2023, 2024, 2025, 2026],
    })

    result = format_business_table(df)

    assert result["Order Year"].tolist() == [
        "2023", "2024", "2025", "2026"
    ]


def test_other_year_columns_without_commas():
    df = pd.DataFrame({
        "Fiscal_Year": [2025],
        "Ship Year": [2026],
    })

    result = format_business_table(df)

    assert result["Fiscal_Year"].iloc[0] == "2025"
    assert result["Ship Year"].iloc[0] == "2026"


def test_markdown_removes_header_underscores():
    df = pd.DataFrame({
        "Total_Sales": [1000],
        "Total_Profit": [200],
        "Profit_Margin_%": [20],
    })

    markdown = business_table_to_markdown(df)

    assert "Total Sales" in markdown
    assert "Total Profit" in markdown
    assert "Profit Margin %" in markdown
    assert "Total_Sales" not in markdown
    assert "Total_Profit" not in markdown
    assert "Profit_Margin_%" not in markdown


def test_percentage_header_cleanup():
    df = pd.DataFrame({
        "Profit_Margin_Pct": [12.56],
    })

    markdown = business_table_to_markdown(df)

    assert "Profit Margin %" in markdown
    assert "12.56%" in markdown


def test_header_cleanup_preserves_original_dataframe():
    df = pd.DataFrame({
        "Order_Year": [2026],
        "Total_Sales": [1000],
    })

    original = df.copy(deep=True)

    business_table_to_markdown(df)

    pd.testing.assert_frame_equal(df, original)