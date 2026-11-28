import pandas as pd
import pytest

from app.table_formatter import (
    format_business_table,
    business_table_to_markdown,
    get_column_type,
)


# Verify that money formatting.
def test_money_formatting():
    df = pd.DataFrame(
        {
            "Total Sales": [2326534.3543],
            "Total Profit": [292296.8146],
        }
    )

    result = format_business_table(df)

    assert result["Total Sales"].iloc[0] == "$2,326,534.35"
    assert result["Total Profit"].iloc[0] == "$292,296.81"


# Verify that negative money.
def test_negative_money():
    df = pd.DataFrame(
        {
            "Profit": [-17753.21],
        }
    )

    result = format_business_table(df)

    assert result["Profit"].iloc[0] == "-$17,753.21"


# Verify that zero money.
def test_zero_money():
    df = pd.DataFrame(
        {
            "Sales": [0],
        }
    )

    result = format_business_table(df)

    assert result["Sales"].iloc[0] == "$0.00"


# Verify that percentage formatting.
def test_percentage_formatting():
    df = pd.DataFrame(
        {
            "Profit Margin %": [12.5636148],
            "Average Discount Pct": [15.53845],
        }
    )

    result = format_business_table(df)

    assert result["Profit Margin %"].iloc[0] == "12.56%"
    assert result["Average Discount Pct"].iloc[0] == "15.54%"


# Verify that percentage not multiplied.
def test_percentage_not_multiplied():
    df = pd.DataFrame(
        {
            "Profit Margin %": [0.5],
        }
    )

    result = format_business_table(df)

    assert result["Profit Margin %"].iloc[0] == "0.50%"


# Verify that count formatting.
def test_count_formatting():
    df = pd.DataFrame(
        {
            "Unique Orders": [5111],
            "Unique Customers": [804],
        }
    )

    result = format_business_table(df)

    assert result["Unique Orders"].iloc[0] == "5,111"
    assert result["Unique Customers"].iloc[0] == "804"


# Verify that year and identifier formatting.
def test_year_and_identifier_formatting():
    df = pd.DataFrame(
        {
            "Year": [2026],
            "Postal Code": [92501],
            "Order ID": ["00123"],
        }
    )

    result = format_business_table(df)

    assert result["Year"].iloc[0] == "2026"
    assert result["Postal Code"].iloc[0] == "92501"
    assert result["Order ID"].iloc[0] == "00123"


# Verify that average quantity formatting.
def test_average_quantity_formatting():
    df = pd.DataFrame(
        {
            "Average Quantity": [3.756],
        }
    )

    result = format_business_table(df)

    assert result["Average Quantity"].iloc[0] == "3.76"


# Verify that missing values.
def test_missing_values():
    df = pd.DataFrame(
        {
            "Profit": [float("nan")],
        }
    )

    result = format_business_table(df)

    assert result["Profit"].iloc[0] == "—"


# Verify that text values preserved.
def test_text_values_preserved():
    df = pd.DataFrame(
        {
            "Category": ["Technology", "Furniture"],
        }
    )

    result = format_business_table(df)

    assert result["Category"].tolist() == [
        "Technology",
        "Furniture",
    ]


# Verify that original dataframe not modified.
def test_original_dataframe_not_modified():
    df = pd.DataFrame(
        {
            "Sales": [1234.567],
            "Quantity": [10],
        }
    )

    original = df.copy(deep=True)

    format_business_table(df)

    pd.testing.assert_frame_equal(df, original)


# Verify that markdown preserves formatting.
def test_markdown_preserves_formatting():
    df = pd.DataFrame(
        {
            "Year": [2026],
            "Total Profit": [292296.81],
            "Profit Margin %": [12.56],
        }
    )

    markdown = business_table_to_markdown(df)

    assert "2026" in markdown
    assert "$292,296.81" in markdown
    assert "12.56%" in markdown
    assert "2,026" not in markdown


# Verify that order year without commas.
def test_order_year_without_commas():
    df = pd.DataFrame(
        {
            "Order Year": [2023, 2024, 2025, 2026],
        }
    )

    result = format_business_table(df)

    assert result["Order Year"].tolist() == ["2023", "2024", "2025", "2026"]


# Verify that other year columns without commas.
def test_other_year_columns_without_commas():
    df = pd.DataFrame(
        {
            "Fiscal_Year": [2025],
            "Ship Year": [2026],
        }
    )

    result = format_business_table(df)

    assert result["Fiscal_Year"].iloc[0] == "2025"
    assert result["Ship Year"].iloc[0] == "2026"


# Verify that markdown removes header underscores.
def test_markdown_removes_header_underscores():
    df = pd.DataFrame(
        {
            "Total_Sales": [1000],
            "Total_Profit": [200],
            "Profit_Margin_%": [20],
        }
    )

    markdown = business_table_to_markdown(df)

    assert "Total Sales" in markdown
    assert "Total Profit" in markdown
    assert "Profit Margin %" in markdown
    assert "Total_Sales" not in markdown
    assert "Total_Profit" not in markdown
    assert "Profit_Margin_%" not in markdown


# Verify that percentage header cleanup.
def test_percentage_header_cleanup():
    df = pd.DataFrame(
        {
            "Profit_Margin_Pct": [12.56],
        }
    )

    markdown = business_table_to_markdown(df)

    assert "Profit Margin %" in markdown
    assert "12.56%" in markdown


# Verify that header cleanup preserves original dataframe.
def test_header_cleanup_preserves_original_dataframe():
    df = pd.DataFrame(
        {
            "Order_Year": [2026],
            "Total_Sales": [1000],
        }
    )

    original = df.copy(deep=True)

    business_table_to_markdown(df)

    pd.testing.assert_frame_equal(df, original)


# Verify that correlation precision.
def test_correlation_precision():
    df = pd.DataFrame(
        {
            "Discount_Profit_Correlation": [-0.4782],
        }
    )

    result = format_business_table(df)

    assert result["Discount_Profit_Correlation"].iloc[0] == "-0.4782"


# Verify that fractional discount not guessed as percentage.
def test_fractional_discount_not_guessed_as_percentage():
    df = pd.DataFrame(
        {
            "Avg_Discount": [0.173],
        }
    )

    result = format_business_table(df)

    assert "%" not in result["Avg_Discount"].iloc[0]


# Verify that markdown table escapes pipe characters.
def test_markdown_table_escapes_pipe_characters():
    df = pd.DataFrame(
        {
            "Category": ["Furniture | Office"],
            "Sales": [1000],
        }
    )

    markdown = business_table_to_markdown(df)

    assert r"Furniture \| Office" in markdown


# Verify that markdown table removes cell line breaks.
def test_markdown_table_removes_cell_line_breaks():
    df = pd.DataFrame(
        {
            "Category": ["Office\nSupplies"],
            "Sales": [1000],
        }
    )

    markdown = business_table_to_markdown(df)

    assert "Office Supplies" in markdown


# Verify that ambiguous discount preserves precision.
def test_ambiguous_discount_preserves_precision():
    df = pd.DataFrame(
        {
            "Avg_Discount": [0.173, 0.1563, 0.1315],
        }
    )

    result = format_business_table(df)

    assert result["Avg_Discount"].tolist() == [
        "0.1730",
        "0.1563",
        "0.1315",
    ]


# Verify that explicit discount percentage formatting.
def test_explicit_discount_percentage_formatting():
    df = pd.DataFrame(
        {
            "Average_Discount_Pct": [
                17.30,
                15.63,
                13.15,
            ],
        }
    )

    result = format_business_table(df)

    assert result["Average_Discount_Pct"].tolist() == [
        "17.30%",
        "15.63%",
        "13.15%",
    ]


# Verify that small discount percentage not multiplied.
def test_small_discount_percentage_not_multiplied():
    df = pd.DataFrame(
        {
            "Discount_Pct": [0.5],
        }
    )

    result = format_business_table(df)

    assert result["Discount_Pct"].iloc[0] == "0.50%"


# Verify that discount and correlation markdown.
def test_discount_and_correlation_markdown():
    df = pd.DataFrame(
        {
            "Category": ["Furniture"],
            "Average_Discount_Pct": [17.30],
            "Profit_Margin_Pct": [2.6141],
            "Discount_Profit_Correlation": [-0.4782],
        }
    )

    markdown = business_table_to_markdown(df)

    assert "Average Discount %" in markdown
    assert "17.30%" in markdown
    assert "2.61%" in markdown
    assert "-0.4782" in markdown


# Verify that discount formatting preserves original data.
def test_discount_formatting_preserves_original_data():
    df = pd.DataFrame(
        {
            "Avg_Discount": [0.173],
            "Average_Discount_Pct": [17.30],
        }
    )

    original = df.copy(deep=True)

    format_business_table(df)

    pd.testing.assert_frame_equal(df, original)


# Verify that average sales threshold currency format.
def test_average_sales_threshold_currency_format():
    df = pd.DataFrame(
        {
            "Average_Sales_Threshold": [136854.961],
        }
    )

    result = format_business_table(df)

    assert result["Average_Sales_Threshold"].iloc[0] == "$136,854.96"


# Verify that short sales threshold alias currency format.
def test_short_sales_threshold_alias_currency_format():
    df = pd.DataFrame(
        {
            "Avg_Sales_Threshold": [136854.961],
        }
    )

    result = format_business_table(df)

    assert result["Avg_Sales_Threshold"].iloc[0] == "$136,854.96"


# Verify that additional business currency columns.
def test_additional_business_currency_columns():
    df = pd.DataFrame(
        {
            "Average_Salary": [100000],
            "Total_Inventory_Value": [13430],
            "Net_Income": [2500],
            "Cost_Per_Conversion": [6.25],
            "Advertising_Spend": [500],
        }
    )

    result = format_business_table(df)

    assert result["Average_Salary"].iloc[0] == "$100,000.00"
    assert result["Total_Inventory_Value"].iloc[0] == "$13,430.00"
    assert result["Net_Income"].iloc[0] == "$2,500.00"
    assert result["Cost_Per_Conversion"].iloc[0] == "$6.25"
    assert result["Advertising_Spend"].iloc[0] == "$500.00"


# Verify that additional business percentage columns.
def test_additional_business_percentage_columns():
    df = pd.DataFrame(
        {
            "ROI_Pct": [500],
            "Conversion_Rate_Pct": [8],
            "Discount_Rate_Pct": [25],
        }
    )

    result = format_business_table(df)

    assert result["ROI_Pct"].iloc[0] == "500.00%"
    assert result["Conversion_Rate_Pct"].iloc[0] == "8.00%"
    assert result["Discount_Rate_Pct"].iloc[0] == "25.00%"


# Verify that inventory quantity formatting.
def test_inventory_quantity_formatting():
    df = pd.DataFrame(
        {
            "Stock_On_Hand": [10.0],
            "Reorder_Level": [25.0],
            "Shortage": [15.0],
        }
    )

    result = format_business_table(df)

    assert result["Stock_On_Hand"].iloc[0] == "10"
    assert result["Reorder_Level"].iloc[0] == "25"
    assert result["Shortage"].iloc[0] == "15"


# Verify that expanded business money formatting.
def test_expanded_business_money_formatting():
    df = pd.DataFrame(
        {
            "Total_Net_Profit": [2000],
            "Generated_Revenue": [4500],
            "Customer_Acquisition_Cost": [25.5],
            "Cost_Of_Goods_Sold": [1500],
        }
    )

    result = format_business_table(df)

    assert result["Total_Net_Profit"].iloc[0] == "$2,000.00"
    assert result["Generated_Revenue"].iloc[0] == "$4,500.00"
    assert result["Customer_Acquisition_Cost"].iloc[0] == "$25.50"
    assert result["Cost_Of_Goods_Sold"].iloc[0] == "$1,500.00"


# Verify that roi and roas use different units.
def test_roi_and_roas_use_different_units():
    df = pd.DataFrame(
        {
            "ROI_Pct": [500],
            "ROAS": [6.0],
            "Inventory_Turnover": [4.5],
        }
    )

    result = format_business_table(df)

    assert result["ROI_Pct"].iloc[0] == "500.00%"
    assert result["ROAS"].iloc[0] == "6.00"
    assert result["Inventory_Turnover"].iloc[0] == "4.50"


# Verify that explicit currency table formatting.
@pytest.mark.parametrize(
    "column, value, expected",
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
            "Revenue_INR",
            25000,
            "INR 25,000.00",
        ),
        (
            "Revenue_CNY",
            12000,
            "CNY 12,000.00",
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
            "Revenue_USD",
            0,
            "$0.00",
        ),
        (
            "Revenue_EUR",
            float("nan"),
            "—",
        ),
        (
            "Revenue_GBP",
            float("inf"),
            "—",
        ),
        (
            "Revenue_USD",
            None,
            "—",
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
            "Total_Revenue",
            10000,
            "$10,000.00",
        ),
    ],
)
def test_explicit_currency_table_formatting(
    column,
    value,
    expected,
):
    df = pd.DataFrame(
        {
            column: [value],
        }
    )

    result = format_business_table(df)

    assert result[column].iloc[0] == expected


# Verify that currency column type detection.
@pytest.mark.parametrize(
    "column, expected",
    [
        ("Revenue_USD", "money:USD"),
        ("Revenue_EUR", "money:EUR"),
        ("Revenue_GBP", "money:GBP"),
        ("Revenue_CAD", "money:CAD"),
        ("Revenue", "money"),
        ("Revenue_USD_EUR", "number"),
        ("Revenue_USD_Pct", "percentage"),
        ("Order_ID", "identifier"),
        ("Quantity", "count"),
        (
            "Discount_Profit_Correlation",
            "correlation",
        ),
    ],
)
def test_currency_column_type_detection(
    column,
    expected,
):
    assert get_column_type(column) == expected


# Verify that multiple currency columns remain separate.
def test_multiple_currency_columns_remain_separate():
    df = pd.DataFrame(
        {
            "Revenue_USD": [15000],
            "Revenue_EUR": [8000],
            "Revenue_GBP": [6000],
        }
    )

    result = format_business_table(df)

    assert result["Revenue_USD"].iloc[0] == ("$15,000.00")

    assert result["Revenue_EUR"].iloc[0] == ("€8,000.00")

    assert result["Revenue_GBP"].iloc[0] == ("£6,000.00")


# Verify that currency markdown preserves units.
def test_currency_markdown_preserves_units():
    df = pd.DataFrame(
        {
            "Revenue_USD": [15000],
            "Revenue_EUR": [8000],
            "Revenue_GBP": [6000],
            "Revenue_CAD": [3000],
        }
    )

    markdown = business_table_to_markdown(df)

    assert "$15,000.00" in markdown
    assert "€8,000.00" in markdown
    assert "£6,000.00" in markdown
    assert "CAD 3,000.00" in markdown

    assert "Revenue USD" in markdown
    assert "Revenue EUR" in markdown
    assert "Revenue GBP" in markdown
    assert "Revenue CAD" in markdown


# Verify that currency formatting does not modify data.
def test_currency_formatting_does_not_modify_data():
    df = pd.DataFrame(
        {
            "Revenue_USD": [10000.5678],
            "Revenue_EUR": [8000.1234],
            "Quantity": [210],
        }
    )

    original = df.copy(deep=True)

    format_business_table(df)
    business_table_to_markdown(df)

    pd.testing.assert_frame_equal(df, original)


# Verify that currency and percentage columns together.
def test_currency_and_percentage_columns_together():
    df = pd.DataFrame(
        {
            "Department": ["Marketing"],
            "Revenue_USD": [8000],
            "Net_Profit_USD": [-500],
            "Net_Profit_Margin_Pct": [-6.25],
            "Excluded_Rows": [3],
        }
    )

    result = format_business_table(df)

    assert result["Department"].iloc[0] == ("Marketing")

    assert result["Revenue_USD"].iloc[0] == ("$8,000.00")

    assert result["Net_Profit_USD"].iloc[0] == ("-$500.00")

    assert result["Net_Profit_Margin_Pct"].iloc[0] == ("-6.25%")

    assert result["Excluded_Rows"].iloc[0] == "3"
