import math
import re
from numbers import Real, Integral

import pandas as pd


MONEY_COLUMNS = {
    "sales",
    "total sales",
    "average sales",
    "revenue",
    "total revenue",
    "profit",
    "total profit",
    "average profit",
    "gross profit",
    "net profit",
    "cost",
    "total cost",
    "unit cost",
    "price",
    "unit price",
    "average price",
    "average order value",
    "aov",
    "expenses",
    "total expenses",
    "budget",
    "marketing budget",
    "loss",
    "total loss",
}

PERCENTAGE_COLUMNS = {
    "profit margin",
    "profit margin pct",
    "profit margin percent",
    "gross margin",
    "net margin",
    "average discount percentage",
    "average discount pct",
    "discount percentage",
    "growth percentage",
    "growth rate",
    "sales growth",
    "sales growth pct",
    "profit growth",
    "profit growth pct",
}

COUNT_COLUMNS = {
    "count",
    "orders",
    "total orders",
    "unique orders",
    "customers",
    "total customers",
    "unique customers",
    "customer count",
    "order count",
    "quantity",
    "total quantity",
    "units sold",
    "number of orders",
    "number of customers",
}


def normalize_column(column):
    name = str(column).lower().strip()
    name = name.replace("_", " ")
    name = name.replace("%", " pct ")
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def get_column_type(column):
    """Identify the display type using the column name."""

    name = normalize_column(column)

    if (
        name in {"year", "month", "day", "quarter"}
        or name.endswith((" year", " month", " day", " quarter"))
        or name == "id"
        or name.endswith(" id")
        or name in {"postal code", "zip code"}
    ):
        return "identifier"

    if (
        name in PERCENTAGE_COLUMNS
        or name.endswith(" pct")
        or name.endswith(" percentage")
        or name.endswith(" percent")
    ):
        return "percentage"

    if "correlation" in name:
        return "correlation"

    if name in MONEY_COLUMNS:
        return "money"

    if name in COUNT_COLUMNS:
        return "count"

    return "number"


def format_value(value, column_type):
    """Format a single value without changing its meaning."""

    if value is None or value is pd.NA:
        return "—"

    if isinstance(value, bool):
        return str(value)

    if not isinstance(value, Real):
        return str(value)

    if not math.isfinite(value):
        return "—"

    if column_type == "identifier":
        if isinstance(value, Integral):
            return str(value)

        if float(value).is_integer():
            return str(int(value))

        return str(value)

    if column_type == "money":
        if value < 0:
            return f"-${abs(value):,.2f}"

        return f"${value:,.2f}"

    if column_type == "percentage":
        return f"{value:,.2f}%"

    if column_type == "correlation":
        return f"{value:.4f}"

    if column_type == "count":
        if float(value).is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    if isinstance(value, Integral):
        return f"{value:,}"

    return f"{value:,.2f}"


def format_business_table(df):
    """Return a formatted copy for display purposes only."""

    formatted = df.copy()

    for column in formatted.columns:
        column_type = get_column_type(column)

        formatted[column] = formatted[column].map(
            lambda value: format_value(
                value,
                column_type,
            )
        )

    return formatted


def clean_column_header(column):
    """Convert technical column names into readable labels."""

    name = str(column).replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    name = re.sub(
        r"\b(?:pct|percent|percentage)\b",
        "%",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(r"\s+", " ", name).strip()

    return name


def business_table_to_markdown(df):
    """Convert formatted business data into a Markdown table."""

    def escape_cell(value):
        text = " ".join(str(value).split())

        return re.sub(
            r"([\\`*_{}\[\]()<>#+!|~])",
            r"\\\1",
            text,
        )

    formatted = format_business_table(df)

    formatted.columns = [
        escape_cell(clean_column_header(column))
        for column in formatted.columns
    ]

    for column in formatted.columns:
        formatted[column] = formatted[column].map(
            escape_cell
        )

    return formatted.to_markdown(
        index=False,
        disable_numparse=True,
    )