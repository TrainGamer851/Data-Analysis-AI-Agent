import math
import re
from numbers import Real, Integral

import pandas as pd

from agent.currency_units import (
    get_explicit_currency,
    format_currency,
)

from agent.debug_logger import debug_log, DebugTimer


MONEY_COLUMNS = {
    "sales",
    "total sales",
    "average sales",
    "average group sales",
    "average total sales",
    "average sales threshold",
    "avg sales threshold",
    "sales threshold",
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
    "total net profit",
    "total gross profit",
    "generated revenue",
    "total generated revenue",
    "operating expenses",
    "total operating expenses",
    "fixed costs",
    "variable costs",
    "unit variable cost",
    "unit selling price",
    "cost of goods sold",
    "cogs",
    "customer acquisition cost",
    "acquisition cost",
    "investment cost",
    "investment return",
    "shipping cost",
    "total shipping cost",
    "average shipping cost",
    "customer lifetime value",
    "average price",
    "average order value",
    "aov",
    "expenses",
    "total expenses",
    "budget",
    "marketing budget",
    "loss",
    "total loss",
    "salary",
    "average salary",
    "avg salary",
    "total salary",
    "salary difference",
    "average salary difference",
    "income",
    "total income",
    "net income",
    "net revenue",
    "expense",
    "total expense",
    "inventory value",
    "total inventory value",
    "stock value",
    "advertising spend",
    "total advertising spend",
    "marketing spend",
    "cost per conversion",
    "cost per acquisition",
    "cost per click",
    "average cost",
    "avg cost",
    "total net profit",
    "total gross profit",
    "generated revenue",
    "total generated revenue",
    "operating expenses",
    "total operating expenses",
    "fixed costs",
    "variable costs",
    "unit variable cost",
    "unit selling price",
    "cost of goods sold",
    "cogs",
    "customer acquisition cost",
    "acquisition cost",
    "investment cost",
    "investment return",
    "shipping cost",
    "total shipping cost",
    "average shipping cost",
    "customer lifetime value",
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
    "stock on hand",
    "reorder level",
    "shortage",
    "additional quantity",
    "required quantity",
    "conversions",
    "clicks",
    "impressions",
    "employees",
    "employee count",
    "total employees",
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

    if name in {
        "discount",
        "avg discount",
        "average discount",
        "mean discount",
        "discount ratio",
        "average discount ratio",
    }:
        return "discount_value"

    # Explicit currency suffixes take priority over
    # traditional monetary column names.
    currency = get_explicit_currency(column)

    if currency is not None:
        return f"money:{currency}"

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

    # Format explicitly identified currencies.
    if column_type.startswith("money:"):
        currency = column_type.split(":", 1)[1]

        return format_currency(
            value,
            currency,
        )

    # Preserve historical dollar formatting for
    # monetary columns without explicit currency units.
    if column_type == "money":
        return format_currency(
            value,
            "USD",
        )

    if column_type == "percentage":
        return f"{value:,.2f}%"

    if column_type == "correlation":
        return f"{value:.4f}"

    if column_type == "discount_value":
        return f"{value:,.4f}"

    if column_type == "count":
        if float(value).is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    if isinstance(value, Integral):
        return f"{value:,}"

    return f"{value:,.2f}"


def format_business_table(df):
    """Return a formatted copy for display purposes only."""

    timer = DebugTimer("TABLE FORMAT")

    debug_log(
        "TABLE FORMAT",
        f"Preparing business table | "
        f"Rows: {len(df):,} | "
        f"Columns: {len(df.columns):,}",
    )

    formatted = df.copy()

    for column in formatted.columns:
        column_type = get_column_type(column)

        debug_log(
            "TABLE FORMAT",
            f"Column: {column} | "
            f"Display type: {column_type}",
        )

        if column_type.startswith("money:"):
            currency = column_type.split(":", 1)[1]

            debug_log(
                "TABLE FORMAT",
                f"Explicit currency detected: {currency} | "
                f"Column: {column}",
            )

        formatted[column] = formatted[column].map(
            lambda value: format_value(
                value,
                column_type,
            )
        )

    timer.finish(
        "Business table formatting completed"
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