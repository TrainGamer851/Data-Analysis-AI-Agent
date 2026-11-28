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


# Normalize a column label into lowercase words and standardize percent notation.
def normalize_column(column):
    name = str(column).lower().strip()
    name = name.replace("_", " ")
    name = name.replace("%", " pct ")
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# Classify a column for display using its label, business meaning, and currency suffix.
def get_column_type(column):

    name = normalize_column(column)

    # Treat date parts, IDs, and postal codes as identifiers so they are not formatted as measurements.
    if (
        name in {"year", "month", "day", "quarter"}
        or name.endswith((" year", " month", " day", " quarter"))
        or name == "id"
        or name.endswith(" id")
        or name in {"postal code", "zip code"}
    ):
        return "identifier"

    # Recognize percentage labels and suffixes before applying general numeric formatting.
    if (
        name in PERCENTAGE_COLUMNS
        or name.endswith(" pct")
        or name.endswith(" percentage")
        or name.endswith(" percent")
    ):
        return "percentage"

    # Keep correlation coefficients distinct from percentages and monetary values.
    if "correlation" in name:
        return "correlation"

    # Keep discount values at their original scale instead of assuming they are percentages.
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

    # Prefer an explicit currency suffix so display formatting preserves the dataset's stated unit.
    if currency is not None:
        return f"money:{currency}"

    # Use monetary formatting for recognized financial fields without explicit currency suffixes.
    if name in MONEY_COLUMNS:
        return "money"

    # Use count formatting for known quantity and record-count fields.
    if name in COUNT_COLUMNS:
        return "count"

    return "number"


# Format one finite value according to its display type while preserving its units and meaning.
def format_value(value, column_type):

    # Show missing values with the unavailable-value marker rather than formatting them as zero.
    if value is None or value is pd.NA:
        return "—"

    # Preserve booleans as text because Python treats them as numeric subclasses.
    if isinstance(value, bool):
        return str(value)

    # Leave strings and other nonnumeric values unchanged instead of coercing user data.
    if not isinstance(value, Real):
        return str(value)

    # Hide infinite and NaN results rather than presenting misleading numeric values.
    if not math.isfinite(value):
        return "—"

    # Keep identifier formatting separate so numeric-looking IDs retain their original representation.
    if column_type == "identifier":
        # Preserve whole-number identifiers as plain strings without numeric separators.
        if isinstance(value, Integral):
            return str(value)

        # Remove a redundant decimal portion when an identifier represents a whole number.
        if float(value).is_integer():
            return str(int(value))

        return str(value)

    # Format values using the explicit currency code inferred from the column name.
    if column_type.startswith("money:"):
        currency = column_type.split(":", 1)[1]

        return format_currency(
            value,
            currency,
        )

    # Retain the application's legacy USD display for monetary columns without a stated unit.
    if column_type == "money":
        return format_currency(
            value,
            "USD",
        )

    # Display percentage metrics with a percent sign and two decimal places.
    if column_type == "percentage":
        return f"{value:,.2f}%"

    # Display correlation coefficients to four decimal places without a percent sign.
    if column_type == "correlation":
        return f"{value:.4f}"

    # Preserve discount ratios to four decimal places without converting their scale.
    if column_type == "discount_value":
        return f"{value:,.4f}"

    # Format whole counts with grouping separators and retain decimals for fractional counts.
    if column_type == "count":
        # Format whole counts as integers with grouping separators when they have no fractional part.
        if float(value).is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    # Add grouping separators to whole numbers while omitting unnecessary decimal places.
    if isinstance(value, Integral):
        return f"{value:,}"

    return f"{value:,.2f}"


# Format a copy of the dataframe's values for display without modifying the input.
def format_business_table(df):

    timer = DebugTimer("TABLE FORMAT")

    debug_log(
        "TABLE FORMAT",
        f"Preparing business table | "
        f"Rows: {len(df):,} | "
        f"Columns: {len(df.columns):,}",
    )

    formatted = df.copy()

    # Format every copied dataframe column according to its inferred display type.
    for column in formatted.columns:
        column_type = get_column_type(column)

        debug_log(
            "TABLE FORMAT",
            f"Column: {column} | " f"Display type: {column_type}",
        )

        # Log the explicit unit used for this column before formatting its values.
        if column_type.startswith("money:"):
            currency = column_type.split(":", 1)[1]

            debug_log(
                "TABLE FORMAT",
                f"Explicit currency detected: {currency} | " f"Column: {column}",
            )

        formatted[column] = formatted[column].map(
            lambda value: format_value(
                value,
                column_type,
            )
        )

    timer.finish("Business table formatting completed")

    return formatted


# Convert a technical column name into a readable label with a percent sign where appropriate.
def clean_column_header(column):

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


# Format dataframe values, escape Markdown-sensitive text, and render a display-ready table.
def business_table_to_markdown(df):

    # Collapse cell whitespace and escape Markdown characters so values cannot alter table layout.
    def escape_cell(value):
        text = " ".join(str(value).split())

        return re.sub(
            r"([\\`*_{}\[\]()<>#+!|~])",
            r"\\\1",
            text,
        )

    formatted = format_business_table(df)

    formatted.columns = [
        escape_cell(clean_column_header(column)) for column in formatted.columns
    ]

    # Escape every formatted cell before rendering the Markdown table.
    for column in formatted.columns:
        formatted[column] = formatted[column].map(escape_cell)

    return formatted.to_markdown(
        index=False,
        disable_numparse=True,
    )
