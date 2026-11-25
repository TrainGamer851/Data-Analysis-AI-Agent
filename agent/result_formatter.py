"""Deterministic formatting of calculated analysis results."""

import math
import re

from agent.debug_logger import debug_log


MAX_TABLE_ROWS = 200


def readable_label(value):
    """Convert technical names into readable labels."""

    text = str(value).replace("_", " ")
    text = " ".join(text.split())

    text = re.sub(
        r"\b(?:pct|percent|percentage)\b",
        "%",
        text,
        flags=re.IGNORECASE,
    )

    return text[:1].upper() + text[1:]


def escape_text(value):
    """Prevent calculated labels from becoming Markdown markup."""

    text = " ".join(str(value).split())

    return re.sub(
        r"([\\`*_{}\[\]()<>#+!|~])",
        r"\\\1",
        text,
    )


def is_scalar(value):
    """Check whether a value can be displayed directly."""

    return value is None or isinstance(
        value,
        (str, int, float, bool),
    )


def format_named_value(name, value):
    """Format a named metric without guessing its units."""

    label = str(name).lower().replace("_", " ")
    label = " ".join(label.split())

    if value is None:
        return "Unavailable"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "Unavailable"

        if "correlation" in label:
            return f"{value:.4f}"

        if label.endswith(
            (" pct", " percent", " percentage", "%")
        ):
            return f"{value:,.2f}%"

        if label in {
            "sales",
            "total sales",
            "average sales",
            "profit",
            "total profit",
            "average profit",
            "revenue",
            "total revenue",
            "cost",
            "total cost",
            "average order value",
        }:
            if value < 0:
                return f"-${abs(value):,.2f}"

            return f"${value:,.2f}"

        if isinstance(value, int):
            return f"{value:,}"

        return str(value)

    return escape_text(value)


def extract_record_table(value):
    """Extract one unambiguous table from structured results."""

    debug_log(
        "RESULT FORMAT",
        "Checking for structured record tables",
    )

    if isinstance(value, list):
        candidates = [("Results", value)]

    elif isinstance(value, dict):
        candidates = [
            (key, item)
            for key, item in value.items()
            if isinstance(item, list) and item
        ]

    else:
        return None, None

    valid_tables = []

    for name, records in candidates:
        if not records:
            continue

        if not all(
            isinstance(row, dict) and row
            for row in records
        ):
            continue

        columns = set(records[0])

        if not all(
            set(row) == columns
            and all(
                is_scalar(cell)
                for cell in row.values()
            )
            for row in records
        ):
            continue

        valid_tables.append((name, records))

    if len(valid_tables) != 1:
        debug_log(
            "RESULT FORMAT",
            f"Unambiguous record tables: "
            f"{len(valid_tables)}",
        )

        return None, None

    name, records = valid_tables[0]

    debug_log(
        "RESULT FORMAT",
        f"Extracted table: {name} | "
        f"Rows: {len(records)}",
    )

    if len(records) > MAX_TABLE_ROWS:
        debug_log(
            "RESULT FORMAT",
            f"Limiting displayed table to "
            f"{MAX_TABLE_ROWS} rows",
        )

    table = [
        dict(row)
        for row in records[:MAX_TABLE_ROWS]
    ]

    return name, table


def describe_record_table(value, table_name, table):
    """Summarize structured results without another AI request."""

    lines = [
        f"**{escape_text(readable_label(table_name))}**",
        "",
        f"The analysis returned {len(table):,} "
        "row(s) of supporting data.",
    ]

    if isinstance(value, dict):
        original_records = value[table_name]

        for key, item in value.items():
            if key == table_name:
                continue

            if is_scalar(item):
                label = escape_text(readable_label(key))
                formatted = format_named_value(key, item)

                lines.append(
                    f"- **{label}:** {formatted}"
                )

            else:
                lines.append(
                    f"- **{escape_text(readable_label(key))}:** "
                    f"{escape_text(item)}"
                )

    else:
        original_records = value

    if len(original_records) > len(table):
        lines.append(
            f"\nShowing {len(table):,} of "
            f"{len(original_records):,} calculated rows."
        )

    if any(
        "correlation" in str(column).lower()
        for column in table[0]
    ):
        lines.append(
            "\nCorrelation does not establish causation."
        )

    debug_log(
        "RESULT FORMAT",
        "Structured result summary prepared",
    )

    return "\n".join(lines)



def normalize_metric_name(name):
    """Normalize calculated column names for comparison."""

    return " ".join(
        str(name).lower().replace("_", " ").split()
    )


def is_valid_number(value):
    """Accept only finite numeric calculation results."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def summarize_financial_table(records):
    """Summarize supported financial tables without an LLM."""

    debug_log(
        "DETERMINISTIC SUMMARY",
        "Inspecting calculated financial table",
    )

    if not isinstance(records, list) or not records:
        debug_log(
            "DETERMINISTIC SUMMARY",
            "No non-empty table available",
        )

        return None

    if not all(
        isinstance(row, dict) and row
        for row in records
    ):
        return None

    columns = list(records[0])

    if not all(
        set(row) == set(columns)
        for row in records
    ):
        return None

    column_map = {
        normalize_metric_name(column): column
        for column in columns
    }

    if len(column_map) != len(columns):
        return None

    dimension_columns = [
        column
        for column in columns
        if all(
            isinstance(row[column], str)
            and row[column].strip()
            for row in records
        )
    ]

    if len(dimension_columns) != 1:
        debug_log(
            "DETERMINISTIC SUMMARY",
            "No unambiguous grouping column",
        )

        return None

    dimension = dimension_columns[0]

    if len({
        row[dimension] for row in records
    }) != len(records):
        return None

    def find_metric(*names):
        for name in names:
            if name in column_map:
                return column_map[name]

        return None

    def valid_metric(column):
        return (
            column is not None
            and all(
                is_valid_number(row[column])
                for row in records
            )
        )

    sales_column = find_metric(
        "total sales",
        "sales",
        "total revenue",
        "revenue",
    )

    profit_column = find_metric(
        "total profit",
        "profit",
    )

    margin_column = find_metric(
        "profit margin pct",
        "profit margin %",
        "profit margin percentage",
    )

    discount_column = find_metric(
        "average discount pct",
        "avg discount pct",
        "average discount %",
        "average discount percentage",
    )

    correlation_column = find_metric(
        "discount profit correlation",
    )

    def group_name(row):
        return (
            f"**{escape_text(row[dimension])}**"
        )

    def format_metric(row, column):
        return format_named_value(
            column,
            row[column],
        )

    # Case 1: Discount and profitability relationship.
    if (
        valid_metric(discount_column)
        and valid_metric(margin_column)
        and valid_metric(correlation_column)
        and len(records) >= 2
        and all(
            -1 <= row[correlation_column] <= 1
            for row in records
        )
    ):
        highest_discount = max(
            records,
            key=lambda row: row[discount_column],
        )

        lowest_margin = min(
            records,
            key=lambda row: row[margin_column],
        )

        strongest_negative = min(
            records,
            key=lambda row: row[correlation_column],
        )

        lines = [
            "Among the returned groups:",
            "",
            f"- {group_name(highest_discount)} has "
            "an average discount of "
            f"**{format_metric(highest_discount, discount_column)}**, "
            "the highest value shown.",
            f"- {group_name(lowest_margin)} has "
            "a profit margin of "
            f"**{format_metric(lowest_margin, margin_column)}**, "
            "the lowest value shown.",
        ]

        if strongest_negative[correlation_column] < 0:
            lines.append(
                f"- {group_name(strongest_negative)} has "
                "the most negative discount-profit "
                "correlation shown "
                f"(**{format_metric(strongest_negative, correlation_column)}**)."
            )

        if all(
            row[correlation_column] < 0
            for row in records
        ):
            lines.append(
                "\nAll returned groups show a negative "
                "association between discount and profit."
            )

        lines.append(
            "\n**Note:** Correlation does not "
            "establish causation."
        )

        debug_log(
            "DETERMINISTIC SUMMARY",
            "Discount-profit relationship summarized",
        )

        return "\n".join(lines)

    # Case 2: Financial results containing one group.
    if (
        len(records) == 1
        and valid_metric(sales_column)
        and valid_metric(profit_column)
    ):
        row = records[0]

        lines = [
            f"The calculated result identifies "
            f"{group_name(row)}.",
            "",
            f"- **Total sales/revenue:** "
            f"{format_metric(row, sales_column)}",
            f"- **Total profit:** "
            f"{format_metric(row, profit_column)}",
        ]

        if valid_metric(margin_column):
            lines.append(
                f"- **Profit margin:** "
                f"{format_metric(row, margin_column)}"
            )

        threshold_column = find_metric(
            "average group sales",
            "average total sales",
            "average sales threshold",
            "sales threshold",
        )

        if valid_metric(threshold_column):
            lines.append(
                f"- **Sales comparison threshold:** "
                f"${row[threshold_column]:,.2f}"
            )

        if row[profit_column] < 0:
            lines.append(
                "\nThe returned group recorded "
                "a negative total profit."
            )

        debug_log(
            "DETERMINISTIC SUMMARY",
            "Single-group financial result summarized",
        )

        return "\n".join(lines)

    # Case 3: Average discount comparison.
    if (
        valid_metric(discount_column)
        and len(records) >= 2
        and len(columns) == 2
    ):
        highest = max(
            row[discount_column]
            for row in records
        )

        lowest = min(
            row[discount_column]
            for row in records
        )

        highest_names = ", ".join(
            group_name(row)
            for row in records
            if row[discount_column] == highest
        )

        lowest_names = ", ".join(
            group_name(row)
            for row in records
            if row[discount_column] == lowest
        )

        if highest == lowest:
            answer = (
                "All returned groups have the same "
                "average discount of "
                f"**{highest:.2f}%**."
            )

        else:
            answer = (
                f"{highest_names} recorded the highest "
                "average discount among the returned "
                f"groups (**{highest:.2f}%**), while "
                f"{lowest_names} recorded the lowest "
                f"(**{lowest:.2f}%**)."
            )

        debug_log(
            "DETERMINISTIC SUMMARY",
            "Average discount comparison summarized",
        )

        return answer

    debug_log(
        "DETERMINISTIC SUMMARY",
        "No supported financial summary pattern",
    )

    return None