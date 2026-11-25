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