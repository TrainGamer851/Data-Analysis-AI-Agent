import math
import re

from agent.debug_logger import debug_log
from agent.currency_units import (
    get_explicit_currency,
    format_currency,
)

MAX_TABLE_ROWS = 200


# Convert technical names into readable labels.
def readable_label(value):

    text = str(value).replace("_", " ")
    text = " ".join(text.split())

    text = re.sub(
        r"\b(?:pct|percent|percentage)\b",
        "%",
        text,
        flags=re.IGNORECASE,
    )

    return text[:1].upper() + text[1:]


# Prevent calculated labels from becoming Markdown markup.
def escape_text(value):

    text = " ".join(str(value).split())

    return re.sub(
        r"([\\`*_{}\[\]()<>#+!|~])",
        r"\\\1",
        text,
    )


# Check whether a value can be displayed directly.
def is_scalar(value):

    return value is None or isinstance(
        value,
        (str, int, float, bool),
    )


# Format a named metric without guessing its units.
def format_named_value(name, value):

    label = str(name).lower().replace("_", " ")
    label = " ".join(label.split())

    # Return `'Unavailable'` when `value is None`.
    if value is None:
        return "Unavailable"

    # Return the selected result when `isinstance(value, bool)`.
    if isinstance(value, bool):
        return str(value)

    # Evaluate the nested condition when `isinstance(value, (int, float))`.
    if isinstance(value, (int, float)):
        # Return `'Unavailable'` when `not math.isfinite(value)`.
        if not math.isfinite(value):
            return "Unavailable"

        # Return the selected result when `'correlation' in label`.
        if "correlation" in label:
            return f"{value:.4f}"

        # Return the selected result when `label.endswith((' pct', ' percent', ' percentage', '%'))`.
        if label.endswith((" pct", " percent", " percentage", "%")):
            return f"{value:,.2f}%"

        # Explicit currency units take priority over
        # traditional monetary metric names.
        currency = get_explicit_currency(name)

        # Call `debug_log` when `currency is not None`.
        if currency is not None:
            debug_log(
                "RESULT FORMAT",
                f"Explicit currency detected: {currency} | " f"Metric: {name}",
            )

            return format_currency(value, currency)

        # Call `debug_log` when `label in {'sales', 'total sales', 'average sales', 'profit', 'total profit', 'average...`.
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
            "expenses",
            "total expense",
            "total expenses",
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
            "unit cost",
            "price",
            "unit price",
            "gross profit",
            "net profit",
            "total gross profit",
            "total net profit",
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
            "budget",
            "marketing budget",
            "loss",
            "total loss",
            "average group sales",
            "average total sales",
            "average sales threshold",
            "avg sales threshold",
            "sales threshold",
            "aov",
        }:
            debug_log(
                "RESULT FORMAT",
                f"Recognized monetary metric: {name}",
            )

            return format_currency(value, "USD")

        # Return the selected result when `isinstance(value, int)`.
        if isinstance(value, int):
            return f"{value:,}"

        return str(value)

    return escape_text(value)


# Extract one unambiguous table from structured results.
def extract_record_table(value):

    debug_log(
        "RESULT FORMAT",
        "Checking for structured record tables",
    )

    # Update `candidates` when `isinstance(value, list)`.
    if isinstance(value, list):
        candidates = [("Results", value)]

    # Update `candidates` when `isinstance(value, dict)`.
    elif isinstance(value, dict):
        candidates = [
            (key, item)
            for key, item in value.items()
            if isinstance(item, list) and item
        ]

    # Return the selected result when no earlier condition matches.
    else:
        return None, None

    valid_tables = []

    # Process each `(name, records)` from `candidates`.
    for name, records in candidates:
        # Skip the current item when `not records`.
        if not records:
            continue

        # Skip the current item when `not all((isinstance(row, dict) and row for row in records))`.
        if not all(isinstance(row, dict) and row for row in records):
            continue

        columns = set(records[0])

        # Skip the current item when `not all((set(row) == columns and all((is_scalar(cell) for cell in row.values())) for...`.
        if not all(
            set(row) == columns and all(is_scalar(cell) for cell in row.values())
            for row in records
        ):
            continue

        valid_tables.append((name, records))

    # Call `debug_log` when `len(valid_tables) != 1`.
    if len(valid_tables) != 1:
        debug_log(
            "RESULT FORMAT",
            f"Unambiguous record tables: " f"{len(valid_tables)}",
        )

        return None, None

    name, records = valid_tables[0]

    debug_log(
        "RESULT FORMAT",
        f"Extracted table: {name} | " f"Rows: {len(records)}",
    )

    # Call `debug_log` when `len(records) > MAX_TABLE_ROWS`.
    if len(records) > MAX_TABLE_ROWS:
        debug_log(
            "RESULT FORMAT",
            f"Limiting displayed table to " f"{MAX_TABLE_ROWS} rows",
        )

    table = [dict(row) for row in records[:MAX_TABLE_ROWS]]

    return name, table


# Summarize structured results without another AI request.
def describe_record_table(value, table_name, table):

    lines = [
        f"**{escape_text(readable_label(table_name))}**",
        "",
        f"The analysis returned {len(table):,} " "row(s) of supporting data.",
    ]

    # Update `original_records` when `isinstance(value, dict)`.
    if isinstance(value, dict):
        original_records = value[table_name]

        # Process each `(key, item)` from `value.items()`.
        for key, item in value.items():
            # Skip the current item when `key == table_name`.
            if key == table_name:
                continue

            # Update `label` when `is_scalar(item)`.
            if is_scalar(item):
                label = escape_text(readable_label(key))
                formatted = format_named_value(key, item)

                lines.append(f"- **{label}:** {formatted}")

            # Append the value to `lines` when no earlier condition matches.
            else:
                lines.append(
                    f"- **{escape_text(readable_label(key))}:** " f"{escape_text(item)}"
                )

    # Update `original_records` when no earlier condition matches.
    else:
        original_records = value

    # Append the value to `lines` when `len(original_records) > len(table)`.
    if len(original_records) > len(table):
        lines.append(
            f"\nShowing {len(table):,} of "
            f"{len(original_records):,} calculated rows."
        )

    # Append the value to `lines` when `any(('correlation' in str(column).lower() for column in table[0]))`.
    if any("correlation" in str(column).lower() for column in table[0]):
        lines.append("\nCorrelation does not establish causation.")

    debug_log(
        "RESULT FORMAT",
        "Structured result summary prepared",
    )

    return "\n".join(lines)


# Normalize calculated column names for comparison.
def normalize_metric_name(name):

    return " ".join(str(name).lower().replace("_", " ").split())


# Accept only finite numeric calculation results.
def is_valid_number(value):

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# Present a supported scalar correlation calculation.
def summarize_scalar_correlation(question, value):

    debug_log(
        "RESULT FORMAT",
        "Checking for a scalar correlation result",
    )

    # Return `None` when `not isinstance(question, str)`.
    if not isinstance(question, str):
        return None

    # Call `debug_log` when `not is_valid_number(value)`.
    if not is_valid_number(value):
        debug_log(
            "STATISTICAL ACCURACY",
            "Scalar correlation result is not a finite number",
        )

        return None

    normalized = " ".join(question.lower().split())

    # Only recognize explicit correlation calculation
    # requests, rather than every mention of correlation.
    correlation_request = re.match(
        r"^(?:please\s+)?"
        r"(?:calculate|compute|find|determine|"
        r"what is|what's|give me|show me)\s+"
        r"(?:the\s+)?"
        r"(?:(pearson|spearman|kendall)\s+)?"
        r"correlation(?:\s+coefficient)?\b",
        normalized,
    )

    # Call `debug_log` when `correlation_request is None`.
    if correlation_request is None:
        debug_log(
            "RESULT FORMAT",
            "No supported scalar correlation intent detected",
        )

        return None

    # Avoid interpreting matrices, counts, significance tests and compound requests as single coefficients.
    if re.search(
        r"\b(matrix|matrices|p[- ]?value|"
        r"significance|significant|"
        r"count|number of|how many|"
        r"highest|lowest|strongest|weakest|"
        r"by region|by category|by department|"
        r"and also|as well as)\b",
        normalized,
    ):
        debug_log(
            "STATISTICAL ACCURACY",
            "Unsupported or compound correlation request",
        )

        return None

    # Supported correlation coefficients must be within the mathematical range of -1 to 1.
    if not -1 <= value <= 1:
        debug_log(
            "STATISTICAL ACCURACY",
            "Scalar correlation result outside valid range",
        )

        return None

    method = correlation_request.group(1)

    # Update `label` when `method is not None`.
    if method is not None:
        label = f"The {method.capitalize()} " "correlation coefficient"
    # Update `label` when no earlier condition matches.
    else:
        label = "The correlation coefficient"

    debug_log(
        "STATISTICAL ACCURACY",
        f"Scalar correlation detected | "
        f"Method: {method or 'unspecified'} | "
        f"Original coefficient: {value}",
    )

    formatted = format_named_value(
        "Correlation",
        value,
    )

    answer = (
        f"{label} is **{formatted}**." "\n\nCorrelation does not establish causation."
    )

    debug_log(
        "STATISTICAL ACCURACY",
        "Correlation disclaimer appended",
    )

    debug_log(
        "RESULT FORMAT",
        "Natural scalar correlation answer prepared",
    )

    return answer


# Extract supported correlation evidence without model prose.
def extract_single_correlation(question, value):

    debug_log(
        "STATISTICAL ACCURACY",
        "Checking result for single correlation evidence",
    )

    # Return `None` when `not isinstance(question, str)`.
    if not isinstance(question, str):
        return None

    # Evaluate the nested condition when `is_valid_number(value)`.
    if is_valid_number(value):
        # Return the selected result when `summarize_scalar_correlation(question, value) is not None`.
        if (
            summarize_scalar_correlation(
                question,
                value,
            )
            is not None
        ):
            return value

        return None

    # Return `None` when `not isinstance(value, dict) or not value`.
    if not isinstance(value, dict) or not value:
        return None

    narrative_fields = {
        "interpretation",
        "explanation",
        "recommendation",
        "recommendations",
        "conclusion",
        "business interpretation",
        "business implications",
        "summary",
        "insight",
        "insights",
    }

    candidates = []
    discarded_fields = 0

    # Process each `(name, item)` from `value.items()`.
    for name, item in value.items():
        # Return `None` when `not isinstance(name, str)`.
        if not isinstance(name, str):
            return None

        normalized = normalize_metric_name(name)

        # Update `discarded_fields` when `normalized in narrative_fields and isinstance(item, str)`.
        if normalized in narrative_fields and isinstance(item, str):
            discarded_fields += 1

            debug_log(
                "STATISTICAL ACCURACY",
                f"Discarding unverified narrative field: {name}",
            )

            continue

        # Call `debug_log` when `not re.search('\\bcorrelation\\b', normalized)`.
        if not re.search(
            r"\bcorrelation\b",
            normalized,
        ):
            debug_log(
                "STATISTICAL ACCURACY",
                "Additional result evidence detected; "
                "single-correlation extraction skipped",
            )

            return None

        # Return `None` when `re.search('\\b(count|matrix|significance|p value|p-value|threshold)\\b', normalized)`.
        if re.search(
            r"\b(count|matrix|significance|" r"p value|p-value|threshold)\b",
            normalized,
        ):
            return None

        # Return `None` when `not is_valid_number(item)`.
        if not is_valid_number(item):
            return None

        # Return `None` when `not -1 <= item <= 1`.
        if not -1 <= item <= 1:
            return None

        candidates.append((normalized, item))

    # Call `debug_log` when `len(candidates) != 1`.
    if len(candidates) != 1:
        debug_log(
            "STATISTICAL ACCURACY",
            f"Expected one correlation coefficient; " f"found {len(candidates)}",
        )

        return None

    name, coefficient = candidates[0]

    requested_method = re.search(
        r"\b(pearson|spearman|kendall)\b",
        question.lower(),
    )

    returned_methods = set(
        re.findall(
            r"\b(pearson|spearman|kendall)\b",
            name,
        )
    )

    # Call `debug_log` when `requested_method is not None and returned_methods and (returned_methods !=...`.
    if (
        requested_method is not None
        and returned_methods
        and returned_methods != {requested_method.group(1)}
    ):
        debug_log(
            "STATISTICAL ACCURACY",
            "Correlation method mismatch detected",
        )

        return None

    # Return `None` when `summarize_scalar_correlation(question, coefficient) is None`.
    if (
        summarize_scalar_correlation(
            question,
            coefficient,
        )
        is None
    ):
        return None

    debug_log(
        "STATISTICAL ACCURACY",
        f"Single correlation evidence extracted | "
        f"Coefficient: {coefficient} | "
        f"Discarded narrative fields: {discarded_fields}",
    )

    return coefficient


# Explain supported correlation results deterministically.
def summarize_correlation_interpretation(question, value):

    debug_log(
        "STATISTICAL ACCURACY",
        "Preparing deterministic correlation interpretation",
    )

    coefficient = extract_single_correlation(
        question,
        value,
    )

    # Return `None` when `coefficient is None`.
    if coefficient is None:
        return None

    base_answer = summarize_scalar_correlation(
        question,
        coefficient,
    )

    # Return `None` when `base_answer is None`.
    if base_answer is None:
        return None

    magnitude = abs(coefficient)

    # Update `strength` when `magnitude < 0.1`.
    if magnitude < 0.1:
        strength = "very weak"
    # Update `strength` when `magnitude < 0.3`.
    elif magnitude < 0.3:
        strength = "weak"
    # Update `strength` when `magnitude < 0.5`.
    elif magnitude < 0.5:
        strength = "moderate"
    # Update `strength` when `magnitude < 0.7`.
    elif magnitude < 0.7:
        strength = "strong"
    # Update `strength` when no earlier condition matches.
    else:
        strength = "very strong"

    # Update `direction` when `coefficient > 0`.
    if coefficient > 0:
        direction = "positive"
    # Update `direction` when `coefficient < 0`.
    elif coefficient < 0:
        direction = "negative"
    # Update `direction` when no earlier condition matches.
    else:
        direction = "zero"

    method_match = re.search(
        r"\b(pearson|spearman|kendall)\b",
        question.lower(),
    )

    method = method_match.group(1) if method_match is not None else "pearson"

    # Update `association_type` when `method == 'pearson'`.
    if method == "pearson":
        association_type = "linear association"

    # Update `association_type` when `method == 'spearman'`.
    elif method == "spearman":
        association_type = "monotonic association"

    # Update `association_type` when no earlier condition matches.
    else:
        association_type = "rank association"

    debug_log(
        "STATISTICAL ACCURACY",
        f"Method: {method} | " f"Strength: {strength} | " f"Direction: {direction}",
    )

    # Update `interpretation` when `coefficient == 0`.
    if coefficient == 0:
        interpretation = (
            f"The calculated coefficient indicates no "
            f"{association_type} according to this measure."
        )

    # Update `interpretation` when no earlier condition matches.
    else:
        interpretation = (
            f"This indicates a {strength} {direction} "
            f"{association_type} in the dataset."
        )

    limitation = (
        "The coefficient alone does not establish "
        "business impact or justify a change in strategy. "
        "Further analysis may be needed before making "
        "business decisions."
    )

    disclaimer = "Correlation does not establish causation."

    numeric_answer = base_answer.removesuffix("\n\n" + disclaimer)

    answer = (
        f"{numeric_answer}\n\n"
        f"{interpretation}\n\n"
        f"{limitation}\n\n"
        f"{disclaimer}"
    )

    debug_log(
        "STATISTICAL ACCURACY",
        "Deterministic statistical interpretation prepared",
    )

    debug_log(
        "RESULT INTEGRITY",
        "Original coefficient preserved; " "unverified causal claims excluded",
    )

    return answer


# Present one named calculation as a natural sentence.
def summarize_single_metric(value):

    debug_log(
        "RESULT FORMAT",
        "Checking for a single named metric",
    )

    # Return `None` when `not isinstance(value, dict) or len(value) != 1`.
    if not isinstance(value, dict) or len(value) != 1:
        return None

    name, number = next(iter(value.items()))

    # Return `None` when `not is_valid_number(number)`.
    if not is_valid_number(number):
        return None

    label = escape_text(readable_label(name))

    formatted = format_named_value(
        name,
        number,
    )

    answer = f"{label} is **{formatted}**."

    # Update `answer` when `'correlation' in normalize_metric_name(name)`.
    if "correlation" in normalize_metric_name(name):
        answer += "\n\nCorrelation does not establish causation."

        debug_log(
            "RESULT FORMAT",
            "Correlation disclaimer appended",
        )

    debug_log(
        "RESULT FORMAT",
        "Natural single-metric answer prepared",
    )

    return answer


# Normalize harmless presentation differences.
def normalize_explanation_text(text):

    text = text.replace("**", "")
    text = text.replace("−", "-")

    return " ".join(text.split()).casefold()


# Return a readable metric label without a currency suffix.
def get_metric_display_label(name):

    label = readable_label(name)

    currency = get_explicit_currency(name)

    # Update `label` when `currency is not None`.
    if currency is not None:
        label = re.sub(
            rf"\s+{re.escape(currency)}$",
            "",
            label,
            flags=re.IGNORECASE,
        )

    return label


# Build a deterministic summary of independent numeric metrics.
def summarize_multiple_metrics(value):

    debug_log(
        "RESULT FORMAT",
        "Checking for multiple named metrics",
    )

    # Return `None` when `not isinstance(value, dict)`.
    if not isinstance(value, dict):
        return None

    # Return `None` when `not 2 <= len(value) <= 10`.
    if not 2 <= len(value) <= 10:
        return None

    # Return `None` when `not all((isinstance(name, str) and name.strip() and is_valid_number(number) for name,...`.
    if not all(
        isinstance(name, str) and name.strip() and is_valid_number(number)
        for name, number in value.items()
    ):
        return None

    # Special handling for a valid subtotal and excluded rows.
    excluded_keys = [
        name
        for name in value
        if normalize_metric_name(name)
        in {
            "excluded rows",
            "excluded rows count",
            "excluded row count",
        }
    ]

    # Update `excluded_key` when `len(value) == 2 and len(excluded_keys) == 1`.
    if len(value) == 2 and len(excluded_keys) == 1:
        excluded_key = excluded_keys[0]
        excluded_count = value[excluded_key]

        metric_name = next(name for name in value if name != excluded_key)

        metric_label = get_metric_display_label(metric_name)

        # Update `formatted` when `normalize_metric_name(metric_label) in {'valid revenue', 'total valid revenue', 'valid...`.
        if (
            normalize_metric_name(metric_label)
            in {
                "valid revenue",
                "total valid revenue",
                "valid finite revenue",
                "total valid finite revenue",
            }
            and excluded_count >= 0
            and excluded_count == int(excluded_count)
        ):
            formatted = format_named_value(
                metric_name,
                value[metric_name],
            )

            row_label = "row" if excluded_count == 1 else "rows"

            answer = (
                f"The {metric_label.lower()} is "
                f"**{formatted}**, excluding "
                f"**{int(excluded_count):,} {row_label}**."
            )

            debug_log(
                "RESULT FORMAT",
                "Valid subtotal and exclusion summary prepared",
            )

            return answer

    # Preserve every independent calculated metric.
    lines = []

    # Process each `(name, number)` from `value.items()`.
    for name, number in value.items():
        label = escape_text(readable_label(name))

        formatted = format_named_value(
            name,
            number,
        )

        lines.append(f"{label} is **{formatted}**.")

    # Append the value to `lines` when `any(('correlation' in normalize_metric_name(name) for name in value))`.
    if any("correlation" in normalize_metric_name(name) for name in value):
        lines.append("Correlation does not establish causation.")

    debug_log(
        "RESULT FORMAT",
        f"Multi-metric summary prepared | " f"Metrics: {len(value)}",
    )

    return "\n\n".join(lines)


# Accept only supported summaries of calculated metrics.
def validate_multiple_metric_explanation(text, value):

    debug_log(
        "RESULT INTEGRITY",
        "Validating multi-metric explanation",
    )

    # Return `False` when `not isinstance(text, str) or not text.strip()`.
    if not isinstance(text, str) or not text.strip():
        return False

    # Return `False` when `len(text) > 5000`.
    if len(text) > 5000:
        return False

    expected = summarize_multiple_metrics(value)

    # Call `debug_log` when `expected is None`.
    if expected is None:
        debug_log(
            "RESULT INTEGRITY",
            "Unsupported multi-metric result structure",
        )

        return False

    # Call `debug_log` when `normalize_explanation_text(text) != normalize_explanation_text(expected)`.
    if normalize_explanation_text(text) != normalize_explanation_text(expected):
        debug_log(
            "RESULT INTEGRITY",
            "Multi-metric explanation differs from " "supported calculated summary",
        )

        return False

    debug_log(
        "RESULT INTEGRITY",
        "Multi-metric explanation validation passed",
    )

    return True


# Check the number and known unit in a simple explanation.
def validate_single_metric_explanation(text, value):

    debug_log(
        "RESULT INTEGRITY",
        "Validating single-metric explanation",
    )

    # Return `False` when `not isinstance(text, str) or not text.strip()`.
    if not isinstance(text, str) or not text.strip():
        return False

    metric_name = None

    # Evaluate the nested condition when `isinstance(value, dict)`.
    if isinstance(value, dict):
        # Return `False` when `len(value) != 1`.
        if len(value) != 1:
            return False

        metric_name, value = next(iter(value.items()))

    # Return `False` when `not is_valid_number(value)`.
    if not is_valid_number(value):
        return False

    # Return `False` when `len(text) > 500 or '\n' in text.strip()`.
    if len(text) > 500 or "\n" in text.strip():
        return False

    # Explicit currency metrics require the correct metric association, displayed value and currency.
    if metric_name is not None:
        currency = get_explicit_currency(metric_name)

        # Call `debug_log` when `currency is not None`.
        if currency is not None:
            debug_log(
                "RESULT INTEGRITY",
                f"Validating explicit currency: {currency}",
            )

            label = get_metric_display_label(metric_name)

            full_label = readable_label(metric_name)

            formatted = format_named_value(
                metric_name,
                value,
            )

            candidates = {
                f"{label} is {formatted}.",
                f"The {label} is {formatted}.",
                f"{full_label} is {formatted}.",
                f"The {full_label} is {formatted}.",
            }

            normalized = normalize_explanation_text(text)

            accepted = any(
                normalized == normalize_explanation_text(candidate)
                for candidate in candidates
            )

            debug_log(
                "RESULT INTEGRITY",
                (
                    "Explicit currency validation passed"
                    if accepted
                    else "Explicit currency validation failed"
                ),
            )

            return accepted

    cleaned = text.replace("**", "")
    cleaned = cleaned.replace("−", "-")

    tokens = re.findall(
        r"(?<!\w)(-?\s*\$?\s*\d[\d,]*" r"(?:\.\d+)?\s*%?)",
        cleaned,
    )

    # Call `debug_log` when `len(tokens) != 1`.
    if len(tokens) != 1:
        debug_log(
            "RESULT INTEGRITY",
            "Explanation contains missing or additional numbers",
        )

        return False

    token = tokens[0].strip()

    try:
        number = float(
            token.replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
        )

    except (ValueError, OverflowError):
        return False

    # Call `debug_log` when `not math.isclose(number, float(value), rel_tol=0, abs_tol=1e-09)`.
    if not math.isclose(
        number,
        float(value),
        rel_tol=0,
        abs_tol=0.000000001,
    ):
        debug_log(
            "RESULT INTEGRITY",
            "Explanation number differs from calculation",
        )

        return False

    # Update `expected` when `metric_name is not None`.
    if metric_name is not None:
        expected = format_named_value(
            metric_name,
            value,
        )

        # Call `debug_log` when `expected.endswith('%') != token.endswith('%')`.
        if expected.endswith("%") != token.endswith("%"):
            debug_log(
                "RESULT INTEGRITY",
                "Percentage unit mismatch",
            )

            return False

        # Call `debug_log` when `('$' in expected) != ('$' in token)`.
        if ("$" in expected) != ("$" in token):
            debug_log(
                "RESULT INTEGRITY",
                "Currency unit mismatch",
            )

            return False

    debug_log(
        "RESULT INTEGRITY",
        "Single-metric numerical validation passed",
    )

    return True


# Present a single calculated row using its actual values.
def summarize_single_row(records):

    debug_log(
        "DETERMINISTIC SUMMARY",
        "Checking for a single-row result",
    )

    # Return `None` when `not isinstance(records, list) or len(records) != 1 or (not isinstance(records[0],...`.
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], dict)
        or not records[0]
    ):
        return None

    row = records[0]

    # Return `None` when `not all((is_scalar(value) for value in row.values()))`.
    if not all(is_scalar(value) for value in row.values()):
        return None

    lines = []

    # Process each `(column, value)` from `row.items()`.
    for column, value in row.items():
        label = escape_text(readable_label(column))

        formatted = format_named_value(
            column,
            value,
        )

        lines.append(f"**{label}:** {formatted}")

    # Append the value to `lines` when `any(('correlation' in str(column).lower() for column in row))`.
    if any("correlation" in str(column).lower() for column in row):
        lines.append("\nCorrelation does not establish causation.")

    debug_log(
        "DETERMINISTIC SUMMARY",
        "Single-row calculated values prepared",
    )

    return "\n\n".join(lines)


# Summarize supported financial tables without an LLM.
def summarize_financial_table(records):

    debug_log(
        "DETERMINISTIC SUMMARY",
        "Inspecting calculated financial table",
    )

    # Call `debug_log` when `not isinstance(records, list) or not records`.
    if not isinstance(records, list) or not records:
        debug_log(
            "DETERMINISTIC SUMMARY",
            "No non-empty table available",
        )

        return None

    # Return `None` when `not all((isinstance(row, dict) and row for row in records))`.
    if not all(isinstance(row, dict) and row for row in records):
        return None

    columns = list(records[0])

    # Return `None` when `not all((set(row) == set(columns) for row in records))`.
    if not all(set(row) == set(columns) for row in records):
        return None

    column_map = {normalize_metric_name(column): column for column in columns}

    # Return `None` when `len(column_map) != len(columns)`.
    if len(column_map) != len(columns):
        return None

    dimension_columns = [
        column
        for column in columns
        if all(isinstance(row[column], str) and row[column].strip() for row in records)
    ]

    # Call `debug_log` when `len(dimension_columns) != 1`.
    if len(dimension_columns) != 1:
        debug_log(
            "DETERMINISTIC SUMMARY",
            "No unambiguous grouping column",
        )

        return None

    dimension = dimension_columns[0]

    # Return `None` when `len({row[dimension] for row in records}) != len(records)`.
    if len({row[dimension] for row in records}) != len(records):
        return None

    # Find metric.
    def find_metric(*names):
        # Process each `name` from `names`.
        for name in names:
            # Return the selected result when `name in column_map`.
            if name in column_map:
                return column_map[name]

        return None

    # Check whether a metric name is supported for the requested calculation.
    def valid_metric(column):
        return column is not None and all(
            is_valid_number(row[column]) for row in records
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

    # Normalize a group label for use in the formatted result.
    def group_name(row):
        return f"**{escape_text(row[dimension])}**"

    # Format metric.
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
        and all(-1 <= row[correlation_column] <= 1 for row in records)
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

        # Append the value to `lines` when `strongest_negative[correlation_column] < 0`.
        if strongest_negative[correlation_column] < 0:
            lines.append(
                f"- {group_name(strongest_negative)} has "
                "the most negative discount-profit "
                "correlation shown "
                f"(**{format_metric(strongest_negative, correlation_column)}**)."
            )

        # Append the value to `lines` when `all((row[correlation_column] < 0 for row in records))`.
        if all(row[correlation_column] < 0 for row in records):
            lines.append(
                "\nAll returned groups show a negative "
                "association between discount and profit."
            )

        lines.append("\n**Note:** Correlation does not " "establish causation.")

        debug_log(
            "DETERMINISTIC SUMMARY",
            "Discount-profit relationship summarized",
        )

        return "\n".join(lines)

    # Case 2: Financial results containing one group.
    if len(records) == 1 and valid_metric(sales_column) and valid_metric(profit_column):
        row = records[0]

        lines = [
            f"The calculated result identifies " f"{group_name(row)}.",
            "",
            f"- **Total sales/revenue:** " f"{format_metric(row, sales_column)}",
            f"- **Total profit:** " f"{format_metric(row, profit_column)}",
        ]

        # Append the value to `lines` when `valid_metric(margin_column)`.
        if valid_metric(margin_column):
            lines.append(
                f"- **Profit margin:** " f"{format_metric(row, margin_column)}"
            )

        threshold_column = find_metric(
            "average group sales",
            "average total sales",
            "average sales threshold",
            "avg sales threshold",
            "sales threshold",
        )

        # Append the value to `lines` when `valid_metric(threshold_column)`.
        if valid_metric(threshold_column):
            lines.append(
                f"- **Sales comparison threshold:** " f"${row[threshold_column]:,.2f}"
            )

        # Append the value to `lines` when `row[profit_column] < 0`.
        if row[profit_column] < 0:
            lines.append("\nThe returned group recorded " "a negative total profit.")

        debug_log(
            "DETERMINISTIC SUMMARY",
            "Single-group financial result summarized",
        )

        return "\n".join(lines)

    # Case 3: Average discount comparison.
    if valid_metric(discount_column) and len(records) >= 2 and len(columns) == 2:
        highest = max(row[discount_column] for row in records)

        lowest = min(row[discount_column] for row in records)

        highest_names = ", ".join(
            group_name(row) for row in records if row[discount_column] == highest
        )

        lowest_names = ", ".join(
            group_name(row) for row in records if row[discount_column] == lowest
        )

        # Update `answer` when `highest == lowest`.
        if highest == lowest:
            answer = (
                "All returned groups have the same "
                "average discount of "
                f"**{highest:.2f}%**."
            )

        # Update `answer` when no earlier condition matches.
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
