import re

from agent.dataset_schema import normalize_column_name
from agent.debug_logger import debug_log, DebugTimer


DATE_CONCEPTS = {
    "transaction_date",
    "order_date",
    "shipping_date",
}


DATE_WARNINGS = {
    "ambiguous_date_convention",
    "conflicting_date_conventions",
    "mixed_or_invalid_date_format",
    "unrecognized_date_format",
}


DISCOUNT_WARNINGS = {
    "potentially_mixed_discount_scale",
    "mixed_percentage_notation",
    "discount_ratio_above_one",
}


AGGREGATION_PATTERN = (
    r"(?:calculate|compute|find|show|what is|what are|"
    r"give me|determine)?\s*"
    r"(?:the\s+)?"
    r"(?:overall\s+)?"
    r"(?:total|sum of|average|mean)\s+"
    r"(?:the\s+)?"
)


def normalize_question(question):
    """Normalize question wording for conservative matching."""

    return normalize_column_name(question)


def make_response(answer):
    """Return a normal analysis response, not an execution error."""

    return {
        "answer": answer,
        "table": None,
        "error": None,
    }


def explicitly_allows_exclusions(question):
    """Identify explicit requests to calculate valid values only."""

    text = normalize_question(question)

    patterns = (
        r"\bonly (?:the )?valid\b",
        r"\bvalid finite\b",
        r"\bvalid values only\b",
        r"\bexclude (?:the )?invalid\b",
        r"\bignore (?:the )?invalid\b",
        r"\bexclude (?:the )?missing\b",
        r"\bignore (?:the )?missing\b",
    )

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def explicit_date_convention(question):
    """Recognize an explicitly supplied date convention."""

    text = normalize_question(question)

    month_first = bool(
        re.search(
            r"\b(?:mm dd yyyy|mm dd yy|"
            r"month first|month day year)\b",
            text,
        )
    )

    day_first = bool(
        re.search(
            r"\b(?:dd mm yyyy|dd mm yy|"
            r"day first|day month year)\b",
            text,
        )
    )

    if month_first == day_first:
        return None

    return (
        "month_first"
        if month_first
        else "day_first"
    )


def explicitly_defines_discount_scale(question):
    """Recognize an explicit mixed-discount conversion rule."""

    text = normalize_question(question)

    fractional_rule = bool(
        re.search(
            r"\b(?:fractional values|fractions|"
            r"values below 1|values less than 1)\b",
            text,
        )
    )

    percentage_rule = bool(
        re.search(
            r"\b(?:percentage points|percentages|"
            r"values above 1|values greater than 1)\b",
            text,
        )
    )

    conversion_rule = bool(
        re.search(
            r"\b(?:multiply|convert|treat|interpret|"
            r"represent|assume)\b",
            text,
        )
    )

    return (
        fractional_rule
        and percentage_rule
        and conversion_rule
    )


def resolve_simple_aggregation(question, schema):
    """
    Resolve a simple whole-column aggregation.

    Complex filters, comparisons and unfamiliar expressions
    are intentionally left to the general AI agent.
    """

    text = normalize_question(question)

    match = re.fullmatch(
        AGGREGATION_PATTERN + r"(.+)",
        text,
    )

    if not match:
        return None

    target = match.group(1)

    # Remove a limited set of harmless trailing qualifiers.
    target = re.sub(
        r"\s+(?:column|values|regardless of date)$",
        "",
        target,
    )

    target = re.sub(
        r"\s+in\s+(usd|eur|gbp|cad|aud|jpy|inr|cny)$",
        r" \1",
        target,
    )

    columns = schema.get("columns", [])

    # Prefer an explicitly identified column.
    exact = [
        column
        for column in columns
        if column["normalized_name"] == target
    ]

    if len(exact) == 1:
        return exact[0]

    # Resolve a business concept only when it has one candidate.
    from agent.dataset_schema import identify_business_concepts

    concepts = identify_business_concepts(target)

    if len(concepts) != 1:
        return None

    candidates = [
        column
        for column in columns
        if concepts[0] in column.get(
            "business_concepts", []
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    return None


def check_numeric_aggregation(question, schema):
    """Prevent incomplete whole-column numeric totals."""

    column = resolve_simple_aggregation(
        question,
        schema,
    )

    if column is None:
        return None

    quality = column.get("data_quality", {})

    name = column["name"]

    missing_count = column.get("missing_count", 0)

    invalid_count = quality.get(
        "invalid_numeric_count", 0
    )

    valid_count = quality.get("numeric_count", 0)

    compatibility = quality.get(
        "numeric_compatibility"
    )

    debug_log(
        "INTEGRITY",
        f"Resolved aggregation column: {name} | "
        f"Valid: {valid_count} | "
        f"Missing: {missing_count} | "
        f"Invalid: {invalid_count}",
    )

    # Do not guess how arbitrary text or identifiers
    # should be aggregated.
    if compatibility in {
        "boolean",
        "datetime",
        "not_numeric",
        "unsupported_numeric",
        "empty",
    }:
        return make_response(
            f"Cannot answer: '{name}' does not contain "
            "usable numeric values for this calculation."
        )

    excluded_count = missing_count + invalid_count

    if excluded_count == 0:
        return None

    if (
        explicitly_allows_exclusions(question)
        and valid_count > 0
    ):
        debug_log(
            "INTEGRITY",
            "Explicit valid-values-only request detected",
        )

        return None

    return make_response(
        f"Cannot calculate a complete result for '{name}' "
        f"because {excluded_count:,} row(s) contain "
        f"missing or invalid numeric values "
        f"({missing_count:,} missing, "
        f"{invalid_count:,} invalid). "
        f"There are {valid_count:,} valid numeric row(s). "
        "Would you like me to calculate the result using "
        "only the valid values?"
    )


def check_date_integrity(question, schema):
    """Check date ambiguity for clearly time-based analysis."""

    text = normalize_question(question)

    # Do not apply date checks to ordinary totals.
    if not re.search(
        r"\b(?:daily|weekly|monthly|quarterly|yearly|"
        r"annual|by day|by week|by month|by quarter|"
        r"by year|year over year|month over month)\b",
        text,
    ):
        return None

    date_columns = [
        column
        for column in schema.get("columns", [])
        if DATE_CONCEPTS.intersection(
            column.get("business_concepts", [])
        )
    ]

    # Prefer a date column explicitly named in the question.
    named_columns = [
        column
        for column in date_columns
        if re.search(
            r"\b"
            + re.escape(column["normalized_name"])
            + r"\b",
            text,
        )
    ]

    if len(named_columns) == 1:
        date_columns = named_columns

    # Avoid guessing which date column the question requires.
    if len(date_columns) != 1:
        return None

    column = date_columns[0]

    warnings = set(
        column.get("data_quality", {}).get(
            "warnings", []
        )
    )

    relevant = warnings.intersection(DATE_WARNINGS)

    if not relevant:
        return None

    name = column["name"]

    debug_log(
        "INTEGRITY",
        f"Relevant date column: {name} | "
        f"Warnings: {sorted(relevant)}",
    )

    if "conflicting_date_conventions" in relevant:
        return make_response(
            f"Cannot calculate reliable time-based results "
            f"because '{name}' contains conflicting date "
            "conventions. Please correct the dates or "
            "provide a rule for interpreting the mixed formats."
        )

    if "ambiguous_date_convention" in relevant:
        if explicit_date_convention(question):
            debug_log(
                "INTEGRITY",
                "Explicit date convention supplied",
            )

            return None

        return make_response(
            f"The dates in '{name}' could use either "
            "MM/DD/YYYY or DD/MM/YYYY. "
            "Which date format should I use?"
        )

    return make_response(
        f"Cannot reliably group results by date because "
        f"'{name}' contains mixed, invalid or unrecognized "
        "date formats. Please clarify or correct the dates."
    )


def check_discount_integrity(question, schema):
    """Prevent unsupported discount-scale conversions."""

    text = normalize_question(question)

    if not re.search(
        r"\bdiscounts?\b",
        text,
    ):
        return None

    # Limit this check to calculations involving discount values.
    if not re.search(
        r"\b(?:average|mean|total|percentage|rate|"
        r"correlation|relationship|compare|highest|"
        r"lowest|margin)\b",
        text,
    ):
        return None

    candidates = [
        column
        for column in schema.get("columns", [])
        if "discount" in column.get(
            "business_concepts", []
        )
    ]

    if len(candidates) != 1:
        return None

    column = candidates[0]

    warnings = set(
        column.get("data_quality", {}).get(
            "warnings", []
        )
    )

    relevant = warnings.intersection(
        DISCOUNT_WARNINGS
    )

    if not relevant:
        return None

    name = column["name"]

    debug_log(
        "INTEGRITY",
        f"Relevant discount column: {name} | "
        f"Warnings: {sorted(relevant)}",
    )

    if explicitly_defines_discount_scale(question):
        debug_log(
            "INTEGRITY",
            "Explicit discount conversion rule supplied",
        )

        return None

    return make_response(
        f"'{name}' contains potentially mixed or "
        "inconsistent discount units. "
        "Should fractional values such as 0.15 be "
        "converted to 15%, while values such as 20 "
        "remain 20%, or should all values use another scale?"
    )


def validate_analysis_integrity(question, schema):
    """
    Return a clarification response when a relevant
    data-quality issue prevents a reliable calculation.

    Return None when analysis may continue.
    """

    timer = DebugTimer("INTEGRITY")

    debug_log(
        "INTEGRITY",
        "Starting question-aware data-integrity validation",
    )

    if not isinstance(schema, dict):
        timer.finish(
            "Validation skipped: dataset profile unavailable"
        )

        return None

    checks = (
        check_date_integrity,
        check_discount_integrity,
        check_numeric_aggregation,
    )

    for check in checks:
        debug_log(
            "INTEGRITY",
            f"Running check: {check.__name__}",
        )

        response = check(question, schema)

        if response is not None:
            debug_log(
                "INTEGRITY",
                f"Clarification required by {check.__name__}",
            )

            timer.finish(
                "Analysis paused for data-integrity clarification"
            )

            return response

    timer.finish(
        "Data-integrity checks completed; analysis may continue"
    )

    return None
