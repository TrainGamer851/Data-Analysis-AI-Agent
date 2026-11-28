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


# Normalize question wording for conservative matching.
def normalize_question(question):

    return normalize_column_name(question)


# Return a normal analysis response, not an execution error.
def make_response(answer):

    return {
        "answer": answer,
        "table": None,
        "error": None,
    }


# Identify explicit requests to calculate valid values only.
def explicitly_allows_exclusions(question):

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

    return any(re.search(pattern, text) for pattern in patterns)


# Recognize an explicitly supplied date convention.
def explicit_date_convention(question):

    text = normalize_question(question)

    month_first = bool(
        re.search(
            r"\b(?:mm dd yyyy|mm dd yy|" r"month first|month day year)\b",
            text,
        )
    )

    day_first = bool(
        re.search(
            r"\b(?:dd mm yyyy|dd mm yy|" r"day first|day month year)\b",
            text,
        )
    )

    # Return `None` when `month_first == day_first`.
    if month_first == day_first:
        return None

    return "month_first" if month_first else "day_first"


# Recognize an explicit mixed-discount conversion rule.
def explicitly_defines_discount_scale(question):

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
            r"\b(?:multiply|convert|treat|interpret|" r"represent|assume)\b",
            text,
        )
    )

    return fractional_rule and percentage_rule and conversion_rule


# Resolve a simple whole-column aggregation.
def resolve_simple_aggregation(question, schema):

    text = normalize_question(question)

    match = re.fullmatch(
        AGGREGATION_PATTERN + r"(.+)",
        text,
    )

    # Return `None` when `not match`.
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
    exact = [column for column in columns if column["normalized_name"] == target]

    # Return the selected result when `len(exact) == 1`.
    if len(exact) == 1:
        return exact[0]

    # Resolve a business concept only when it has one candidate.
    from agent.dataset_schema import identify_business_concepts

    concepts = identify_business_concepts(target)

    # Return `None` when `len(concepts) != 1`.
    if len(concepts) != 1:
        return None

    candidates = [
        column
        for column in columns
        if concepts[0] in column.get("business_concepts", [])
    ]

    # Return the selected result when `len(candidates) == 1`.
    if len(candidates) == 1:
        return candidates[0]

    return None


# Prevent incomplete whole-column numeric totals.
def check_numeric_aggregation(question, schema):

    column = resolve_simple_aggregation(
        question,
        schema,
    )

    # Return `None` when `column is None`.
    if column is None:
        return None

    quality = column.get("data_quality", {})

    name = column["name"]

    missing_count = column.get("missing_count", 0)

    invalid_count = quality.get("invalid_numeric_count", 0)

    valid_count = quality.get("numeric_count", 0)

    compatibility = quality.get("numeric_compatibility")

    debug_log(
        "INTEGRITY",
        f"Resolved aggregation column: {name} | "
        f"Valid: {valid_count} | "
        f"Missing: {missing_count} | "
        f"Invalid: {invalid_count}",
    )

    # Do not guess how arbitrary text or identifiers should be aggregated.
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

    # Return `None` when `excluded_count == 0`.
    if excluded_count == 0:
        return None

    # Call `debug_log` when `explicitly_allows_exclusions(question) and valid_count > 0`.
    if explicitly_allows_exclusions(question) and valid_count > 0:
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


# Check date ambiguity for clearly time-based analysis.
def check_date_integrity(question, schema):

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
        if DATE_CONCEPTS.intersection(column.get("business_concepts", []))
    ]

    # Prefer a date column explicitly named in the question.
    named_columns = [
        column
        for column in date_columns
        if re.search(
            r"\b" + re.escape(column["normalized_name"]) + r"\b",
            text,
        )
    ]

    # Update `date_columns` when `len(named_columns) == 1`.
    if len(named_columns) == 1:
        date_columns = named_columns

    # Avoid guessing which date column the question requires.
    if len(date_columns) != 1:
        return None

    column = date_columns[0]

    warnings = set(column.get("data_quality", {}).get("warnings", []))

    relevant = warnings.intersection(DATE_WARNINGS)

    # Return `None` when `not relevant`.
    if not relevant:
        return None

    name = column["name"]

    debug_log(
        "INTEGRITY",
        f"Relevant date column: {name} | " f"Warnings: {sorted(relevant)}",
    )

    # Return the selected result when `'conflicting_date_conventions' in relevant`.
    if "conflicting_date_conventions" in relevant:
        return make_response(
            f"Cannot calculate reliable time-based results "
            f"because '{name}' contains conflicting date "
            "conventions. Please correct the dates or "
            "provide a rule for interpreting the mixed formats."
        )

    # Evaluate the nested condition when `'ambiguous_date_convention' in relevant`.
    if "ambiguous_date_convention" in relevant:
        # Call `debug_log` when `explicit_date_convention(question)`.
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


# Prevent unsupported discount-scale conversions.
def check_discount_integrity(question, schema):

    text = normalize_question(question)

    # Return `None` when `not re.search('\\bdiscounts?\\b', text)`.
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
        if "discount" in column.get("business_concepts", [])
    ]

    # Return `None` when `len(candidates) != 1`.
    if len(candidates) != 1:
        return None

    column = candidates[0]

    warnings = set(column.get("data_quality", {}).get("warnings", []))

    relevant = warnings.intersection(DISCOUNT_WARNINGS)

    # Return `None` when `not relevant`.
    if not relevant:
        return None

    name = column["name"]

    debug_log(
        "INTEGRITY",
        f"Relevant discount column: {name} | " f"Warnings: {sorted(relevant)}",
    )

    # Call `debug_log` when `explicitly_defines_discount_scale(question)`.
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


# Return a clarification response when a relevant
def validate_analysis_integrity(question, schema):

    timer = DebugTimer("INTEGRITY")

    debug_log(
        "INTEGRITY",
        "Starting question-aware data-integrity validation",
    )

    # Call `timer.finish` when `not isinstance(schema, dict)`.
    if not isinstance(schema, dict):
        timer.finish("Validation skipped: dataset profile unavailable")

        return None

    checks = (
        check_date_integrity,
        check_discount_integrity,
        check_numeric_aggregation,
    )

    # Process each `check` from `checks`.
    for check in checks:
        debug_log(
            "INTEGRITY",
            f"Running check: {check.__name__}",
        )

        response = check(question, schema)

        # Call `debug_log` when `response is not None`.
        if response is not None:
            debug_log(
                "INTEGRITY",
                f"Clarification required by {check.__name__}",
            )

            timer.finish("Analysis paused for data-integrity clarification")

            return response

    timer.finish("Data-integrity checks completed; analysis may continue")

    return None
