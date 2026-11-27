import re

import numpy as np
import pandas as pd


CURRENCY_SYMBOLS = {
    "$": "dollar_symbol",
    "€": "EUR",
    "£": "GBP",
    "¥": "yen_or_yuan_symbol",
}

CURRENCY_CODES = {
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "AUD",
    "JPY",
    "INR",
    "CNY",
}

NUMERIC_PATTERN = (
    r"[+-]?"
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)

GROUPED_NUMBER_PATTERN = (
    r"[+-]?[0-9]{1,3}"
    r"(?:,[0-9]{3})+"
    r"(?:\.[0-9]+)?"
)

DATE_PATTERN = (
    r"^(\d{1,2})[/-]"
    r"(\d{1,2})[/-]"
    r"(\d{2}|\d{4})$"
)


def inspect_numeric_values(series):
    """Inspect numeric compatibility without modifying values."""

    valid = series.dropna()
    count = len(valid)

    result = {
        "non_missing_count": count,
        "numeric_count": 0,
        "invalid_numeric_count": 0,
        "nonfinite_count": 0,
        "numeric_compatibility": "empty",
        "currency_hints": [],
        "percentage_symbol_count": 0,
        "formatted_numeric_count": 0,
    }

    if count == 0:
        return result

    if pd.api.types.is_bool_dtype(series.dtype):
        result["numeric_compatibility"] = "boolean"
        return result

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        result["numeric_compatibility"] = "datetime"
        return result

    if pd.api.types.is_complex_dtype(series.dtype):
        result["numeric_compatibility"] = "unsupported_numeric"
        return result

    if pd.api.types.is_numeric_dtype(series.dtype):
        values = valid.to_numpy(dtype=float)
        finite = np.isfinite(values)

        result["numeric_count"] = int(finite.sum())
        result["nonfinite_count"] = int((~finite).sum())
        result["invalid_numeric_count"] = int(
            (~finite).sum()
        )

        result["numeric_compatibility"] = (
            "numeric"
            if finite.all()
            else "numeric_with_invalid_values"
        )

        return result

    text = valid.astype(str).str.strip()
    cleaned = text.copy()

    currencies = set()

    # Identify currency symbols without assuming
    # that every dollar symbol means USD.
    for symbol, currency in CURRENCY_SYMBOLS.items():
        matches = text.str.contains(
            symbol,
            regex=False,
        )

        if matches.any():
            currencies.add(currency)

        cleaned = cleaned.str.replace(
            symbol,
            "",
            regex=False,
        )

    currency_pattern = (
        r"\b(?:"
        + "|".join(sorted(CURRENCY_CODES))
        + r")\b"
    )

    uppercase = cleaned.str.upper()

    detected_codes = uppercase.str.findall(
        currency_pattern
    )

    for codes in detected_codes:
        currencies.update(codes)

    cleaned = uppercase.str.replace(
        currency_pattern,
        "",
        regex=True,
    ).str.strip()

    percentage_mask = cleaned.str.endswith("%")

    result["percentage_symbol_count"] = int(
        percentage_mask.sum()
    )

    cleaned = cleaned.str.replace(
        r"%$",
        "",
        regex=True,
    ).str.strip()

    # Support accounting-style negative values.
    cleaned = cleaned.str.replace(
        r"^\(([^()]+)\)$",
        r"-\1",
        regex=True,
    )

    plain_mask = cleaned.str.fullmatch(
        NUMERIC_PATTERN,
        na=False,
    )

    grouped_mask = cleaned.str.fullmatch(
        GROUPED_NUMBER_PATTERN,
        na=False,
    )

    accepted_mask = plain_mask | grouped_mask

    normalized = cleaned.where(
        accepted_mask
    ).str.replace(
        ",",
        "",
        regex=False,
    )

    converted = pd.to_numeric(
        normalized,
        errors="coerce",
    )

    values = converted.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    finite_mask = pd.Series(
        np.isfinite(values),
        index=valid.index,
    )

    numeric_count = int(finite_mask.sum())

    # Ignore capitalization changes introduced during
    # currency detection, including scientific notation.
    formatted_mask = (
        text.str.upper().ne(cleaned.str.upper())
        | grouped_mask
    ) & finite_mask

    formatted_count = int(
        formatted_mask.sum()
    )

    result["numeric_count"] = numeric_count

    result["invalid_numeric_count"] = (
        count - numeric_count
    )

    result["nonfinite_count"] = int(
        np.isinf(values).sum()
    )

    result["formatted_numeric_count"] = (
        formatted_count
    )

    result["currency_hints"] = sorted(currencies)

    if numeric_count == 0:
        compatibility = "not_numeric"

    elif numeric_count < count:
        compatibility = "partially_numeric_text"

    elif formatted_count > 0:
        compatibility = "formatted_numeric_text"

    else:
        compatibility = "numeric_text"

    result["numeric_compatibility"] = compatibility

    return result


def inspect_date_convention(series):
    """Identify date-format evidence without guessing."""

    valid = series.dropna()

    if valid.empty:
        return "empty"

    if pd.api.types.is_datetime64_any_dtype(
        series.dtype
    ):
        return "datetime"

    text = valid.astype(str).str.strip()

    iso_mask = text.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        na=False,
    )

    if iso_mask.all():
        return "iso_date_candidate"

    parts = text.str.extract(DATE_PATTERN)

    matched = parts.notna().all(axis=1)

    if not matched.any():
        return "unrecognized_date_format"

    if not matched.all():
        return "mixed_or_invalid_date_format"

    first = pd.to_numeric(parts[0])
    second = pd.to_numeric(parts[1])

    invalid = (
        (first < 1)
        | (second < 1)
        | (first > 31)
        | (second > 31)
        | ((first > 12) & (second > 12))
    )

    if invalid.any():
        return "mixed_or_invalid_date_format"

    day_first = (
        (first > 12) & (second <= 12)
    ).any()

    month_first = (
        (second > 12) & (first <= 12)
    ).any()

    if day_first and month_first:
        return "conflicting_date_conventions"

    if day_first:
        return "day_first_candidate"

    if month_first:
        return "month_first_candidate"

    return "ambiguous_date_convention"


def inspect_column_quality(
    series,
    business_concepts=None,
    unit_hint=None,
):
    """Collect data-quality evidence and warnings."""

    concepts = set(business_concepts or [])

    quality = inspect_numeric_values(series)

    warnings = []

    compatibility = quality[
        "numeric_compatibility"
    ]

    if compatibility == "empty":
        warnings.append("all_values_missing")

    if compatibility == "numeric_with_invalid_values":
        warnings.append("nonfinite_numeric_values")

    if compatibility == "numeric_text":
        warnings.append("numeric_conversion_required")

    if compatibility == "formatted_numeric_text":
        warnings.append(
            "formatted_numeric_conversion_required"
        )

    if compatibility == "partially_numeric_text":
        warnings.append("invalid_numeric_text_values")

    if compatibility == "unsupported_numeric":
        warnings.append("unsupported_numeric_type")

    currencies = quality["currency_hints"]

    if len(currencies) > 1:
        warnings.append(
            "multiple_currency_markers"
        )

    if unit_hint == "ambiguous_currency":
        warnings.append(
            "ambiguous_currency_header"
        )

    if unit_hint in CURRENCY_CODES:
        explicit_codes = (
            set(currencies) & CURRENCY_CODES
        )

        if explicit_codes - {unit_hint}:
            warnings.append(
                "currency_header_value_conflict"
            )

    percentage_count = quality[
        "percentage_symbol_count"
    ]

    if (
        0 < percentage_count
        < quality["non_missing_count"]
    ):
        warnings.append(
            "mixed_percentage_notation"
        )

    date_concepts = {
        "transaction_date",
        "order_date",
        "shipping_date",
    }

    if concepts.intersection(date_concepts):
        convention = inspect_date_convention(
            series
        )

        quality["date_convention"] = convention

        if convention in {
            "ambiguous_date_convention",
            "conflicting_date_conventions",
            "mixed_or_invalid_date_format",
            "unrecognized_date_format",
        }:
            warnings.append(convention)

    if (
        "discount" in concepts
        and compatibility == "numeric"
    ):
        valid = series.dropna()

        if not valid.empty:
            if (
                (valid < 0).any()
                or (valid > 100).any()
            ):
                warnings.append(
                    "discount_outside_expected_range"
                )

            if (
                unit_hint == "ratio"
                and (valid > 1).any()
            ):
                warnings.append(
                    "discount_ratio_above_one"
                )

            if (
                unit_hint is None
                and ((valid > 0) & (valid < 1)).any()
                and (valid > 1).any()
            ):
                warnings.append(
                    "potentially_mixed_discount_scale"
                )

    quality["warnings"] = warnings

    return quality
