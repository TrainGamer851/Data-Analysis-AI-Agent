import math
import re
from numbers import Real


CURRENCY_PREFIXES = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "CAD": "CAD ",
    "AUD": "AUD ",
    "JPY": "JPY ",
    "INR": "INR ",
    "CNY": "CNY ",
}


def get_explicit_currency(name):
    """Recognize an explicit currency suffix."""

    normalized = str(name).strip().upper()

    tokens = re.findall(
        r"[A-Z0-9]+",
        normalized,
    )

    currencies = {
        token
        for token in tokens
        if token in CURRENCY_PREFIXES
    }

    # Never select arbitrarily between multiple currencies.
    if len(currencies) != 1:
        return None

    if not tokens:
        return None

    currency = tokens[-1]

    if currency not in CURRENCY_PREFIXES:
        return None

    # A currency code by itself is not a metric name.
    if len(tokens) < 2:
        return None

    return currency


def format_currency(value, currency):
    """Format a finite monetary value without conversion."""

    if currency not in CURRENCY_PREFIXES:
        raise ValueError("Unsupported currency")

    if isinstance(value, bool):
        return str(value)

    if not isinstance(value, Real):
        return str(value)

    if not math.isfinite(value):
        return "Unavailable"

    prefix = CURRENCY_PREFIXES[currency]
    sign = "-" if value < 0 else ""

    return f"{sign}{prefix}{abs(value):,.2f}"