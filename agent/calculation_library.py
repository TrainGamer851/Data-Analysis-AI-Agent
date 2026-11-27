"""Trusted, read-only calculations. No generated code or network access."""
import math
import numpy as np
import pandas as pd
from agent.currency_units import get_explicit_currency, format_currency
from agent.dataset_schema import identify_business_concepts, normalize_column_name

class CalculationError(ValueError):
    pass


def resolve_column(df, name, concept=False):
    matches = [c for c in df.columns if (name in identify_business_concepts(c) if concept else normalize_column_name(c) == normalize_column_name(name))]
    if len(matches) != 1:
        raise CalculationError(f"Please specify a unique column for {name} (found {len(matches)} candidates).")
    return matches[0]


def numeric(df, column, nonnegative=False):
    s = df[column]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_complex_dtype(s):
        raise CalculationError(f"Column {column} must contain numeric values with consistent units.")
    a = s.to_numpy(dtype=float, na_value=np.nan)
    if not len(a) or not np.isfinite(a).all():
        raise CalculationError(f"Column {column} contains missing or nonfinite values, or is empty. Please clean the data first.")
    if nonnegative and (a < 0).any():
        raise CalculationError(f"Column {column} must be nonnegative for this calculation.")
    return a


def divide(n, d):
    if not math.isfinite(float(d)) or d <= 0:
        raise CalculationError("The denominator must be finite and positive.")
    return n / d


def compatible_currency(*columns):
    units = [get_explicit_currency(c) for c in columns]
    if any(units) and len(set(units)) != 1:
        raise CalculationError("Currency units differ or are unspecified. Please confirm consistent currency units.")
    return units[0]


def display(value, unit=None):
    if not math.isfinite(float(value)):
        raise CalculationError("The calculated result is not finite.")
    if unit == "%":
        return f"{value:,.2f}%"
    if unit:
        return format_currency(value, unit)
    return f"{value:,.4f}".rstrip("0").rstrip(".")

# Each ratio uses sums over the uploaded records, not averages of row ratios.
RATIOS = {
    "return on ad spend": ("revenue", "advertising_spend", 1, False, True),
    "roas": ("revenue", "advertising_spend", 1, False, True),
    "marketing roi": ("revenue", "advertising_spend", 100, True, True),
    "customer acquisition cost": ("acquisition_cost", "new_customers", 1, False, False),
    "conversion rate": ("conversions", "clicks", 100, False, False),
    "click through rate": ("clicks", "impressions", 100, False, False),
    "cost per conversion": ("advertising_spend", "conversions", 1, False, False),
    "inventory turnover": ("cost_of_goods_sold", "average_inventory_value", 1, False, True),
    "employee attrition rate": ("employee_departures", "average_headcount", 100, False, False),
    "gross profit margin": ("gross_profit", "revenue", 100, False, True),
    "net profit margin": ("net_profit", "revenue", 100, False, True),
    "investment roi": ("investment_return", "investment_cost", 100, True, True),
}


def ratio(df, metric):
    n, d, scale, subtract, same_currency = RATIOS[metric]
    nc, dc = resolve_column(df, n, True), resolve_column(df, d, True)
    if same_currency:
        compatible_currency(nc, dc)
    a, b = numeric(df, nc, nonnegative=n not in {"gross_profit", "net_profit", "investment_return"}), numeric(df, dc, True)
    numerator, denominator = float(a.sum()), float(b.sum())
    if metric in {"conversion rate", "click through rate"} and (a > b).any():
        raise CalculationError("Event counts exceed the corresponding opportunity counts.")
    value = divide(numerator - denominator if subtract else numerator, denominator) * scale
    unit = "%" if scale == 100 else get_explicit_currency(nc) if metric in {"customer acquisition cost", "cost per conversion"} else None
    return value, unit, f"({'sum(' + str(nc) + ') - ' + 'sum(' + str(dc) + ')' if subtract else 'sum(' + str(nc) + ')'}) / sum({dc})" + (" × 100" if scale == 100 else "")


def describe(df, column, metric):
    a = numeric(df, column)
    if metric in {"sample variance", "sample standard deviation"} and len(a) < 2:
        raise CalculationError("At least two observations are required.")
    funcs = {"mean": np.mean, "median": np.median, "minimum": np.min, "maximum": np.max, "sum": np.sum, "sample variance": lambda x: np.var(x, ddof=1), "sample standard deviation": lambda x: np.std(x, ddof=1), "population variance": np.var, "population standard deviation": np.std}
    return float(funcs[metric](a))


def inventory_value(df):
    q, c = [resolve_column(df, x, True) for x in ("inventory_quantity", "unit_cost")]
    return float(np.dot(numeric(df, q, True), numeric(df, c, True))), get_explicit_currency(c)


def unique_ids(df, concept):
    c = resolve_column(df, concept, True)
    s = df[c]
    if s.isna().any() or s.astype(str).str.strip().eq("").any():
        raise CalculationError(f"Missing identifiers in {c}.")
    return s


def repeat_purchase_rate(df):
    customers, orders = unique_ids(df, "customer_id"), unique_ids(df, "order_id")
    pairs = pd.DataFrame({"customer": customers, "order": orders}).drop_duplicates()
    if pairs["order"].duplicated().any():
        raise CalculationError("An order belongs to multiple customers.")
    counts = pairs.groupby("customer")["order"].nunique()
    return divide(int((counts > 1).sum()), len(counts)) * 100
