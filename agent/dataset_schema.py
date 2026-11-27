import re
import math
from numbers import Real

import pandas as pd

from agent.debug_logger import debug_log, DebugTimer
from agent.data_quality import inspect_column_quality


# Candidate business concepts, not automatic substitutions.
BUSINESS_ALIASES = {
    "revenue": {
        "sales",
        "total sales",
        "revenue",
        "net sales",
        "generated revenue",
        "sales revenue",
        "total revenue",
        "net revenue",
        "sales amount",
        "revenue amount",
    },
    "profit": {
        "profit",
        "total profit",
    },
    "net_profit": {
        "net profit",
        "total net profit",
    },
    "gross_profit": {
        "gross profit",
        "total gross profit",
    },
    "net_income": {
        "net income",
        "total net income",
    },
    "product_category": {
        "category",
        "product type",
        "product category",
    },
    "product": {
        "product",
        "product name",
        "item name",
    },
    "region": {
        "region",
        "sales region",
        "market region",
    },
    "customer_segment": {
        "segment",
        "customer segment",
    },
    "customer_id": {
        "customer id",
        "client id",
    },
    "order_id": {
        "order id",
        "order number",
    },
    "transaction_date": {
        "transaction date",
        "sale date",
        "sales date",
    },
    "order_date": {
        "order date",
        "purchase date",
    },
    "shipping_date": {
        "ship date",
        "shipping date",
    },
    "discount": {
        "discount",
        "discount rate",
        "discount pct",
        "discount percentage",
        "discount ratio",
    },
    "quantity": {
        "quantity",
        "units sold",
        "sales quantity",
    },
    "advertising_spend": {
        "advertising spend",
        "ad spend",
        "advertising cost",
        "ad cost",
    },
    "campaign": {
        "campaign",
        "campaign name",
        "marketing campaign",
    },
    "conversions": {
        "conversions",
        "conversion count",
    },
    "clicks": {
        "clicks",
        "ad clicks",
        "click count",
    },
    "impressions": {
        "impressions",
        "ad impressions",
    },
    "inventory_quantity": {
        "stock on hand",
        "inventory quantity",
        "stock quantity",
        "quantity on hand",
    },
    "reorder_level": {
        "reorder level",
        "reorder point",
    },
    "unit_cost": {
        "unit cost",
        "cost per unit",
    },
    "unit_price": {
        "unit price",
        "selling price",
        "unit selling price",
    },
    "cost_of_goods_sold": {
        "cost of goods sold",
        "cogs",
    },
    "average_inventory_value": {
        "average inventory value",
        "avg inventory value",
    },
    "salary": {
        "salary",
        "base salary",
        "base pay",
    },
    "annual_salary": {
        "annual salary",
        "yearly salary",
    },
    "monthly_salary": {
        "monthly salary",
    },
    "department": {
        "department",
        "department name",
    },
    "employee_id": {
        "employee id",
        "staff id",
    },
    "years_experience": {
        "years experience",
        "years of experience",
    },
    "transaction_type": {
        "transaction type",
    },
    "amount": {
        "amount",
        "transaction amount",
    },
    "income": {
        "income",
        "total income",
    },
    "expenses": {
        "expenses",
        "expense",
        "total expenses",
    },
    "fixed_costs": {
        "fixed cost",
        "fixed costs",
    },
    "variable_costs": {
        "variable cost",
        "variable costs",
    },
    "unit_variable_cost": {
        "unit variable cost",
        "variable cost per unit",
    },
    "product_subcategory": {
        "subcategory",
        "sub category",
        "product subcategory",
        "product sub category",
    },
    "investment_cost": {
        "investment cost",
        "initial investment",
        "capital invested",
    },
    "investment_return": {
        "investment return",
        "investment proceeds",
    },
    "roi": {
        "roi",
        "roi pct",
        "roi percentage",
        "return on investment",
        "return on investment pct",
    },
    "roas": {
        "roas",
        "return on ad spend",
        "return on advertising spend",
    },
    "profit_margin": {
        "profit margin",
        "profit margin pct",
        "profit margin percentage",
    },
    "net_profit_margin": {
        "net profit margin",
        "net profit margin pct",
    },
    "gross_profit_margin": {
        "gross profit margin",
        "gross profit margin pct",
    },
    "operating_expenses": {
        "operating expense",
        "operating expenses",
        "operating costs",
        "opex",
    },
    "acquisition_cost": {
        "acquisition cost",
        "customer acquisition expenditure",
    },
    "new_customers": {
        "new customers",
        "newly acquired customers",
        "acquired customers",
    },
    "inventory_value": {
        "inventory value",
        "stock value",
        "total inventory value",
    },
    "beginning_inventory_value": {
        "beginning inventory value",
        "opening inventory value",
    },
    "ending_inventory_value": {
        "ending inventory value",
        "closing inventory value",
    },
    "employee_departures": {
        "employee departures",
        "employee separations",
    },
    "average_headcount": {
        "average headcount",
        "average employee headcount",
    },
    "customer_lifetime_value": {
        "customer lifetime value",
        "clv",
        "customer ltv",
    },
}


def normalize_column_name(name):
    """Normalize a column name without modifying the dataset."""

    text = str(name).strip()

    # Support names such as NetProfit and ProductCategory.
    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text,
    )

    # Separate acronyms from subsequent words.
    # Example: ROIValue becomes ROI Value.
    text = re.sub(
        r"([A-Z]+)([A-Z][a-z])",
        r"\1 \2",
        text,
    )

    text = text.lower()
    text = text.replace("%", " pct ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(text.split())


def identify_business_concepts(name):
    """Return candidate business concepts for a column."""

    normalized = normalize_column_name(name)

    words = normalized.split()

    # Currency suffixes describe units rather than
    # changing the underlying business concept.
    if words and words[-1] in {
        "usd",
        "eur",
        "gbp",
        "cad",
        "aud",
        "jpy",
        "inr",
    }:
        words = words[:-1]

    base_name = " ".join(words)

    return [
        concept
        for concept, aliases in BUSINESS_ALIASES.items()
        if normalized in aliases or base_name in aliases
    ]


def identify_column_type(series):
    """Identify the actual pandas column type."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"

    return "text"


def safe_number(value):
    """Convert a finite numeric value into a Python number."""

    if isinstance(value, bool):
        return None

    if not isinstance(value, Real):
        return None

    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (ValueError, TypeError, OverflowError):
        return None


def identify_unit_hint(name):
    """Identify explicit unit hints without guessing scale."""

    normalized = normalize_column_name(name)
    words = normalized.split()

    if any(
        word in words
        for word in ("pct", "percent", "percentage")
    ):
        return "percentage"

    if "ratio" in words:
        return "ratio"

    currencies = {
        "usd",
        "eur",
        "gbp",
        "cad",
        "aud",
        "jpy",
        "inr",
    }

    detected = currencies.intersection(words)

    if len(detected) == 1:
        return next(iter(detected)).upper()

    # Multiple currencies require clarification.
    if len(detected) > 1:
        return "ambiguous_currency"

    return None


def profile_column(name, series):
    """Collect descriptive information for one column."""

    column_type = identify_column_type(series)

    row_count = len(series)
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    profile = {
        "name": str(name),
        "normalized_name": normalize_column_name(name),
        "dtype": str(series.dtype),
        "type": column_type,
        "missing_count": missing_count,
        "missing_pct": (
            round(missing_count / row_count * 100, 2)
            if row_count
            else 0.0
        ),
        "unique_count": unique_count,
        "business_concepts": identify_business_concepts(
            name
        ),
        "unit_hint": identify_unit_hint(name),
    }

    if column_type == "numeric":
        valid = series.dropna()

        if not valid.empty:
            profile["minimum"] = safe_number(
                valid.min()
            )

            profile["maximum"] = safe_number(
                valid.max()
            )
        else:
            profile["minimum"] = None
            profile["maximum"] = None

    profile["data_quality"] = inspect_column_quality(
        series,
        business_concepts=profile["business_concepts"],
        unit_hint=profile["unit_hint"],
    )

    return profile


def profile_dataset(dataframe):
    """Build a non-mutating business schema profile."""

    debug_log(
        "SCHEMA",
        "Starting dataset profiling",
    )

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Dataset must be a pandas DataFrame."
        )

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Dataset contains duplicate column names."
        )

    profiles = []
    candidates = {}

    for name in dataframe.columns:
        column_profile = profile_column(
            name,
            dataframe[name],
        )

        profiles.append(column_profile)

        for concept in column_profile[
            "business_concepts"
        ]:
            candidates.setdefault(
                concept,
                [],
            ).append(str(name))

    ambiguous = {
        concept: columns
        for concept, columns in candidates.items()
        if len(columns) > 1
    }

    unique_candidates = {
        concept: columns[0]
        for concept, columns in candidates.items()
        if len(columns) == 1
    }

    result = {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "columns": profiles,
        "business_candidates": candidates,
        "unique_candidates": unique_candidates,
        "ambiguous_candidates": ambiguous,
    }

    debug_log(
        "SCHEMA",
        f"Profiled {result['row_count']} rows "
        f"and {result['column_count']} columns",
    )

    debug_log(
        "SCHEMA",
        f"Recognized business concepts: "
        f"{list(candidates)}",
    )

    debug_log(
        "SCHEMA",
        f"Ambiguous business concepts: "
        f"{list(ambiguous)}",
    )

    quality_timer = DebugTimer("DATA QUALITY")

    warning_count = 0
    affected_columns = 0

    for column in profiles:
        quality = column.get(
            "data_quality",
            {},
        )

        warnings = quality.get(
            "warnings",
            [],
        )

        if warnings:
            affected_columns += 1
            warning_count += len(warnings)

            debug_log(
                "DATA QUALITY",
                f"Column: {column['name']} | "
                f"Warnings: {warnings}",
            )

        if quality.get("numeric_compatibility") in {
            "numeric_text",
            "formatted_numeric_text",
            "partially_numeric_text",
        }:
            debug_log(
                "DATA QUALITY",
                f"Column: {column['name']} | "
                f"Numeric compatibility: "
                f"{quality['numeric_compatibility']} | "
                f"Convertible values: "
                f"{quality['numeric_count']} | "
                f"Invalid values: "
                f"{quality['invalid_numeric_count']}",
            )

    debug_log(
        "DATA QUALITY",
        f"Columns with warnings: {affected_columns} | "
        f"Total warnings: {warning_count}",
    )

    quality_timer.finish(
        "Data-quality reporting completed"
    )

    debug_log(
        "SCHEMA",
        "Dataset profiling completed",
    )

    return result