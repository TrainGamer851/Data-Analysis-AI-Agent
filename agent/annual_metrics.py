"""Deterministic annual business analysis."""

import math
import re

import pandas as pd


ANNUAL_QUESTIONS = {
    "show annual sales and profit by year": "annual",
    "show annual sales and profit": "annual",
    "calculate year over year sales growth": "growth",
    "show year over year sales growth": "growth",
    "are there any invalid or missing order dates": "dates",
    "which year had the highest total profit": "highest_profit",
}


def normalize_question(question):
    question = question.lower().strip()
    question = re.sub(r"[^\w\s]", " ", question)
    return " ".join(question.split())


def money(value):
    if value < 0:
        return f"-${abs(value):,.2f}"

    return f"${value:,.2f}"


def response(answer, table=None):
    return {
        "answer": answer,
        "table": table,
        "error": None,
    }


def parse_order_dates(values):
    """Parse supported dates without guessing ambiguous conventions."""

    text = values.astype("string").str.strip()

    parsed = pd.Series(
        pd.NaT,
        index=values.index,
        dtype="datetime64[ns]",
    )

    # ISO dates are unambiguous.
    iso = text.str.fullmatch(
        r"\d{4}-\d{1,2}-\d{1,2}",
        na=False,
    )

    parsed.loc[iso] = pd.to_datetime(
        text.loc[iso],
        format="%Y-%m-%d",
        errors="coerce",
    )

    # Inspect slash-separated dates.
    parts = text.str.extract(
        r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$"
    )

    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")

    month_first = (
        first.between(1, 12)
        & second.between(13, 31)
    ).any()

    day_first = (
        first.between(13, 31)
        & second.between(1, 12)
    ).any()

    slash_dates = parts[0].notna()

    # Do not guess when the convention is ambiguous
    # or when conflicting conventions are detected.
    if slash_dates.any():
        if month_first == day_first:
            return None

        prefix = "%m/%d/" if month_first else "%d/%m/"

        for length, suffix in [(2, "%y"), (4, "%Y")]:
            mask = (
                slash_dates
                & parts[2].str.len().eq(length)
            )

            parsed.loc[mask] = pd.to_datetime(
                text.loc[mask],
                format=prefix + suffix,
                errors="coerce",
            )

    # Unsupported nonblank formats require the AI fallback.
    unsupported = (
        text.notna()
        & text.ne("")
        & ~iso
        & ~slash_dates
    )

    if unsupported.any():
        return None

    return parsed


def try_annual_analysis(df, question):
    """Return a deterministic result or use the AI fallback."""

    intent = ANNUAL_QUESTIONS.get(
        normalize_question(question)
    )

    if intent is None:
        return None

    if "Order Date" not in df.columns:
        return None

    dates = parse_order_dates(df["Order Date"])

    if dates is None:
        return None

    invalid_count = int(dates.isna().sum())

    # Date validation does not require financial columns.
    if intent == "dates":
        if invalid_count == 0:
            return response(
                "**No invalid or missing order dates.** "
                f"All {len(df):,} records contain valid "
                "order dates."
            )

        return response(
            f"**{invalid_count:,} records** contain "
            "missing or invalid order dates."
        )

    # Never silently exclude transactions.
    if invalid_count > 0:
        return response(
            "A complete annual analysis cannot be "
            f"calculated because {invalid_count:,} "
            "records contain missing or invalid order "
            "dates. Correct these dates and try again."
        )

    required = {
        "annual": ["Sales", "Profit"],
        "growth": ["Sales"],
        "highest_profit": ["Profit"],
    }[intent]

    if not set(required).issubset(df.columns):
        return None

    data = df[required].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if (
            data[column].isna().any()
            or not data[column].map(math.isfinite).all()
        ):
            return None

    data["Year"] = dates.dt.year.astype(int)

    annual = (
        data.groupby("Year")[required]
        .sum()
        .sort_index()
    )

    if annual.empty:
        return response(
            "No annual financial data is available."
        )

    if intent == "highest_profit":
        highest = annual["Profit"].max()

        years = annual.index[
            annual["Profit"] == highest
        ]

        year_text = ", ".join(map(str, years))

        return response(
            f"The highest total profit was "
            f"**{money(highest)}**, recorded in "
            f"**{year_text}**."
        )

    if intent == "annual":
        table = (
            annual.rename(columns={
                "Sales": "Total Sales",
                "Profit": "Total Profit",
            })
            .reset_index()
            .round(2)
            .to_dict(orient="records")
        )

        best_year = int(annual["Profit"].idxmax())
        best_profit = annual["Profit"].max()

        return response(
            f"Annual sales and profit are shown below. "
            f"The highest annual profit was "
            f"**{money(best_profit)}** in "
            f"**{best_year}**.",
            table,
        )

    if intent == "growth":
        rows = []

        for year, sales in annual["Sales"].items():
            year = int(year)
            previous_year = year - 1
            growth = None

            if previous_year in annual.index:
                previous_sales = float(
                    annual.loc[previous_year, "Sales"]
                )

                if previous_sales > 0:
                    growth = round(
                        (float(sales) - previous_sales)
                        / previous_sales * 100,
                        2,
                    )

            rows.append({
                "Year": year,
                "Total Sales": round(float(sales), 2),
                "Sales Growth %": growth,
            })

        return response(
            "Year-over-year sales growth is calculated "
            "against the preceding calendar year. "
            "Growth is unavailable when the preceding "
            "year is missing or has nonpositive sales. "
            "These figures compare recorded annual totals; "
            "partial years may not be directly comparable.",
            rows,
        )

    return None