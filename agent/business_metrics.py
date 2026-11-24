"""Fast, deterministic calculations for common business questions."""

import re

import pandas as pd


OVERVIEW_QUESTIONS = {
    "analyze the overall business performance",
    "analyze overall business performance",
    "summarize business performance",
    "provide an overview of business performance",
    "how is the business performing",
}

RECOMMENDATION_QUESTIONS = {
    "based on the dataset provide three actionable recommendations to improve profitability",
    "provide three actionable recommendations to improve profitability",
    "give me three recommendations to improve profitability",
}


def normalize_question(question):
    question = question.lower().strip()
    question = re.sub(r"[^\w\s]", "", question)
    return " ".join(question.split())


def money(value):
    if value < 0:
        return f"-${abs(value):,.2f}"

    return f"${value:,.2f}"


def percentage(value):
    return f"{value:,.2f}%"


def prepare_financial_data(df):
    """Validate the financial columns required by the fast path."""

    required = {"Sales", "Profit"}

    if not required.issubset(df.columns):
        return None

    data = df.copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any():
            return None

        if data[column].isin(
            [float("inf"), float("-inf")]
        ).any():
            return None

    return data


def grouped_profit(data, column):
    """Calculate total profit for each group."""

    return (
        data.groupby(column)["Profit"]
        .sum()
        .sort_values(ascending=False)
    )


def overall_performance(data):
    """Generate a verified business performance summary."""

    total_sales = float(data["Sales"].sum())
    total_profit = float(data["Profit"].sum())

    lines = [
        "### Overall Business Performance",
        "",
        f"- **Total sales:** {money(total_sales)}",
        f"- **Total profit:** {money(total_profit)}",
    ]

    if total_sales != 0:
        margin = total_profit / total_sales * 100

        lines.append(
            f"- **Profit margin:** {percentage(margin)}"
        )

    if "Customer ID" in data.columns:
        customers = data["Customer ID"].nunique()

        lines.append(
            f"- **Unique customers:** {customers:,}"
        )

    if "Order ID" in data.columns:
        orders = data["Order ID"].nunique()

        lines.append(
            f"- **Unique orders:** {orders:,}"
        )

        if orders > 0 and data["Order ID"].notna().all():
            aov = total_sales / orders

            lines.append(
                f"- **Average order value:** {money(aov)}"
            )

    lines.extend([
        "",
        "### Key Business Findings",
        "",
    ])

    if "Category" in data.columns:
        profits = grouped_profit(data, "Category")

        if not profits.empty:
            lines.append(
                f"- **Highest-profit category:** "
                f"{profits.index[0]} "
                f"({money(profits.iloc[0])})."
            )

            lines.append(
                f"- **Lowest-profit category:** "
                f"{profits.index[-1]} "
                f"({money(profits.iloc[-1])})."
            )

    if "Region" in data.columns:
        profits = grouped_profit(data, "Region")

        if not profits.empty:
            lines.append(
                f"- **Highest-profit region:** "
                f"{profits.index[0]} "
                f"({money(profits.iloc[0])})."
            )

            lines.append(
                f"- **Lowest-profit region:** "
                f"{profits.index[-1]} "
                f"({money(profits.iloc[-1])})."
            )

    if "Sub-Category" in data.columns:
        profits = grouped_profit(data, "Sub-Category")
        losses = profits[profits < 0]

        if not losses.empty:
            lines.append(
                f"- **Loss-generating subcategories:** "
                f"{', '.join(map(str, losses.index))}."
            )

    return {
        "answer": "\n".join(lines),
        "table": None,
        "error": None,
    }


def profitability_recommendations(data):
    """Generate recommendations supported by actual metrics."""

    recommendations = []

    # Recommendation 1: Investigate loss-generating products.
    if "Sub-Category" in data.columns:
        profits = grouped_profit(data, "Sub-Category")
        losses = profits[profits < 0]

        if not losses.empty:
            worst = losses.idxmin()
            loss = float(losses.min())

            recommendations.append(
                "**Review loss-generating products.** "
                f"{worst} generated {money(loss)} in profit. "
                "Review pricing, discounts, and product costs "
                "before deciding whether to reduce or "
                "restructure this product offering."
            )

    # Recommendation 2: Improve category profitability.
    if "Category" in data.columns:
        summary = data.groupby("Category")[
            ["Sales", "Profit"]
        ].sum()

        summary = summary[summary["Sales"] > 0].copy()

        if len(summary) >= 2:
            margins = (
                summary["Profit"]
                / summary["Sales"]
                * 100
            )

            weakest = margins.idxmin()
            strongest = margins.idxmax()

            if margins[weakest] < margins[strongest]:
                recommendations.append(
                    "**Improve low-margin categories.** "
                    f"{weakest} has a profit margin of "
                    f"{percentage(margins[weakest])}, "
                    f"compared with "
                    f"{percentage(margins[strongest])} "
                    f"for {strongest}. "
                    "Investigate differences in pricing, "
                    "product mix, and costs."
                )

    # Recommendation 3: Improve regional profitability.
    if "Region" in data.columns:
        summary = data.groupby("Region")[
            ["Sales", "Profit"]
        ].sum()

        summary = summary[summary["Sales"] > 0].copy()

        if len(summary) >= 2:
            margins = (
                summary["Profit"]
                / summary["Sales"]
                * 100
            )

            weakest = margins.idxmin()
            strongest = margins.idxmax()

            if margins[weakest] < margins[strongest]:
                recommendations.append(
                    "**Improve regional profit margins.** "
                    f"{weakest} has a profit margin of "
                    f"{percentage(margins[weakest])}, "
                    f"compared with "
                    f"{percentage(margins[strongest])} "
                    f"for {strongest}. "
                    "Investigate regional pricing, costs, "
                    "and sales mix before applying practices "
                    "from higher-margin regions."
                )

    # Let the general-purpose agent handle datasets that
    # cannot support three recommendations through this path.
    if len(recommendations) < 3:
        return None

    lines = [
        "### Recommendations to Improve Profitability",
        "",
    ]

    for number, recommendation in enumerate(
        recommendations[:3],
        start=1,
    ):
        lines.append(f"{number}. {recommendation}")
        lines.append("")

    return {
        "answer": "\n".join(lines),
        "table": None,
        "error": None,
    }


def try_business_analysis(df, question):
    """Return a fast result or None to use the AI agent."""

    normalized = normalize_question(question)

    if normalized not in (
        OVERVIEW_QUESTIONS | RECOMMENDATION_QUESTIONS
    ):
        return None

    data = prepare_financial_data(df)

    if data is None:
        return None

    if normalized in OVERVIEW_QUESTIONS:
        return overall_performance(data)

    if normalized in RECOMMENDATION_QUESTIONS:
        return profitability_recommendations(data)

    return None