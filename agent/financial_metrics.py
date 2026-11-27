import math
import re

import pandas as pd

from agent.debug_logger import debug_log, DebugTimer
from agent.result_formatter import escape_text, format_named_value


GROUP_ALIASES = {
    "region": "Region",
    "regions": "Region",
    "category": "Category",
    "categories": "Category",
    "product category": "Category",
    "product categories": "Category",
    "subcategory": "Sub-Category",
    "subcategories": "Sub-Category",
    "product subcategory": "Sub-Category",
    "product subcategories": "Sub-Category",
    "segment": "Segment",
    "segments": "Segment",
}

GROUP_PATTERN = "|".join(
    re.escape(name)
    for name in sorted(
        GROUP_ALIASES,
        key=len,
        reverse=True,
    )
)

MAX_RESULT_ROWS = 200


def group_count_text(group, count):
    """Return a readable singular or plural group count."""

    names = {
        "Region": ("region", "regions"),
        "Category": ("category", "categories"),
        "Sub-Category": (
            "subcategory",
            "subcategories",
        ),
        "Segment": ("segment", "segments"),
    }

    singular, plural = names.get(
        group,
        ("group", "groups"),
    )

    if count == 0:
        return f"No {plural}"

    if count == 1:
        return f"1 {singular}"

    return f"{count:,} {plural}"


def normalize_financial_question(question):
    """Normalize wording while preserving financial symbols."""

    question = question.lower().strip()

    question = question.replace("sub-categor", "subcategor")
    question = question.replace("above-average", "above average")
    question = question.replace("below-average", "below average")

    question = question.rstrip("?.!")

    return " ".join(question.split())


def resolve_financial_intent(question):
    """Recognize supported standalone financial questions."""

    q = normalize_financial_question(question)

    # Discount and profitability relationships.
    match = re.fullmatch(
        r"(?:analyze|analyse|show) the relationship "
        r"between discount and (?:profitability|profit) "
        rf"(?:across|by) ({GROUP_PATTERN})",
        q,
    )

    if match:
        return {
            "kind": "discount_relationship",
            "group": GROUP_ALIASES[match.group(1)],
        }

    # Average discount comparisons.
    match = re.fullmatch(
        r"(?:show|compare|calculate) "
        r"(?:the )?average discount "
        rf"(?:by|across) ({GROUP_PATTERN})"
        r"(?: in a table)?",
        q,
    )

    if match:
        return {
            "kind": "average_discount",
            "group": GROUP_ALIASES[match.group(1)],
        }

    # Combined sales and profit conditions.
    match = re.fullmatch(
        rf"which ({GROUP_PATTERN}) "
        r"have above average (?:total )?sales "
        r"(?:but|and) negative (?:total )?profit",
        q,
    )

    if match:
        return {
            "kind": "above_average_loss",
            "group": GROUP_ALIASES[match.group(1)],
        }

    # Profit margin thresholds.
    match = re.fullmatch(
        rf"which ({GROUP_PATTERN}) "
        r"have (?:a )?profit margins? "
        r"(above|below|over|under) "
        r"(-?\d+(?:\.\d+)?)\s*%",
        q,
    )

    if match:
        group, direction, threshold = match.groups()

        threshold = float(threshold)

        if not math.isfinite(threshold):
            return None

        return {
            "kind": "margin_threshold",
            "group": GROUP_ALIASES[group],
            "direction": direction,
            "threshold": threshold,
        }

    # Monetary thresholds.
    match = re.fullmatch(
        rf"which ({GROUP_PATTERN}) "
        r"have (?:total )?(sales|profit) "
        r"(above|below|over|under) "
        r"(-?\$?(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?)",
        q,
    )

    if match:
        group, metric, direction, threshold = match.groups()

        threshold = float(
            threshold.replace("$", "").replace(",", "")
        )

        if not math.isfinite(threshold):
            return None

        return {
            "kind": "money_threshold",
            "group": GROUP_ALIASES[group],
            "metric": metric.title(),
            "direction": direction,
            "threshold": threshold,
        }

    return None


def calculate_grouped_metrics(df, group, metrics):
    """Calculate validated group totals without modifying input."""

    timer = DebugTimer("FINANCIAL CALCULATION")

    debug_log(
        "FINANCIAL CALCULATION",
        f"Grouping by: {group} | Metrics: {metrics}",
    )

    required = [group, *metrics]

    if (
        df.empty
        or not df.columns.is_unique
        or not set(required).issubset(df.columns)
    ):
        debug_log(
            "FINANCIAL VALIDATION",
            "Required financial columns unavailable",
        )

        return None

    data = df[required].copy()

    if data[group].isna().any():
        debug_log(
            "FINANCIAL VALIDATION",
            "Missing grouping values detected",
        )

        return None

    if not data[group].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        debug_log(
            "FINANCIAL VALIDATION",
            "Unsupported grouping values",
        )

        return None

    for metric in metrics:
        if data[metric].map(
            lambda value: isinstance(value, bool)
        ).any():
            return None

        data[metric] = pd.to_numeric(
            data[metric],
            errors="coerce",
        )

        if (
            data[metric].isna().any()
            or not data[metric].map(math.isfinite).all()
        ):
            debug_log(
                "FINANCIAL VALIDATION",
                f"Invalid numeric values in {metric}",
            )

            return None

    grouped = (
        data.groupby(group, observed=True)[metrics]
        .sum()
        .reset_index()
    )

    if grouped.empty:
        return None

    for metric in metrics:
        if not grouped[metric].map(math.isfinite).all():
            return None

    debug_log(
        "FINANCIAL CALCULATION",
        f"Calculated groups: {len(grouped)}",
    )

    for row in grouped.head(30).to_dict("records"):
        debug_log(
            "FINANCIAL GROUP",
            str(row),
        )

    timer.finish("Grouped financial totals calculated")

    return grouped


def calculate_profit_margin(grouped):
    """Calculate margins using total profit divided by total sales."""

    grouped = grouped.copy()

    valid_sales = grouped["Sales"].where(
        grouped["Sales"] > 0
    )

    grouped["Profit_Margin_Pct"] = (
        grouped["Profit"] / valid_sales * 100
    )

    grouped["Profit_Margin_Pct"] = (
        grouped["Profit_Margin_Pct"].replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )
    )

    debug_log(
        "FINANCIAL CALCULATION",
        "Profit margins calculated",
    )

    return grouped


def threshold_mask(values, direction, threshold):
    """Apply a strict upper or lower threshold."""

    if direction in {"above", "over"}:
        return values > threshold

    return values < threshold


def financial_response(answer, rows=None):
    """Return the existing agent response structure."""

    return {
        "answer": answer,
        "table": rows,
        "error": None,
    }


def prepare_result_rows(grouped):
    """Convert calculated values into frontend-compatible records."""

    records = []

    for row in grouped.head(MAX_RESULT_ROWS).to_dict("records"):
        record = {}

        for column, value in row.items():
            if pd.isna(value):
                record[column] = None

            elif isinstance(value, str):
                record[column] = value

            elif "correlation" in column.lower():
                record[column] = round(float(value), 4)

            else:
                record[column] = round(float(value), 2)

        records.append(record)

    debug_log(
        "FINANCIAL RESULT",
        f"Supporting rows prepared: {len(records)}",
    )

    return records


def append_display_limit(answer, count):
    """Disclose when supporting results exceed the display limit."""

    if count > MAX_RESULT_ROWS:
        answer += (
            f"\n\nShowing the first {MAX_RESULT_ROWS:,} "
            f"of {count:,} matching groups."
        )

    return answer


def above_average_loss_result(df, group):
    """Find groups with above-average total sales and negative profit."""

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    if grouped is None:
        return None

    average_sales = float(grouped["Sales"].mean())

    if not math.isfinite(average_sales):
        return None

    debug_log(
        "FINANCIAL FILTER",
        f"Average group sales threshold: {average_sales:,.6f}",
    )

    matches = grouped.loc[
        (grouped["Sales"] > average_sales)
        & (grouped["Profit"] < 0)
    ].copy()

    matches = matches.sort_values(
        "Sales",
        ascending=False,
    )

    count = len(matches)

    debug_log(
        "FINANCIAL FILTER",
        f"Groups matching both conditions: {count}",
    )

    threshold_text = format_named_value(
        "Sales",
        average_sales,
    )

    if matches.empty:
        return financial_response(
            "No groups have both above-average total "
            "sales and negative total profit.\n\n"
            f"The average group sales threshold is "
            f"**{threshold_text}**."
        )

    matches = calculate_profit_margin(matches)

    matches["Average_Sales_Threshold"] = average_sales

    matches = matches.rename(columns={
        "Sales": "Total_Sales",
        "Profit": "Total_Profit",
    })

    answer = (
        f"**{group_count_text(group, count)}** "
        f"have total sales above "
        f"the average group sales threshold of "
        f"**{threshold_text}** and negative total profit."
    )

    if count == 1:
        row = matches.iloc[0]

        answer = (
            f"**{escape_text(row[group])}** is the only "
            "group that satisfies both conditions.\n\n"
            f"- **Total sales:** "
            f"{format_named_value('Sales', row['Total_Sales'])}\n"
            f"- **Total profit:** "
            f"{format_named_value('Profit', row['Total_Profit'])}\n"
            f"- **Average group sales threshold:** "
            f"{threshold_text}"
        )

        if pd.notna(row["Profit_Margin_Pct"]):
            answer += (
                f"\n- **Profit margin:** "
                f"{row['Profit_Margin_Pct']:.2f}%"
            )

    return financial_response(
        append_display_limit(answer, count),
        prepare_result_rows(matches),
    )


def margin_threshold_result(df, group, direction, threshold):
    """Find groups satisfying a profit margin threshold."""

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    if grouped is None:
        return None

    grouped = calculate_profit_margin(grouped)

    unavailable = int(
        grouped["Profit_Margin_Pct"].isna().sum()
    )

    matches = grouped.loc[
        threshold_mask(
            grouped["Profit_Margin_Pct"],
            direction,
            threshold,
        )
    ].copy()

    matches = matches.sort_values(
        "Profit_Margin_Pct",
        ascending=direction in {"below", "under"},
    )

    count = len(matches)

    debug_log(
        "FINANCIAL FILTER",
        f"Margin threshold: {threshold}% | Matches: {count}",
    )

    answer = (
        f"**{group_count_text(group, count)}** "
        f"have a profit margin "
        f"{direction} **{threshold:g}%**."
    )

    if count == 1:
        row = matches.iloc[0]

        answer = (
            f"**{escape_text(row[group])}** has a profit "
            f"margin of **{row['Profit_Margin_Pct']:.2f}%**, "
            f"which is {direction} **{threshold:g}%**."
        )

    if unavailable:
        answer += (
            f"\n\nProfit margins were unavailable for "
            f"{unavailable:,} group(s) with nonpositive "
            "sales or nonfinite calculated margins."
        )

    matches = matches.rename(columns={
        "Sales": "Total_Sales",
        "Profit": "Total_Profit",
    })

    return financial_response(
        append_display_limit(answer, count),
        prepare_result_rows(matches) if count else None,
    )


def money_threshold_result(
    df,
    group,
    metric,
    direction,
    threshold,
):
    """Find groups satisfying a monetary threshold."""

    grouped = calculate_grouped_metrics(
        df,
        group,
        [metric],
    )

    if grouped is None:
        return None

    matches = grouped.loc[
        threshold_mask(
            grouped[metric],
            direction,
            threshold,
        )
    ].copy()

    matches = matches.sort_values(
        metric,
        ascending=direction in {"below", "under"},
    )

    count = len(matches)

    threshold_text = format_named_value(
        metric,
        threshold,
    )

    debug_log(
        "FINANCIAL FILTER",
        f"{metric} threshold: {threshold_text} | Matches: {count}",
    )

    answer = (
        f"**{group_count_text(group, count)}** "
        f"recorded total "
        f"{metric.lower()} {direction} "
        f"**{threshold_text}**."
    )

    if count:
        lines = [answer, ""]

        for _, row in matches.head(5).iterrows():
            lines.append(
                f"- **{escape_text(row[group])}:** "
                f"{format_named_value(metric, row[metric])}"
            )

        answer = "\n".join(lines)

        if count > 5:
            answer += (
                "\n\nAdditional matching groups "
                "are shown in the supporting data."
            )

    matches = matches.rename(columns={
        metric: f"Total_{metric}",
    })

    return financial_response(
        append_display_limit(answer, count),
        prepare_result_rows(matches) if count else None,
    )


def calculate_discount_metrics(df, group, relationship=False):
    """Calculate grouped discounts and optional financial metrics."""

    timer = DebugTimer("DISCOUNT CALCULATION")

    debug_log(
        "DISCOUNT CALCULATION",
        f"Grouping by: {group} | "
        f"Relationship analysis: {relationship}",
    )

    if (
        df.empty
        or not df.columns.is_unique
        or group not in df.columns
    ):
        debug_log(
            "DISCOUNT CALCULATION",
            "Required grouping column unavailable",
        )

        return None

    labels = df[group]

    if labels.isna().any() or not labels.map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        debug_log(
            "DISCOUNT CALCULATION",
            "Invalid grouping labels",
        )

        return None

    validated = validate_discount_values(df)

    if validated is None:
        return None

    discounts, assumption = validated

    data = pd.DataFrame({
        group: labels,
        "Average_Discount_Pct": discounts,
    })

    averages = (
        data.groupby(group, observed=True)[
            "Average_Discount_Pct"
        ]
        .mean()
        .reset_index()
    )

    debug_log(
        "DISCOUNT CALCULATION",
        f"Average discounts calculated for "
        f"{len(averages):,} groups",
    )

    if not relationship:
        timer.finish("Average discount calculation completed")

        return averages, assumption

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    if grouped is None:
        return None

    grouped = calculate_profit_margin(grouped)

    data["Profit"] = pd.to_numeric(
        df["Profit"],
        errors="coerce",
    )

    correlations = []

    debug_log(
        "DISCOUNT CORRELATION",
        "Calculating Pearson correlations",
    )

    for label, subset in data.groupby(
        group,
        observed=True,
    ):
        paired = subset[
            ["Average_Discount_Pct", "Profit"]
        ].dropna()

        correlation = None

        if (
            len(paired) >= 2
            and paired["Average_Discount_Pct"].nunique() > 1
            and paired["Profit"].nunique() > 1
        ):
            calculated = paired[
                "Average_Discount_Pct"
            ].corr(
                paired["Profit"],
                method="pearson",
            )

            if pd.notna(calculated) and math.isfinite(
                calculated
            ):
                correlation = max(
                    -1.0,
                    min(1.0, float(calculated)),
                )

        debug_log(
            "DISCOUNT CORRELATION",
            f"{label} | Paired observations: "
            f"{len(paired):,} | "
            f"Correlation: {correlation}",
        )

        correlations.append({
            group: label,
            "Discount_Profit_Correlation": correlation,
        })

    grouped = grouped.merge(
        averages,
        on=group,
        how="left",
        validate="one_to_one",
    )

    grouped = grouped.merge(
        pd.DataFrame(correlations),
        on=group,
        how="left",
        validate="one_to_one",
    )

    grouped = grouped.rename(columns={
        "Sales": "Total_Sales",
        "Profit": "Total_Profit",
    })

    grouped = grouped[[
        group,
        "Total_Sales",
        "Total_Profit",
        "Profit_Margin_Pct",
        "Average_Discount_Pct",
        "Discount_Profit_Correlation",
    ]]

    debug_log(
        "DISCOUNT CALCULATION",
        f"Completed financial metrics for "
        f"{len(grouped):,} groups",
    )

    timer.finish("Discount relationship calculation completed")

    return grouped, assumption


def validate_discount_values(df):
    """Validate discount values and identify their representation."""

    timer = DebugTimer("DISCOUNT VALIDATION")

    debug_log(
        "DISCOUNT VALIDATION",
        "Checking available discount columns",
    )

    supported_columns = {
        "Discount": "infer",
        "Discount_Pct": "percentage",
        "Discount Percentage": "percentage",
        "Discount %": "percentage",
        "Discount_Ratio": "fraction",
        "Discount Ratio": "fraction",
    }

    available = [
        column
        for column in supported_columns
        if column in df.columns
    ]

    if len(available) != 1 or not df.columns.is_unique:
        debug_log(
            "DISCOUNT VALIDATION",
            "Missing or ambiguous discount columns",
        )

        return None

    column = available[0]
    unit = supported_columns[column]

    raw = df[column]

    if raw.map(
        lambda value: isinstance(value, bool)
    ).any():
        debug_log(
            "DISCOUNT VALIDATION",
            "Boolean discount values are unsupported",
        )

        return None

    values = pd.to_numeric(
        raw,
        errors="coerce",
    )

    invalid = raw.notna() & values.isna()

    if invalid.any():
        debug_log(
            "DISCOUNT VALIDATION",
            "Invalid nonnumeric discount values detected",
        )

        return None

    valid = values.dropna()

    if valid.empty:
        debug_log(
            "DISCOUNT VALIDATION",
            "No valid discount observations",
        )

        return None

    if not valid.map(math.isfinite).all():
        debug_log(
            "DISCOUNT VALIDATION",
            "Nonfinite discount values detected",
        )

        return None

    minimum = float(valid.min())
    maximum = float(valid.max())

    debug_log(
        "DISCOUNT VALIDATION",
        f"Column: {column} | Minimum: {minimum} | "
        f"Maximum: {maximum}",
    )

    if minimum < 0 or maximum > 100:
        debug_log(
            "DISCOUNT VALIDATION",
            "Discount values outside supported range",
        )

        return None

    assumption = None

    if unit == "infer":
        if maximum <= 1:
            unit = "fraction"

            if maximum > 0:
                assumption = (
                    "Discount values between 0 and 1 "
                    "were interpreted as fractional "
                    "discounts and converted to percentages."
                )

        elif ((valid > 0) & (valid < 1)).any():
            debug_log(
                "DISCOUNT VALIDATION",
                "Potentially mixed discount representations",
            )

            return None

        else:
            unit = "percentage"

            assumption = (
                "Discount values were interpreted "
                "as percentage points."
            )

    if unit == "fraction":
        if maximum > 1:
            debug_log(
                "DISCOUNT VALIDATION",
                "Fractional discounts exceed 1",
            )

            return None

        values = values * 100

    debug_log(
        "DISCOUNT VALIDATION",
        f"Discount representation: {unit}",
    )

    debug_log(
        "DISCOUNT VALIDATION",
        f"Valid observations: {len(valid):,} | "
        f"Missing observations: {values.isna().sum():,}",
    )

    timer.finish("Discount validation completed")

    return values, assumption


def calculate_discount_metrics(df, group, relationship=False):
    """Calculate grouped discounts and optional financial metrics."""

    timer = DebugTimer("DISCOUNT CALCULATION")

    debug_log(
        "DISCOUNT CALCULATION",
        f"Grouping by: {group} | "
        f"Relationship analysis: {relationship}",
    )

    if (
        df.empty
        or not df.columns.is_unique
        or group not in df.columns
    ):
        debug_log(
            "DISCOUNT CALCULATION",
            "Required grouping column unavailable",
        )

        return None

    labels = df[group]

    if labels.isna().any() or not labels.map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        debug_log(
            "DISCOUNT CALCULATION",
            "Invalid grouping labels",
        )

        return None

    validated = validate_discount_values(df)

    if validated is None:
        return None

    discounts, assumption = validated

    data = pd.DataFrame({
        group: labels,
        "Average_Discount_Pct": discounts,
    })

    averages = (
        data.groupby(group, observed=True)[
            "Average_Discount_Pct"
        ]
        .mean()
        .reset_index()
    )

    debug_log(
        "DISCOUNT CALCULATION",
        f"Average discounts calculated for "
        f"{len(averages):,} groups",
    )

    if not relationship:
        timer.finish("Average discount calculation completed")

        return averages, assumption

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    if grouped is None:
        return None

    grouped = calculate_profit_margin(grouped)

    data["Profit"] = pd.to_numeric(
        df["Profit"],
        errors="coerce",
    )

    correlations = []

    debug_log(
        "DISCOUNT CORRELATION",
        "Calculating Pearson correlations",
    )

    for label, subset in data.groupby(
        group,
        observed=True,
    ):
        paired = subset[
            ["Average_Discount_Pct", "Profit"]
        ].dropna()

        correlation = None

        if (
            len(paired) >= 2
            and paired["Average_Discount_Pct"].nunique() > 1
            and paired["Profit"].nunique() > 1
        ):
            calculated = paired[
                "Average_Discount_Pct"
            ].corr(
                paired["Profit"],
                method="pearson",
            )

            if pd.notna(calculated) and math.isfinite(
                calculated
            ):
                correlation = max(
                    -1.0,
                    min(1.0, float(calculated)),
                )

        debug_log(
            "DISCOUNT CORRELATION",
            f"{label} | Paired observations: "
            f"{len(paired):,} | "
            f"Correlation: {correlation}",
        )

        correlations.append({
            group: label,
            "Discount_Profit_Correlation": correlation,
        })

    grouped = grouped.merge(
        averages,
        on=group,
        how="left",
        validate="one_to_one",
    )

    grouped = grouped.merge(
        pd.DataFrame(correlations),
        on=group,
        how="left",
        validate="one_to_one",
    )

    grouped = grouped.rename(columns={
        "Sales": "Total_Sales",
        "Profit": "Total_Profit",
    })

    grouped = grouped[[
        group,
        "Total_Sales",
        "Total_Profit",
        "Profit_Margin_Pct",
        "Average_Discount_Pct",
        "Discount_Profit_Correlation",
    ]]

    debug_log(
        "DISCOUNT CALCULATION",
        f"Completed financial metrics for "
        f"{len(grouped):,} groups",
    )

    timer.finish("Discount relationship calculation completed")

    return grouped, assumption


def discount_analysis_result(df, group, relationship=False):
    """Return a deterministic discount analysis response."""

    calculated = calculate_discount_metrics(
        df,
        group,
        relationship,
    )

    if calculated is None:
        return None

    grouped, assumption = calculated

    grouped = grouped.sort_values(
        group,
        kind="stable",
    )

    lines = [
        f"Discount analysis by {group.lower()}:",
        "",
    ]

    def append_extreme(column, label, highest=True):
        available = grouped.loc[
            grouped[column].notna()
        ]

        if available.empty:
            lines.append(
                f"- **{label}:** Unavailable."
            )

            return

        value = (
            available[column].max()
            if highest
            else available[column].min()
        )

        matches = available.loc[
            available[column] == value
        ]

        names = ", ".join(
            f"**{escape_text(name)}**"
            for name in matches[group].head(5)
        )

        if len(matches) > 5:
            names += (
                f" and {len(matches) - 5:,} others"
            )

        formatted = format_named_value(
            column,
            float(value),
        )

        lines.append(
            f"- **{label}:** {names} "
            f"({formatted})."
        )

    append_extreme(
        "Average_Discount_Pct",
        "Highest average discount",
    )

    if relationship:
        append_extreme(
            "Profit_Margin_Pct",
            "Lowest profit margin",
            highest=False,
        )

        correlations = grouped[
            "Discount_Profit_Correlation"
        ].dropna()

        if correlations.empty:
            lines.append(
                "- **Discount-profit correlation:** "
                "Unavailable for all groups."
            )

        else:
            if correlations.min() < 0:
                append_extreme(
                    "Discount_Profit_Correlation",
                    "Most negative discount-profit correlation",
                    highest=False,
                )

            if correlations.max() > 0:
                append_extreme(
                    "Discount_Profit_Correlation",
                    "Most positive discount-profit correlation",
                )

            if (correlations == 0).all():
                lines.append(
                    "\nAll available correlations are zero."
                )

            if (correlations < 0).all():
                lines.append(
                    "\nAll groups with available correlations "
                    "show a negative association between "
                    "discount and profit."
                )

        unavailable = int(
            grouped[
                "Discount_Profit_Correlation"
            ].isna().sum()
        )

        if unavailable:
            lines.append(
                f"\nCorrelation was unavailable for "
                f"{unavailable:,} group(s) because of "
                "insufficient observations, constant values "
                "or undefined calculations."
            )

        lines.append(
            "\nCorrelation does not "
            "establish causation."
        )

    else:
        append_extreme(
            "Average_Discount_Pct",
            "Lowest average discount",
            highest=False,
        )

    if assumption:
        lines.append(
            f"\n{assumption}"
        )

    missing_discounts = int(
        grouped["Average_Discount_Pct"].isna().sum()
    )

    if missing_discounts:
        lines.append(
            f"\nAverage discount was unavailable for "
            f"{missing_discounts:,} group(s)."
        )

    answer = append_display_limit(
        "\n".join(lines),
        len(grouped),
    )

    debug_log(
        "DISCOUNT RESULT",
        f"Summary generated | Groups: {len(grouped):,}",
    )

    return financial_response(
        answer,
        prepare_result_rows(grouped),
    )



def try_financial_analysis(df, question):
    """Handle supported financial intents or allow AI fallback."""

    timer = DebugTimer("FINANCIAL")

    debug_log(
        "FINANCIAL",
        "Starting deterministic financial intent resolution",
    )

    intent = resolve_financial_intent(question)

    if intent is None:
        debug_log(
            "FINANCIAL",
            "No supported financial intent matched",
        )

        return None

    debug_log(
        "FINANCIAL INTENT",
        str(intent),
    )

    group = intent["group"]
    kind = intent["kind"]

    if kind == "discount_relationship":
        result = discount_analysis_result(
            df,
            group,
            relationship=True,
        )

    elif kind == "average_discount":
        result = discount_analysis_result(
            df,
            group,
            relationship=False,
        )

    elif kind == "above_average_loss":
        result = above_average_loss_result(
            df,
            group,
        )

    elif kind == "margin_threshold":
        result = margin_threshold_result(
            df,
            group,
            intent["direction"],
            intent["threshold"],
        )

    elif kind == "money_threshold":
        result = money_threshold_result(
            df,
            group,
            intent["metric"],
            intent["direction"],
            intent["threshold"],
        )

    else:
        return None

    if result is None:
        debug_log(
            "FINANCIAL",
            "Financial calculation unavailable; allowing AI fallback",
        )

    else:
        timer.finish(
            "Deterministic financial analysis completed"
        )

    return result