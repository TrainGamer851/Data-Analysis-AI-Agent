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


# Return a readable singular or plural group count.
def group_count_text(group, count):

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

    # Return the selected result when `count == 0`.
    if count == 0:
        return f"No {plural}"

    # Return the selected result when `count == 1`.
    if count == 1:
        return f"1 {singular}"

    return f"{count:,} {plural}"


# Normalize wording while preserving financial symbols.
def normalize_financial_question(question):

    question = question.lower().strip()

    question = question.replace("sub-categor", "subcategor")
    question = question.replace("above-average", "above average")
    question = question.replace("below-average", "below average")

    question = question.rstrip("?.!")

    return " ".join(question.split())


# Recognize supported standalone financial questions.
def resolve_financial_intent(question):

    q = normalize_financial_question(question)

    # Discount and profitability relationships.
    match = re.fullmatch(
        r"(?:analyze|analyse|show) the relationship "
        r"between discount and (?:profitability|profit) "
        rf"(?:across|by) ({GROUP_PATTERN})",
        q,
    )

    # Return the selected result when `match`.
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

    # Return the selected result when `match`.
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

    # Return the selected result when `match`.
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

    # Update `(group, direction, threshold)` when `match`.
    if match:
        group, direction, threshold = match.groups()

        threshold = float(threshold)

        # Return `None` when `not math.isfinite(threshold)`.
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

    # Update `(group, metric, direction, threshold)` when `match`.
    if match:
        group, metric, direction, threshold = match.groups()

        threshold = float(threshold.replace("$", "").replace(",", ""))

        # Return `None` when `not math.isfinite(threshold)`.
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


# Calculate validated group totals without modifying input.
def calculate_grouped_metrics(df, group, metrics):

    timer = DebugTimer("FINANCIAL CALCULATION")

    debug_log(
        "FINANCIAL CALCULATION",
        f"Grouping by: {group} | Metrics: {metrics}",
    )

    required = [group, *metrics]

    # Call `debug_log` when `df.empty or not df.columns.is_unique or (not set(required).issubset(df.columns))`.
    if df.empty or not df.columns.is_unique or not set(required).issubset(df.columns):
        debug_log(
            "FINANCIAL VALIDATION",
            "Required financial columns unavailable",
        )

        return None

    data = df[required].copy()

    # Call `debug_log` when `data[group].isna().any()`.
    if data[group].isna().any():
        debug_log(
            "FINANCIAL VALIDATION",
            "Missing grouping values detected",
        )

        return None

    # Call `debug_log` when `not data[group].map(lambda value: isinstance(value, str) and bool(value.strip())).all()`.
    if (
        not data[group]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        debug_log(
            "FINANCIAL VALIDATION",
            "Unsupported grouping values",
        )

        return None

    # Process each `metric` from `metrics`.
    for metric in metrics:
        # Return `None` when `data[metric].map(lambda value: isinstance(value, bool)).any()`.
        if data[metric].map(lambda value: isinstance(value, bool)).any():
            return None

        data[metric] = pd.to_numeric(
            data[metric],
            errors="coerce",
        )

        # Call `debug_log` when `data[metric].isna().any() or not data[metric].map(math.isfinite).all()`.
        if data[metric].isna().any() or not data[metric].map(math.isfinite).all():
            debug_log(
                "FINANCIAL VALIDATION",
                f"Invalid numeric values in {metric}",
            )

            return None

    grouped = data.groupby(group, observed=True)[metrics].sum().reset_index()

    # Return `None` when `grouped.empty`.
    if grouped.empty:
        return None

    # Process each `metric` from `metrics`.
    for metric in metrics:
        # Return `None` when `not grouped[metric].map(math.isfinite).all()`.
        if not grouped[metric].map(math.isfinite).all():
            return None

    debug_log(
        "FINANCIAL CALCULATION",
        f"Calculated groups: {len(grouped)}",
    )

    # Process each `row` from `grouped.head(30).to_dict('records')`.
    for row in grouped.head(30).to_dict("records"):
        debug_log(
            "FINANCIAL GROUP",
            str(row),
        )

    timer.finish("Grouped financial totals calculated")

    return grouped


# Calculate margins using total profit divided by total sales.
def calculate_profit_margin(grouped):

    grouped = grouped.copy()

    valid_sales = grouped["Sales"].where(grouped["Sales"] > 0)

    grouped["Profit_Margin_Pct"] = grouped["Profit"] / valid_sales * 100

    grouped["Profit_Margin_Pct"] = grouped["Profit_Margin_Pct"].replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    debug_log(
        "FINANCIAL CALCULATION",
        "Profit margins calculated",
    )

    return grouped


# Apply a strict upper or lower threshold.
def threshold_mask(values, direction, threshold):

    # Return the selected result when `direction in {'above', 'over'}`.
    if direction in {"above", "over"}:
        return values > threshold

    return values < threshold


# Return the existing agent response structure.
def financial_response(answer, rows=None):

    return {
        "answer": answer,
        "table": rows,
        "error": None,
    }


# Convert calculated values into frontend-compatible records.
def prepare_result_rows(grouped):

    records = []

    # Process each `row` from `grouped.head(MAX_RESULT_ROWS).to_dict('records')`.
    for row in grouped.head(MAX_RESULT_ROWS).to_dict("records"):
        record = {}

        # Process each `(column, value)` from `row.items()`.
        for column, value in row.items():
            # Update `record[column]` when `pd.isna(value)`.
            if pd.isna(value):
                record[column] = None

            # Update `record[column]` when `isinstance(value, str)`.
            elif isinstance(value, str):
                record[column] = value

            # Update `record[column]` when `'correlation' in column.lower()`.
            elif "correlation" in column.lower():
                record[column] = round(float(value), 4)

            # Update `record[column]` when no earlier condition matches.
            else:
                record[column] = round(float(value), 2)

        records.append(record)

    debug_log(
        "FINANCIAL RESULT",
        f"Supporting rows prepared: {len(records)}",
    )

    return records


# Disclose when supporting results exceed the display limit.
def append_display_limit(answer, count):

    # Update `answer` when `count > MAX_RESULT_ROWS`.
    if count > MAX_RESULT_ROWS:
        answer += (
            f"\n\nShowing the first {MAX_RESULT_ROWS:,} "
            f"of {count:,} matching groups."
        )

    return answer


# Find groups with above-average total sales and negative profit.
def above_average_loss_result(df, group):

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    # Return `None` when `grouped is None`.
    if grouped is None:
        return None

    average_sales = float(grouped["Sales"].mean())

    # Return `None` when `not math.isfinite(average_sales)`.
    if not math.isfinite(average_sales):
        return None

    debug_log(
        "FINANCIAL FILTER",
        f"Average group sales threshold: {average_sales:,.6f}",
    )

    matches = grouped.loc[
        (grouped["Sales"] > average_sales) & (grouped["Profit"] < 0)
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

    # Return the selected result when `matches.empty`.
    if matches.empty:
        return financial_response(
            "No groups have both above-average total "
            "sales and negative total profit.\n\n"
            f"The average group sales threshold is "
            f"**{threshold_text}**."
        )

    matches = calculate_profit_margin(matches)

    matches["Average_Sales_Threshold"] = average_sales

    matches = matches.rename(
        columns={
            "Sales": "Total_Sales",
            "Profit": "Total_Profit",
        }
    )

    answer = (
        f"**{group_count_text(group, count)}** "
        f"have total sales above "
        f"the average group sales threshold of "
        f"**{threshold_text}** and negative total profit."
    )

    # Update `row` when `count == 1`.
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

        # Update `answer` when `pd.notna(row['Profit_Margin_Pct'])`.
        if pd.notna(row["Profit_Margin_Pct"]):
            answer += f"\n- **Profit margin:** " f"{row['Profit_Margin_Pct']:.2f}%"

    return financial_response(
        append_display_limit(answer, count),
        prepare_result_rows(matches),
    )


# Find groups satisfying a profit margin threshold.
def margin_threshold_result(df, group, direction, threshold):

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    # Return `None` when `grouped is None`.
    if grouped is None:
        return None

    grouped = calculate_profit_margin(grouped)

    unavailable = int(grouped["Profit_Margin_Pct"].isna().sum())

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

    # Update `row` when `count == 1`.
    if count == 1:
        row = matches.iloc[0]

        answer = (
            f"**{escape_text(row[group])}** has a profit "
            f"margin of **{row['Profit_Margin_Pct']:.2f}%**, "
            f"which is {direction} **{threshold:g}%**."
        )

    # Update `answer` when `unavailable`.
    if unavailable:
        answer += (
            f"\n\nProfit margins were unavailable for "
            f"{unavailable:,} group(s) with nonpositive "
            "sales or nonfinite calculated margins."
        )

    matches = matches.rename(
        columns={
            "Sales": "Total_Sales",
            "Profit": "Total_Profit",
        }
    )

    return financial_response(
        append_display_limit(answer, count),
        prepare_result_rows(matches) if count else None,
    )


# Find groups satisfying a monetary threshold.
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

    # Return `None` when `grouped is None`.
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

    # Update `lines` when `count`.
    if count:
        lines = [answer, ""]

        # Process each `(_, row)` from `matches.head(5).iterrows()`.
        for _, row in matches.head(5).iterrows():
            lines.append(
                f"- **{escape_text(row[group])}:** "
                f"{format_named_value(metric, row[metric])}"
            )

        answer = "\n".join(lines)

        # Update `answer` when `count > 5`.
        if count > 5:
            answer += (
                "\n\nAdditional matching groups " "are shown in the supporting data."
            )

    matches = matches.rename(
        columns={
            metric: f"Total_{metric}",
        }
    )

    return financial_response(
        append_display_limit(answer, count),
        prepare_result_rows(matches) if count else None,
    )


# Calculate grouped discounts and optional financial metrics.
def calculate_discount_metrics(df, group, relationship=False):

    timer = DebugTimer("DISCOUNT CALCULATION")

    debug_log(
        "DISCOUNT CALCULATION",
        f"Grouping by: {group} | " f"Relationship analysis: {relationship}",
    )

    # Call `debug_log` when `df.empty or not df.columns.is_unique or group not in df.columns`.
    if df.empty or not df.columns.is_unique or group not in df.columns:
        debug_log(
            "DISCOUNT CALCULATION",
            "Required grouping column unavailable",
        )

        return None

    labels = df[group]

    # Call `debug_log` when `labels.isna().any() or not labels.map(lambda value: isinstance(value, str) and...`.
    if (
        labels.isna().any()
        or not labels.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        debug_log(
            "DISCOUNT CALCULATION",
            "Invalid grouping labels",
        )

        return None

    validated = validate_discount_values(df)

    # Return `None` when `validated is None`.
    if validated is None:
        return None

    discounts, assumption = validated

    data = pd.DataFrame(
        {
            group: labels,
            "Average_Discount_Pct": discounts,
        }
    )

    averages = (
        data.groupby(group, observed=True)["Average_Discount_Pct"].mean().reset_index()
    )

    debug_log(
        "DISCOUNT CALCULATION",
        f"Average discounts calculated for " f"{len(averages):,} groups",
    )

    # Call `timer.finish` when `not relationship`.
    if not relationship:
        timer.finish("Average discount calculation completed")

        return averages, assumption

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    # Return `None` when `grouped is None`.
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

    # Process each `(label, subset)` from `data.groupby(group, observed=True)`.
    for label, subset in data.groupby(
        group,
        observed=True,
    ):
        paired = subset[["Average_Discount_Pct", "Profit"]].dropna()

        correlation = None

        # Update `calculated` when `len(paired) >= 2 and paired['Average_Discount_Pct'].nunique() > 1 and...`.
        if (
            len(paired) >= 2
            and paired["Average_Discount_Pct"].nunique() > 1
            and paired["Profit"].nunique() > 1
        ):
            calculated = paired["Average_Discount_Pct"].corr(
                paired["Profit"],
                method="pearson",
            )

            # Update `correlation` when `pd.notna(calculated) and math.isfinite(calculated)`.
            if pd.notna(calculated) and math.isfinite(calculated):
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

        correlations.append(
            {
                group: label,
                "Discount_Profit_Correlation": correlation,
            }
        )

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

    grouped = grouped.rename(
        columns={
            "Sales": "Total_Sales",
            "Profit": "Total_Profit",
        }
    )

    grouped = grouped[
        [
            group,
            "Total_Sales",
            "Total_Profit",
            "Profit_Margin_Pct",
            "Average_Discount_Pct",
            "Discount_Profit_Correlation",
        ]
    ]

    debug_log(
        "DISCOUNT CALCULATION",
        f"Completed financial metrics for " f"{len(grouped):,} groups",
    )

    timer.finish("Discount relationship calculation completed")

    return grouped, assumption


# Validate discount values and identify their representation.
def validate_discount_values(df):

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

    available = [column for column in supported_columns if column in df.columns]

    # Call `debug_log` when `len(available) != 1 or not df.columns.is_unique`.
    if len(available) != 1 or not df.columns.is_unique:
        debug_log(
            "DISCOUNT VALIDATION",
            "Missing or ambiguous discount columns",
        )

        return None

    column = available[0]
    unit = supported_columns[column]

    raw = df[column]

    # Call `debug_log` when `raw.map(lambda value: isinstance(value, bool)).any()`.
    if raw.map(lambda value: isinstance(value, bool)).any():
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

    # Call `debug_log` when `invalid.any()`.
    if invalid.any():
        debug_log(
            "DISCOUNT VALIDATION",
            "Invalid nonnumeric discount values detected",
        )

        return None

    valid = values.dropna()

    # Call `debug_log` when `valid.empty`.
    if valid.empty:
        debug_log(
            "DISCOUNT VALIDATION",
            "No valid discount observations",
        )

        return None

    # Call `debug_log` when `not valid.map(math.isfinite).all()`.
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
        f"Column: {column} | Minimum: {minimum} | " f"Maximum: {maximum}",
    )

    # Call `debug_log` when `minimum < 0 or maximum > 100`.
    if minimum < 0 or maximum > 100:
        debug_log(
            "DISCOUNT VALIDATION",
            "Discount values outside supported range",
        )

        return None

    assumption = None

    # Evaluate the nested condition when `unit == 'infer'`.
    if unit == "infer":
        # Update `unit` when `maximum <= 1`.
        if maximum <= 1:
            unit = "fraction"

            # Update `assumption` when `maximum > 0`.
            if maximum > 0:
                assumption = (
                    "Discount values between 0 and 1 "
                    "were interpreted as fractional "
                    "discounts and converted to percentages."
                )

        # Call `debug_log` when `((valid > 0) & (valid < 1)).any()`.
        elif ((valid > 0) & (valid < 1)).any():
            debug_log(
                "DISCOUNT VALIDATION",
                "Potentially mixed discount representations",
            )

            return None

        # Update `unit` when no earlier condition matches.
        else:
            unit = "percentage"

            assumption = "Discount values were interpreted " "as percentage points."

    # Evaluate the nested condition when `unit == 'fraction'`.
    if unit == "fraction":
        # Call `debug_log` when `maximum > 1`.
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


# Calculate grouped discounts and optional financial metrics.
def calculate_discount_metrics(df, group, relationship=False):

    timer = DebugTimer("DISCOUNT CALCULATION")

    debug_log(
        "DISCOUNT CALCULATION",
        f"Grouping by: {group} | " f"Relationship analysis: {relationship}",
    )

    # Call `debug_log` when `df.empty or not df.columns.is_unique or group not in df.columns`.
    if df.empty or not df.columns.is_unique or group not in df.columns:
        debug_log(
            "DISCOUNT CALCULATION",
            "Required grouping column unavailable",
        )

        return None

    labels = df[group]

    # Call `debug_log` when `labels.isna().any() or not labels.map(lambda value: isinstance(value, str) and...`.
    if (
        labels.isna().any()
        or not labels.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        debug_log(
            "DISCOUNT CALCULATION",
            "Invalid grouping labels",
        )

        return None

    validated = validate_discount_values(df)

    # Return `None` when `validated is None`.
    if validated is None:
        return None

    discounts, assumption = validated

    data = pd.DataFrame(
        {
            group: labels,
            "Average_Discount_Pct": discounts,
        }
    )

    averages = (
        data.groupby(group, observed=True)["Average_Discount_Pct"].mean().reset_index()
    )

    debug_log(
        "DISCOUNT CALCULATION",
        f"Average discounts calculated for " f"{len(averages):,} groups",
    )

    # Call `timer.finish` when `not relationship`.
    if not relationship:
        timer.finish("Average discount calculation completed")

        return averages, assumption

    grouped = calculate_grouped_metrics(
        df,
        group,
        ["Sales", "Profit"],
    )

    # Return `None` when `grouped is None`.
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

    # Process each `(label, subset)` from `data.groupby(group, observed=True)`.
    for label, subset in data.groupby(
        group,
        observed=True,
    ):
        paired = subset[["Average_Discount_Pct", "Profit"]].dropna()

        correlation = None

        # Update `calculated` when `len(paired) >= 2 and paired['Average_Discount_Pct'].nunique() > 1 and...`.
        if (
            len(paired) >= 2
            and paired["Average_Discount_Pct"].nunique() > 1
            and paired["Profit"].nunique() > 1
        ):
            calculated = paired["Average_Discount_Pct"].corr(
                paired["Profit"],
                method="pearson",
            )

            # Update `correlation` when `pd.notna(calculated) and math.isfinite(calculated)`.
            if pd.notna(calculated) and math.isfinite(calculated):
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

        correlations.append(
            {
                group: label,
                "Discount_Profit_Correlation": correlation,
            }
        )

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

    grouped = grouped.rename(
        columns={
            "Sales": "Total_Sales",
            "Profit": "Total_Profit",
        }
    )

    grouped = grouped[
        [
            group,
            "Total_Sales",
            "Total_Profit",
            "Profit_Margin_Pct",
            "Average_Discount_Pct",
            "Discount_Profit_Correlation",
        ]
    ]

    debug_log(
        "DISCOUNT CALCULATION",
        f"Completed financial metrics for " f"{len(grouped):,} groups",
    )

    timer.finish("Discount relationship calculation completed")

    return grouped, assumption


# Return a deterministic discount analysis response.
def discount_analysis_result(df, group, relationship=False):

    calculated = calculate_discount_metrics(
        df,
        group,
        relationship,
    )

    # Return `None` when `calculated is None`.
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

    # Add the extreme-value result to the metric summary when available.
    def append_extreme(column, label, highest=True):
        available = grouped.loc[grouped[column].notna()]

        # Append the value to `lines` when `available.empty`.
        if available.empty:
            lines.append(f"- **{label}:** Unavailable.")

            return

        value = available[column].max() if highest else available[column].min()

        matches = available.loc[available[column] == value]

        names = ", ".join(f"**{escape_text(name)}**" for name in matches[group].head(5))

        # Update `names` when `len(matches) > 5`.
        if len(matches) > 5:
            names += f" and {len(matches) - 5:,} others"

        formatted = format_named_value(
            column,
            float(value),
        )

        lines.append(f"- **{label}:** {names} " f"({formatted}).")

    append_extreme(
        "Average_Discount_Pct",
        "Highest average discount",
    )

    # Call `append_extreme` when `relationship`.
    if relationship:
        append_extreme(
            "Profit_Margin_Pct",
            "Lowest profit margin",
            highest=False,
        )

        correlations = grouped["Discount_Profit_Correlation"].dropna()

        # Append the value to `lines` when `correlations.empty`.
        if correlations.empty:
            lines.append(
                "- **Discount-profit correlation:** " "Unavailable for all groups."
            )

        # Evaluate the nested condition when no earlier condition matches.
        else:
            # Call `append_extreme` when `correlations.min() < 0`.
            if correlations.min() < 0:
                append_extreme(
                    "Discount_Profit_Correlation",
                    "Most negative discount-profit correlation",
                    highest=False,
                )

            # Call `append_extreme` when `correlations.max() > 0`.
            if correlations.max() > 0:
                append_extreme(
                    "Discount_Profit_Correlation",
                    "Most positive discount-profit correlation",
                )

            # Append the value to `lines` when `(correlations == 0).all()`.
            if (correlations == 0).all():
                lines.append("\nAll available correlations are zero.")

            # Append the value to `lines` when `(correlations < 0).all()`.
            if (correlations < 0).all():
                lines.append(
                    "\nAll groups with available correlations "
                    "show a negative association between "
                    "discount and profit."
                )

        unavailable = int(grouped["Discount_Profit_Correlation"].isna().sum())

        # Append the value to `lines` when `unavailable`.
        if unavailable:
            lines.append(
                f"\nCorrelation was unavailable for "
                f"{unavailable:,} group(s) because of "
                "insufficient observations, constant values "
                "or undefined calculations."
            )

        lines.append("\nCorrelation does not " "establish causation.")

    # Call `append_extreme` when no earlier condition matches.
    else:
        append_extreme(
            "Average_Discount_Pct",
            "Lowest average discount",
            highest=False,
        )

    # Append the value to `lines` when `assumption`.
    if assumption:
        lines.append(f"\n{assumption}")

    missing_discounts = int(grouped["Average_Discount_Pct"].isna().sum())

    # Append the value to `lines` when `missing_discounts`.
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


# Handle supported financial intents or allow AI fallback.
def try_financial_analysis(df, question):

    timer = DebugTimer("FINANCIAL")

    debug_log(
        "FINANCIAL",
        "Starting deterministic financial intent resolution",
    )

    intent = resolve_financial_intent(question)

    # Call `debug_log` when `intent is None`.
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

    # Update `result` when `kind == 'discount_relationship'`.
    if kind == "discount_relationship":
        result = discount_analysis_result(
            df,
            group,
            relationship=True,
        )

    # Update `result` when `kind == 'average_discount'`.
    elif kind == "average_discount":
        result = discount_analysis_result(
            df,
            group,
            relationship=False,
        )

    # Update `result` when `kind == 'above_average_loss'`.
    elif kind == "above_average_loss":
        result = above_average_loss_result(
            df,
            group,
        )

    # Update `result` when `kind == 'margin_threshold'`.
    elif kind == "margin_threshold":
        result = margin_threshold_result(
            df,
            group,
            intent["direction"],
            intent["threshold"],
        )

    # Update `result` when `kind == 'money_threshold'`.
    elif kind == "money_threshold":
        result = money_threshold_result(
            df,
            group,
            intent["metric"],
            intent["direction"],
            intent["threshold"],
        )

    # Return `None` when no earlier condition matches.
    else:
        return None

    # Call `debug_log` when `result is None`.
    if result is None:
        debug_log(
            "FINANCIAL",
            "Financial calculation unavailable; allowing AI fallback",
        )

    # Call `timer.finish` when no earlier condition matches.
    else:
        timer.finish("Deterministic financial analysis completed")

    return result
