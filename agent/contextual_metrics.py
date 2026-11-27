"""Deterministic financial analysis and contextual follow-ups."""

import math
import re

import pandas as pd

from agent.debug_logger import debug_log, DebugTimer

from agent.annual_metrics import (
    parse_order_dates,
    normalize_question,
    money,
    response,
)

from agent.conversation_context import (
    build_conversation_context,
)


GROUPS = {
    "region": "Region",
    "category": "Category",
    "year": "Year",
}

METRICS = {
    "sales": "Sales",
    "profit": "Profit",
}

ANNUAL_REQUESTS = {
    "show annual sales and profit by year",
    "show annual sales and profit",
}


def grouped_totals(df, group, metric):
    """Calculate validated totals without modifying the dataset."""

    timer = DebugTimer("CALCULATION")

    debug_log(
        "CALCULATION",
        f"Calculating total {metric} by {group}",
    )

    debug_log(
        "CALCULATION",
        f"Input dataset rows: {len(df):,}",
    )

    if df.empty or metric not in df.columns:
        debug_log(
            "VALIDATION",
            f"Dataset empty or missing metric: {metric}",
        )

        return None

    values = pd.to_numeric(
        df[metric],
        errors="coerce",
    )

    if (
        values.isna().any()
        or not values.map(math.isfinite).all()
    ):
        return None

    if group == "Year":
        if "Order Date" not in df.columns:
            return None

        dates = parse_order_dates(df["Order Date"])

        if dates is None or dates.isna().any():
            return None

        labels = dates.dt.year.astype(int)

    else:
        if group not in df.columns:
            return None

        labels = df[group]

        if labels.isna().any():
            return None

    debug_log(
        "CALCULATION",
        f"Grouping {metric} values by {group}",
    )

    totals = values.groupby(labels).sum()

    debug_log(
        "CALCULATION",
        f"Groups calculated: {len(totals)}",
    )

    if (
        totals.empty
        or not totals.map(math.isfinite).all()
    ):
        return None

    for label, value in totals.head(30).items():
        debug_log(
            "GROUP TOTAL",
            f"{label} | {metric}: {float(value):,.2f}",
        )

    if len(totals) > 30:
        debug_log(
            "GROUP TOTAL",
            f"{len(totals) - 30} additional groups omitted",
        )

    timer.finish("Grouped calculation completed")

    return totals


def ranking_result(totals, group, metric, direction):
    """Return the highest or lowest group and its value."""

    value = (
        totals.max()
        if direction == "highest"
        else totals.min()
    )

    winners = totals.index[totals == value]

    names = ", ".join(map(str, winners))

    debug_log(
        "RANKING",
        f"Selected value: {money(value)}",
    )

    debug_log(
        "RANKING",
        f"Grouping: {group} | Metric: {metric}",
    )

    debug_log(
        "RANKING",
        f"Selected group: {names}",
    )

    debug_log(
        "RANKING",
        f"Selected value: {money(value)}",
    )

    debug_log(
        "RANKING",
        f"Matching winners: {len(winners)}",
    )

    return response(
        f"**{names}** recorded the {direction} "
        f"total {metric.lower()} by {group.lower()}, "
        f"at **{money(value)}**."
    )


def table_result(totals, group, metric, limit=None):
    """Return a structured table for the existing chart builder."""

    if group == "Year" and limit is None:
        totals = totals.sort_index()

    else:
        totals = totals.sort_values(ascending=False)

    if limit is not None:
        totals = totals.head(limit)

    debug_log(
        "TABLE",
        f"Preparing {metric} table grouped by {group}",
    )

    debug_log(
        "TABLE",
        f"Output rows: {len(totals)}",
    )

    debug_log(
        "TABLE",
        f"Requested limit: {limit}",
    )

    rows = [
        {
            group: int(label) if group == "Year" else str(label),
            f"Total {metric}": round(float(value), 2),
        }
        for label, value in totals.items()
    ]

    return response(
        f"Total {metric.lower()} by {group.lower()} "
        f"is shown below.",
        rows,
    )


def match_label(totals, name):
    """Match an actual dataset label without guessing."""

    matches = [
        label
        for label in totals.index
        if normalize_question(str(label)) == name
    ]

    return matches[0] if len(matches) == 1 else None


def unique_winner(totals, direction):
    """Resolve a singular reference only when there is one winner."""

    value = (
        totals.max()
        if direction == "highest"
        else totals.min()
    )

    winners = totals.index[totals == value]

    if len(winners) != 1:
        return None

    return winners[0]


def clarification(message):
    debug_log(
        "CLARIFICATION",
        message,
    )

    return response(
        f"**Clarification needed:** {message}"
    )


def resolve_intent(question, previous=None):
    """Resolve a supported question into a structured intent."""

    q = normalize_question(question)

    # Standalone ranking.
    match = re.fullmatch(
        r"which (region|category|year) "
        r"(?:has|had) the (highest|lowest) "
        r"(?:total )?(sales|profit)",
        q,
    )

    if match:
        group, direction, metric = match.groups()

        return {
            "kind": "ranking",
            "group": GROUPS[group],
            "metric": METRICS[metric],
            "direction": direction,
        }

    # Standalone grouped table.
    match = re.fullmatch(
        r"show (?:total )?(sales|profit) "
        r"by (region|category|year)(?: in a table)?",
        q,
    )

    if match:
        metric, group = match.groups()

        return {
            "kind": "table",
            "group": GROUPS[group],
            "metric": METRICS[metric],
        }

    # Recognize the existing annual analysis as context.
    if q in ANNUAL_REQUESTS:
        return {
            "kind": "annual",
            "group": "Year",
        }

    if previous is None:
        return None

    group = previous.get("group")
    metric = previous.get("metric")

    # Switch metrics while preserving the grouping.
    match = re.fullmatch(
        r"(?:now )?show (sales|profit) instead",
        q,
    )

    if match and group and previous["kind"] in {
        "table", "ranking", "annual", "top"
    }:
        return {
            "kind": "table",
            "group": group,
            "metric": METRICS[match.group(1)],
        }

    # Contextual ranking.
    match = re.fullmatch(
        r"which one (?:has|had) the "
        r"(highest|lowest) (?:total )?(sales|profit)",
        q,
    )

    if match and group and previous["kind"] in {
        "table", "ranking", "annual", "top"
    }:
        direction, new_metric = match.groups()

        return {
            "kind": "ranking",
            "group": group,
            "metric": METRICS[new_metric],
            "direction": direction,
        }

    # Annual ranking follow-up.
    match = re.fullmatch(
        r"which of those years had the "
        r"(highest|lowest) (?:total )?(sales|profit)",
        q,
    )

    if match and group == "Year":
        direction, new_metric = match.groups()

        return {
            "kind": "ranking",
            "group": "Year",
            "metric": METRICS[new_metric],
            "direction": direction,
        }

    # Reverse a previous ranking.
    match = re.fullmatch(
        r"what about the (highest|lowest)",
        q,
    )

    if match and group and metric:
        if previous["kind"] == "ranking":
            return {
                "kind": "ranking",
                "group": group,
                "metric": metric,
                "direction": match.group(1),
            }

    # Compare a previously ranked group against another.
    match = re.fullmatch(
        r"how much (higher|lower) is it than (.+)",
        q,
    )

    if match and previous["kind"] == "ranking":
        return {
            "kind": "comparison",
            "group": group,
            "metric": metric,
            "direction": previous["direction"],
            "reference": match.group(2),
            "percentage": False,
        }

    # Percentage comparison with an explicit baseline.
    match = re.fullmatch(
        r"what is the percentage difference relative to (.+)",
        q,
    )

    if match and previous["kind"] in {
        "ranking", "comparison"
    }:
        return {
            "kind": "comparison",
            "group": group,
            "metric": metric,
            "direction": previous["direction"],
            "reference": match.group(1),
            "percentage": True,
        }

    # Reuse the previous comparison baseline.
    if q in {
        "what is the percentage difference",
        "show me the percentage difference",
    } and previous["kind"] == "comparison":
        return {
            **previous,
            "percentage": True,
        }

    # Retrieve another metric for a previously ranked year.
    match = re.fullmatch(
        r"what was the (?:total )?(sales|profit) in that year",
        q,
    )

    if (
        match
        and group == "Year"
        and previous["kind"] == "ranking"
    ):
        return {
            "kind": "year_value",
            "group": "Year",
            "metric": METRICS[match.group(1)],
            "previous_metric": metric,
            "direction": previous["direction"],
        }

    # Change the number of displayed groups.
    match = re.fullmatch(
        r"show (?:me )?the top (\d+|three|five|ten) instead",
        q,
    )

    if match and group and metric:
        if previous["kind"] in {"ranking", "table", "top"}:
            raw_limit = match.group(1)

            words = {
                "three": 3,
                "five": 5,
                "ten": 10,
            }

            limit = (
                words[raw_limit]
                if raw_limit in words
                else int(raw_limit)
            )

            if 1 <= limit <= 30:
                return {
                    "kind": "top",
                    "group": group,
                    "metric": metric,
                    "limit": limit,
                }

    return None


def resolve_previous_context(context):
    """Reconstruct intent from completed previous questions."""

    history = build_conversation_context(context)

    debug_log(
        "CONTEXT",
        f"Reconstructing intent from {len(history)} messages",
    )

    previous = None

    for message in history:
        if message["role"] == "user":
            debug_log(
                "CONTEXT",
                f"Historical question: {message['content']}",
            )

            previous = resolve_intent(
                message["content"],
                previous,
            )

            debug_log(
                "CONTEXT",
                f"Historical intent: {previous}",
            )

        elif message["role"] == "assistant":
            if message["content"].startswith(
                "**Clarification needed:**"
            ):
                debug_log(
                    "CONTEXT",
                    "Previous clarification invalidated intent",
                )

                previous = None

    debug_log(
        "CONTEXT",
        f"Final reconstructed intent: {previous}",
    )

    return previous


def try_contextual_analysis(df, question, context=None):
    """Return a deterministic answer or None for AI fallback."""

    debug_log(
        "CONTEXTUAL",
        "Starting contextual intent resolution",
    )

    debug_log(
        "CONTEXTUAL",
        f"Current question: {question}",
    )

    previous = resolve_previous_context(context)

    intent = resolve_intent(question, previous)

    if intent is None:
        debug_log(
            "CONTEXTUAL",
            "No supported contextual intent matched",
        )

        return None

    debug_log(
        "INTENT",
        f"Resolved intent: {intent}",
    )

    kind = intent["kind"]

    debug_log(
        "INTENT",
        f"Analysis type: {kind}",
    )

    # Keep annual summaries in the existing annual module.
    if kind == "annual":
        return None

    group = intent["group"]
    metric = intent["metric"]

    debug_log(
        "INTENT",
        f"Grouping column: {group}",
    )

    debug_log(
        "INTENT",
        f"Financial metric: {metric}",
    )

    totals = grouped_totals(df, group, metric)

    if totals is None:
        debug_log(
            "CONTEXTUAL",
            "Grouped calculation unavailable; allowing fallback",
        )

        return None

    if kind == "ranking":
        return ranking_result(
            totals,
            group,
            metric,
            intent["direction"],
        )

    if kind in {"table", "top"}:
        return table_result(
            totals,
            group,
            metric,
            limit=intent.get("limit"),
        )

    if kind == "comparison":
        selected = unique_winner(
            totals,
            intent["direction"],
        )

        if selected is None:
            return clarification(
                "Multiple groups share the ranking. "
                "Which group should be compared?"
            )

        reference = match_label(
            totals,
            intent["reference"],
        )

        if reference is None:
            return clarification(
                "Which dataset group should be "
                "used as the comparison baseline?"
            )

        selected_value = float(totals.loc[selected])
        reference_value = float(totals.loc[reference])

        debug_log(
            "COMPARISON",
            f"Selected group: {selected}",
        )

        debug_log(
            "COMPARISON",
            f"Reference group: {reference}",
        )

        debug_log(
            "COMPARISON",
            f"{selected} {metric}: {selected_value:,.6f}",
        )

        debug_log(
            "COMPARISON",
            f"{reference} {metric}: {reference_value:,.6f}",
        )

        difference = selected_value - reference_value

        debug_log(
            "CALCULATION",
            f"{selected_value:,.6f} - "
            f"{reference_value:,.6f} = "
            f"{difference:,.6f}",
        )

        debug_log(
            "COMPARISON",
            f"Absolute difference: {money(abs(difference))}",
        )

        if intent["percentage"]:
            if reference_value <= 0:
                return response(
                    "A relative percentage difference "
                    "cannot be calculated using a "
                    "nonpositive comparison baseline."
                )

            percent = difference / reference_value * 100

            debug_log(
                "CALCULATION",
                f"({difference:.6f} / "
                f"{reference_value:.6f}) * 100",
            )

            debug_log(
                "COMPARISON",
                f"Relative percentage difference: {percent:+.2f}%",
            )

            return response(
                f"Relative to **{reference}**, "
                f"**{selected}** has a signed difference "
                f"of **{percent:+,.2f}%** in total "
                f"{metric.lower()}."
            )

        if difference == 0:
            return response(
                f"**{selected}** and **{reference}** "
                f"have equal total {metric.lower()}."
            )

        direction = "higher" if difference > 0 else "lower"

        return response(
            f"**{selected}**'s total {metric.lower()} "
            f"is **{money(abs(difference))} {direction}** "
            f"than **{reference}**'s."
        )

    if kind == "year_value":
        previous_totals = grouped_totals(
            df,
            "Year",
            intent["previous_metric"],
        )

        if previous_totals is None:
            return None

        year = unique_winner(
            previous_totals,
            intent["direction"],
        )

        if year is None:
            return clarification(
                "Multiple years share the ranking. "
                "Which year do you mean?"
            )

        if year not in totals.index:
            return None

        debug_log(
            "CONTEXTUAL",
            f"Resolved referenced year: {year}",
        )

        debug_log(
            "CONTEXTUAL",
            f"Previous ranking metric: "
            f"{intent['previous_metric']}",
        )

        debug_log(
            "CALCULATION",
            f"Total {metric} in {year}: "
            f"{money(totals.loc[year])}",
        )

        return response(
            f"Total {metric.lower()} in **{year}** "
            f"was **{money(totals.loc[year])}**."
        )

    return None