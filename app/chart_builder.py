"""Generate interactive charts from structured analysis results."""

import pandas as pd
import plotly.graph_objects as go

from app.table_formatter import (
    normalize_column,
    get_column_type,
    clean_column_header,
)


MAX_CHART_ROWS = 30

IGNORED_COLUMNS = {
    "index",
    "level 0",
    "level_0",
    "unnamed: 0",
    "row",
    "row number",
    "row id",
}

TIME_COLUMNS = {
    "year",
    "order year",
    "fiscal year",
    "ship year",
    "month",
    "date",
    "order date",
    "quarter",
}

DIMENSION_COLUMNS = {
    "category",
    "sub category",
    "subcategory",
    "sub-category",
    "region",
    "segment",
    "customer segment",
    "product",
    "product name",
    "country",
    "state",
    "city",
    "ship mode",
}


def build_chart(table):
    """Return a Plotly figure or None for unsupported data."""

    if table is None:
        return None

    df = pd.DataFrame(table).copy()

    if len(df) < 2 or len(df) > MAX_CHART_ROWS:
        return None

    if df.shape[1] < 2:
        return None

    time_columns = [
        column
        for column in df.columns
        if normalize_column(column) in TIME_COLUMNS
    ]

    dimension_columns = [
        column
        for column in df.columns
        if normalize_column(column) in DIMENSION_COLUMNS
    ]

    if time_columns:
        x_column = time_columns[0]
        is_time_series = True

    elif dimension_columns:
        x_column = dimension_columns[0]
        is_time_series = False

    else:
        return None

    # Avoid misleading charts for multidimensional tables.
    if len(time_columns) + len(dimension_columns) > 1:
        return None

    if df[x_column].isna().any():
        return None

    if df[x_column].duplicated().any():
        return None

    numeric_columns = []

    for column in df.columns:
        if column == x_column:
            continue

        if normalize_column(column) in IGNORED_COLUMNS:
            continue

        if get_column_type(column) == "identifier":
            continue

        if pd.api.types.is_bool_dtype(df[column]):
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].notna().any():
                numeric_columns.append(column)

    if not numeric_columns:
        return None

    # Prioritize growth metrics when the result
    # contains a calculated growth percentage.
    growth_columns = [
        column
        for column in numeric_columns
        if "growth" in normalize_column(column)
        and get_column_type(column) == "percentage"
    ]

    if growth_columns:
        metrics = growth_columns[:3]
        first_type = "percentage"

    else:
        first_type = get_column_type(
            numeric_columns[0]
        )

        metrics = [
            column
            for column in numeric_columns
            if get_column_type(column) == first_type
        ][:3]

    if is_time_series:
        df = df.sort_values(x_column)

    x_values = df[x_column].astype(str).tolist()

    figure = go.Figure()

    for column in metrics:
        label = clean_column_header(column)
        column_type = get_column_type(column)

        if column_type == "money":
            hover_format = "$%{y:,.2f}"

        elif column_type == "percentage":
            hover_format = "%{y:,.2f}%"

        elif column_type == "count":
            hover_format = "%{y:,}"

        else:
            hover_format = "%{y:,.2f}"

        hover_template = (
            "%{x}<br>"
            + label
            + ": "
            + hover_format
            + "<extra></extra>"
        )

        use_line = (
            is_time_series
            and "year" in normalize_column(x_column)
            and column_type != "percentage"
        )

        if use_line:
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=df[column].tolist(),
                    mode="lines+markers",
                    name=label,
                    connectgaps=False,
                    hovertemplate=hover_template,
                )
            )

        else:
            bar_options = {}

            if column_type == "percentage":
                bar_options["marker_color"] = [
                    "#EF4444"
                    if pd.notna(value) and value < 0
                    else "#3B82F6"
                    for value in df[column]
                ]

            figure.add_trace(
                go.Bar(
                    x=x_values,
                    y=df[column].tolist(),
                    name=label,
                    hovertemplate=hover_template,
                    **bar_options,
                )
            )

    metric_title = " and ".join(
        clean_column_header(column)
        for column in metrics
    )

    dimension_title = clean_column_header(x_column)

    figure.update_layout(
        title=f"{metric_title} by {dimension_title}",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            color="#E2E8F0",
            size=13,
        ),
        colorway=[
            "#3B82F6",
            "#10B981",
            "#F59E0B",
            "#A78BFA",
            "#06B6D4",
        ],
        height=380,
        autosize=True,
        barmode="group",
        margin=dict(
            l=65,
            r=25,
            t=65,
            b=65,
        ),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
        ),
        xaxis_title=dimension_title,
        yaxis_title=(
            "Amount (USD)"
            if first_type == "money"
            else "Percentage"
            if first_type == "percentage"
            else "Value"
        ),
        hovermode="closest",
    )

    figure.update_xaxes(
        type="category",
        showgrid=False,
        linecolor="#334155",
        tickfont=dict(color="#CBD5E1"),
        automargin=True,
    )

    figure.update_yaxes(
        gridcolor="#334155",
        zerolinecolor="#64748B",
        tickfont=dict(color="#CBD5E1"),
        automargin=True,
    )

    figure.update_traces(
        selector=dict(type="scatter"),
        line=dict(width=3),
        marker=dict(size=8),
    )

    if first_type == "money":
        figure.update_yaxes(
            tickprefix="$",
            tickformat=",.2f",
        )

    elif first_type == "percentage":
        figure.update_yaxes(
            ticksuffix="%",
            tickformat=".2f",
        )

    elif first_type == "count":
        figure.update_yaxes(
            tickformat=",d",
        )

    return figure