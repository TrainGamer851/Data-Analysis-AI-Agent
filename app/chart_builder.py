import pandas as pd
import plotly.graph_objects as go

from agent.currency_units import (
    CURRENCY_PREFIXES,
    format_currency,
)

from agent.debug_logger import (
    debug_log,
    DebugTimer,
)

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


def get_chart_currency(column_type):
    """Resolve the currency associated with a chart metric."""

    # Preserve existing behavior for legacy monetary columns.
    if column_type == "money":
        return "USD"

    if column_type.startswith("money:"):
        currency = column_type.split(":", 1)[1]

        if currency in CURRENCY_PREFIXES:
            return currency

    return None


def is_chart_dimension(column, series):
    """Identify a supported categorical chart dimension."""

    normalized = normalize_column(column)

    if normalized in IGNORED_COLUMNS:
        return False

    if normalized in TIME_COLUMNS:
        return False

    column_type = get_column_type(column)

    if column_type == "identifier":
        return False

    if column_type != "number":
        return False

    if normalized in DIMENSION_COLUMNS:
        return True

    # Reject identifier-like column names.
    tokens = normalized.replace("-", " ").split()

    if any(
        token in {
            "id",
            "identifier",
            "uuid",
            "sku",
            "email",
            "phone",
            "address",
            "description",
            "notes",
            "comment",
            "comments",
        }
        for token in tokens
    ):
        return False

    # Only infer dimensions from non-empty text values.
    if series.isna().any():
        return False

    values = series.tolist()

    if not all(
        isinstance(value, str)
        and value.strip()
        and len(value.strip()) <= 80
        for value in values
    ):
        return False

    debug_log(
        "CHART DIMENSION",
        f"Inferred categorical dimension: {column}",
    )

    return True


def build_chart(table):
    """Return a Plotly figure or None for unsupported data."""

    chart_timer = DebugTimer("CHART")

    debug_log(
        "CHART",
        "Starting chart generation",
    )

    if table is None:
        debug_log(
            "CHART",
            "Chart skipped: no supporting table",
        )

        return None

    df = pd.DataFrame(table).copy()

    debug_log(
        "CHART",
        f"Input rows: {len(df)} | "
        f"Columns: {df.shape[1]}",
    )

    if len(df) < 2 or len(df) > MAX_CHART_ROWS:
        debug_log(
            "CHART",
            "Chart skipped: unsupported row count",
        )

        return None

    if df.shape[1] < 2:
        debug_log(
            "CHART",
            "Chart skipped: insufficient columns",
        )

        return None

    debug_log(
        "CHART DIMENSION",
        "Inspecting table for chart dimensions",
    )

    time_columns = [
        column
        for column in df.columns
        if normalize_column(column) in TIME_COLUMNS
    ]

    dimension_columns = [
        column
        for column in df.columns
        if is_chart_dimension(
            column,
            df[column],
        )
    ]

    debug_log(
        "CHART DIMENSION",
        f"Time dimensions: {time_columns}",
    )

    debug_log(
        "CHART DIMENSION",
        f"Categorical dimensions: {dimension_columns}",
    )

    if not time_columns and not dimension_columns:
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: no supported grouping dimension",
        )

        return None

    # Avoid misleading charts for multidimensional tables.
    if len(time_columns) + len(dimension_columns) > 1:
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: multiple grouping dimensions",
        )

        return None

    if time_columns:
        x_column = time_columns[0]
        is_time_series = True

    else:
        x_column = dimension_columns[0]
        is_time_series = False

    if df[x_column].isna().any():
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: missing dimension values",
        )

        return None

    if df[x_column].duplicated().any():
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: duplicate dimension values",
        )

        return None

    debug_log(
        "CHART DIMENSION",
        f"Validated chart dimension: {x_column}",
    )

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

    chart_currency = get_chart_currency(
        first_type
    )

    debug_log(
        "CHART",
        f"Selected dimension: {x_column}",
    )

    debug_log(
        "CHART",
        f"Selected metrics: {metrics}",
    )

    debug_log(
        "CHART",
        f"Metric type: {first_type} | "
        f"Currency: {chart_currency}",
    )

    if is_time_series:
        df = df.sort_values(x_column)


    x_values = df[x_column].astype(str).tolist()

    figure = go.Figure()


    for column in metrics:
        label = clean_column_header(column)
        column_type = get_column_type(column)

        currency = get_chart_currency(
            column_type
        )

        hover_options = {}

        if currency is not None:
            # Format hover labels using the shared currency
            # utility without modifying plotted values.
            hover_options["customdata"] = [
                format_currency(value, currency)
                if pd.notna(value)
                else "Unavailable"
                for value in df[column].tolist()
            ]

            hover_format = "%{customdata}"

            debug_log(
                "CHART",
                f"Currency hover formatting: "
                f"{column} | {currency}",
            )

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
                    **hover_options,
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
                    **hover_options,
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
            f"Amount ({chart_currency})"
            if chart_currency is not None
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


    if chart_currency is not None:
        figure.update_yaxes(
            tickprefix=CURRENCY_PREFIXES[
                chart_currency
            ],
            tickformat=",.2f",
        )

        debug_log(
            "CHART",
            f"Currency axis configured: "
            f"{chart_currency}",
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

    debug_log(
        "CHART",
        f"Chart traces generated: "
        f"{len(figure.data)}",
    )

    chart_timer.finish(
        "Chart generation completed"
    )

    return figure
