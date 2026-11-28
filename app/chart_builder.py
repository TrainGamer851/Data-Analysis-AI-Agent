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


# Resolve a supported currency from a chart metric type, including legacy USD columns.
def get_chart_currency(column_type):

    # Treat legacy monetary results without a currency suffix as USD, matching table display behavior.
    if column_type == "money":
        return "USD"

    # Extract the currency code from the metric type before configuring chart labels.
    if column_type.startswith("money:"):
        currency = column_type.split(":", 1)[1]

        # Return only currencies supported by the shared chart and table formatters.
        if currency in CURRENCY_PREFIXES:
            return currency

    return None


# Determine whether a column is a supported categorical grouping dimension rather than an identifier or time field.
def is_chart_dimension(column, series):

    normalized = normalize_column(column)

    # Reject generated index fields because they are not meaningful business groupings.
    if normalized in IGNORED_COLUMNS:
        return False

    # Keep time fields on the time-series path instead of treating them as categories.
    if normalized in TIME_COLUMNS:
        return False

    column_type = get_column_type(column)

    # Exclude identifiers so charts do not group metrics by arbitrary record keys.
    if column_type == "identifier":
        return False

    # Use only ordinary numeric columns as categorical dimensions; exclude money and other measures.
    if column_type != "number":
        return False

    # Accept known business categories even when their labels are unfamiliar to the heuristic.
    if normalized in DIMENSION_COLUMNS:
        return True

    # Reject identifier-like column names.
    tokens = normalized.replace("-", " ").split()

    # Reject identifier-like labels such as emails, SKUs, and addresses as chart dimensions.
    if any(
        token
        in {
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

    # Require complete category labels so missing groups cannot disappear from the visualization.
    if series.isna().any():
        return False

    values = series.tolist()

    # Infer categories only from short, non-empty text values rather than free-form or mixed data.
    if not all(
        isinstance(value, str) and value.strip() and len(value.strip()) <= 80
        for value in values
    ):
        return False

    debug_log(
        "CHART DIMENSION",
        f"Inferred categorical dimension: {column}",
    )

    return True


# Validate the result table, select one grouping dimension and compatible metrics, then build a Plotly chart.
def build_chart(table):

    chart_timer = DebugTimer("CHART")

    debug_log(
        "CHART",
        "Starting chart generation",
    )

    # Skip visualization when analysis returned no supporting table to plot.
    if table is None:
        debug_log(
            "CHART",
            "Chart skipped: no supporting table",
        )

        return None

    df = pd.DataFrame(table).copy()

    debug_log(
        "CHART",
        f"Input rows: {len(df)} | " f"Columns: {df.shape[1]}",
    )

    # Skip charts with too few or too many rows to show a useful comparison.
    if len(df) < 2 or len(df) > MAX_CHART_ROWS:
        debug_log(
            "CHART",
            "Chart skipped: unsupported row count",
        )

        return None

    # Require a grouping field and at least one measure before creating a chart.
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
        column for column in df.columns if normalize_column(column) in TIME_COLUMNS
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

    # Skip the chart when no supported time or categorical grouping field is available.
    if not time_columns and not dimension_columns:
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: no supported grouping dimension",
        )

        return None

    # Avoid ambiguous plots when the result contains multiple possible grouping dimensions.
    if len(time_columns) + len(dimension_columns) > 1:
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: multiple grouping dimensions",
        )

        return None

    # Prefer a time field as the horizontal axis when the result supports a time series.
    if time_columns:
        x_column = time_columns[0]
        is_time_series = True

    # Use the categorical business dimension as the horizontal axis when no time field is present.
    else:
        x_column = dimension_columns[0]
        is_time_series = False

    # Skip the chart rather than silently dropping rows with missing group labels.
    if df[x_column].isna().any():
        debug_log(
            "CHART DIMENSION",
            "Chart skipped: missing dimension values",
        )

        return None

    # Require one row per plotted group so duplicate labels do not imply an unperformed aggregation.
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

    # Inspect each result column to identify the single grouping dimension and valid numeric measures.
    for column in df.columns:
        # Do not treat the selected grouping column as a numeric measure.
        if column == x_column:
            continue

        # Exclude index-like fields from the candidate measures shown in the chart.
        if normalize_column(column) in IGNORED_COLUMNS:
            continue

        # Exclude identifier fields because they are labels, not values to compare.
        if get_column_type(column) == "identifier":
            continue

        # Exclude boolean columns from numeric measures even though pandas may treat them as numeric.
        if pd.api.types.is_bool_dtype(df[column]):
            continue

        # Consider only numeric columns as possible chart measures.
        if pd.api.types.is_numeric_dtype(df[column]):
            # Keep a numeric measure only when it contains at least one plotted value.
            if df[column].notna().any():
                numeric_columns.append(column)

    # Skip chart creation when no valid numeric measure remains after filtering.
    if not numeric_columns:
        return None

    # Identify growth percentage columns so the chart can prioritize the requested trend.
    growth_columns = [
        column
        for column in numeric_columns
        if "growth" in normalize_column(column)
        and get_column_type(column) == "percentage"
    ]

    # Prioritize calculated growth percentages so the chart emphasizes the requested trend.
    if growth_columns:
        metrics = growth_columns[:3]
        first_type = "percentage"

    # When no growth metric is present, select measures that share the first metric's display type.
    else:
        first_type = get_column_type(numeric_columns[0])

        metrics = [
            column
            for column in numeric_columns
            if get_column_type(column) == first_type
        ][:3]

    chart_currency = get_chart_currency(first_type)

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
        f"Metric type: {first_type} | " f"Currency: {chart_currency}",
    )

    # Sort time-series rows chronologically before constructing the x-axis values.
    if is_time_series:
        df = df.sort_values(x_column)

    x_values = df[x_column].astype(str).tolist()

    figure = go.Figure()

    # Create one chart trace for each selected metric and apply its matching unit format.
    for column in metrics:
        label = clean_column_header(column)
        column_type = get_column_type(column)

        currency = get_chart_currency(column_type)

        hover_options = {}

        # Format hover values with their explicit currency while leaving plotted numbers unchanged.
        if currency is not None:
            # Keep currency formatting in the hover text while preserving raw numeric values for plotting.
            hover_options["customdata"] = [
                format_currency(value, currency) if pd.notna(value) else "Unavailable"
                for value in df[column].tolist()
            ]

            hover_format = "%{customdata}"

            debug_log(
                "CHART",
                f"Currency hover formatting: " f"{column} | {currency}",
            )

        # Add percent signs to hover values for percentage metrics.
        elif column_type == "percentage":
            hover_format = "%{y:,.2f}%"

        # Use integer-style grouping in hover labels for count metrics.
        elif column_type == "count":
            hover_format = "%{y:,}"

        # Use two decimal places for ordinary numeric metrics in hover labels.
        else:
            hover_format = "%{y:,.2f}"

        hover_template = "%{x}<br>" + label + ": " + hover_format + "<extra></extra>"

        use_line = (
            is_time_series
            and "year" in normalize_column(x_column)
            and column_type != "percentage"
        )

        # Use a connected line for annual time series; use bars for categorical or nonannual results.
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

        # Use grouped bars when the result is not an annual time series.
        else:
            bar_options = {}

            # Color negative percentage bars red and nonnegative bars blue for quick comparison.
            if column_type == "percentage":
                bar_options["marker_color"] = [
                    "#EF4444" if pd.notna(value) and value < 0 else "#3B82F6"
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

    metric_title = " and ".join(clean_column_header(column) for column in metrics)

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
            else "Percentage" if first_type == "percentage" else "Value"
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

    # Format the vertical-axis ticks and unit label with the selected currency.
    if chart_currency is not None:
        figure.update_yaxes(
            tickprefix=CURRENCY_PREFIXES[chart_currency],
            tickformat=",.2f",
        )

        debug_log(
            "CHART",
            f"Currency axis configured: " f"{chart_currency}",
        )

    # Add percentage units to the vertical axis when the selected metrics are percentages.
    elif first_type == "percentage":
        figure.update_yaxes(
            ticksuffix="%",
            tickformat=".2f",
        )

    # Use integer tick labels when the selected metrics represent counts.
    elif first_type == "count":
        figure.update_yaxes(
            tickformat=",d",
        )

    debug_log(
        "CHART",
        f"Chart traces generated: " f"{len(figure.data)}",
    )

    chart_timer.finish("Chart generation completed")

    return figure
