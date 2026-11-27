"""Conservative standalone intent routing; unsupported requests fall through."""
import re
import numpy as np
from agent import calculation_library as lib
from agent.debug_logger import debug_log, DebugTimer
from agent.result_formatter import escape_text
from agent.currency_units import get_explicit_currency


def try_planned_analysis(df, question):
    timer = DebugTimer("PLANNER")
    debug_log("PLANNER", "Checking trusted calculation registry")
    q = question.strip().rstrip("?.!").strip()
    # Explicit wording keeps legacy routes and compound questions unchanged.
    match = re.fullmatch(r"(?:please\s+)?(?:calculate|compute|find|what is) (?:the )?(?:overall )?(.+)", q, re.I)
    if not match:
        return None
    intent = match.group(1).lower()
    stat = re.fullmatch(r"(mean|median|minimum|maximum|sum|sample variance|sample standard deviation|population variance|population standard deviation) of column (.+)", match.group(1), re.I)
    percentile = re.fullmatch(r"(\d+(?:\.\d+)?) percentile of column (.+)", match.group(1), re.I)
    correlation = re.fullmatch(r"(pearson|spearman|kendall) correlation of columns (.+) and (.+)", match.group(1), re.I)
    try:
        formula = None
        unit = None
        if intent in lib.RATIOS:
            value, unit, formula = lib.ratio(df, intent)
        elif correlation:
            from agent.sandbox_worker import safe_correlation
            method, left, right = correlation.groups()
            lc, rc = lib.resolve_column(df, left), lib.resolve_column(df, right)
            lib.numeric(df, lc)
            lib.numeric(df, rc)
            value = safe_correlation(df[lc], df[rc], method.lower())
            if not np.isfinite(value):
                raise lib.CalculationError("Correlation is undefined for these observations.")
            answer = f"The {method.title()} correlation coefficient is **{value:.4f}**.\n\nCorrelation does not establish causation."
            debug_log("PLANNER", "Trusted correlation completed; no LLM request")
            return {"answer": answer, "table": None, "error": None}
        elif intent == "break even units":
            fc, price, cost = [lib.resolve_column(df, c, True) for c in ("fixed_costs", "unit_price", "unit_variable_cost")]
            lib.compatible_currency(fc, price, cost)
            if len(df) != 1:
                raise lib.CalculationError("Break-even requires one scenario with fixed costs and per-unit prices and costs.")
            value = np.ceil(lib.divide(lib.numeric(df, fc, True)[0], lib.numeric(df, price, True)[0] - lib.numeric(df, cost, True)[0]))
            formula = "Fixed costs / (unit price − unit variable cost), rounded up"
        elif stat:
            metric, name = stat.groups()
            c = lib.resolve_column(df, name)
            value = lib.describe(df, c, metric.lower())
            unit = None if "variance" in metric.lower() else get_explicit_currency(c)
        elif percentile:
            p, name = percentile.groups()
            p = float(p)
            if not 0 <= p <= 100:
                raise lib.CalculationError("Percentile must be between 0 and 100.")
            c = lib.resolve_column(df, name)
            value = float(np.percentile(lib.numeric(df, c), p))
            unit = get_explicit_currency(c)
        elif intent == "total inventory value":
            value, unit = lib.inventory_value(df)
            formula = "Sum of stock on hand × unit cost for each record"
        elif intent == "employee headcount":
            value = lib.unique_ids(df, "employee_id").nunique()
            formula = "Distinct employee identifiers in the uploaded records"
        elif intent == "repeat purchase rate":
            value, unit = lib.repeat_purchase_rate(df), "%"
            formula = "Customers with multiple distinct orders / distinct customers × 100"
        else:
            debug_log("PLANNER", "No exact trusted intent; retaining existing AI route")
            return None
        label = {"roas": "ROAS", "marketing roi": "marketing ROI (advertising-return proxy)", "investment roi": "investment ROI"}.get(intent, intent)
        answer = f"The {escape_text(label)} is **{lib.display(value, unit)}**."
        if formula:
            answer += f"\n\nCalculation: {escape_text(formula)}."
        debug_log("PLANNER", f"Trusted calculation completed: {intent}; no LLM request")
        return {"answer": answer, "table": None, "error": None}
    except lib.CalculationError as exc:
        debug_log("PLANNER", f"Clarification required: {exc}")
        return {"answer": escape_text(str(exc)), "table": None, "error": None}
    finally:
        timer.finish("Trusted route check finished")
