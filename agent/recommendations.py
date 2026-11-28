import re
from agent.calculation_library import ratio, display, CalculationError
from agent.result_formatter import escape_text
from agent.debug_logger import debug_log
from agent.research import try_research

GUIDANCE = {
    "roas": "Compare campaign-level returns and contribution margins before reallocating advertising spend. Revenue per advertising dollar does not establish profitability or incremental lift.",
    "marketing roi": "Check whether product costs, overhead and incremental revenue are included before using this advertising-return proxy to allocate budget.",
    "net profit margin": "Break down revenue and costs by product or period to identify opportunities. Compare like-for-like periods before changing pricing or costs.",
    "gross profit margin": "Review product-level gross margins and cost drivers before changing prices or supplier arrangements.",
}


# Attempt to recommendation.
def try_recommendation(df, question):
    match = re.fullmatch(
        r"recommend actions based on (roas|marketing roi|net profit margin|gross profit margin)(?: with (inflation|gdp growth|unemployment) for ([a-z]{2}))?[?.!]?",
        question.strip(),
        re.I,
    )
    # Return `None` when `not match`.
    if not match:
        return None
    metric = match[1].lower()
    try:
        value, unit, formula = ratio(df, metric)
        label = {
            "roas": "ROAS",
            "marketing roi": "Marketing ROI (advertising-return proxy)",
        }.get(metric, metric.capitalize())
        answer = f"**Dataset finding**\n\n{label}: **{display(value, unit)}**.\n\nCalculation: {escape_text(formula)}.\n\n**Recommended next step**\n\n{GUIDANCE[metric]}\n\n**Limitations**\n\nAssumes compatible reporting periods and additive inputs. No external benchmark for this business or causal effect has been established."
        # Update `external` when `match[2]`.
        if match[2]:
            external = try_research(f"Research {match[2]} for {match[3]}")
            answer += (
                "\n\n**External context (separate evidence)**\n\n" + external["answer"]
            )
            answer += "\n\nCheck that the dataset covers the same country and period before comparing. The national indicator does not establish an effect on this business."
            debug_log(
                "RECOMMENDATION", "External context attached without causal inference"
            )
        debug_log(
            "RECOMMENDATION", "Calculated evidence and conditional guidance prepared"
        )
    except CalculationError as exc:
        answer = "Cannot support a recommendation yet. " + escape_text(str(exc))
        debug_log("RECOMMENDATION", "Calculation clarification required")
    return {"answer": answer, "table": None, "error": None}
