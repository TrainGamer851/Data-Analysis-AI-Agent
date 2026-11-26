"""Tests for the agent runner (agent/agent_runner.py) using a fake LLM.

The fake replaces generate_response(), so these tests run the real mini-swe-agent
loop and the real code sandbox, but never call NRP and need no API key.
"""
import pandas as pd
import pytest

import agent.agent_runner as runner
from agent.agent_runner import DataAnalysisEnvironment, run_analysis


class FakeLLM:
    """Stands in for generate_response(): returns scripted replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []  # the messages sent on each call

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("FakeLLM was called more times than scripted")
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No NRP and no Docker: fake model config, local sandbox."""
    monkeypatch.setenv("SANDBOX_BACKEND", "local")
    monkeypatch.setattr(runner, "get_model_config", lambda: {"model": "fake-llm"})


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeLLM with the given replies and return it."""
    def install(*replies):
        llm = FakeLLM(*replies)
        monkeypatch.setattr(runner, "generate_response", llm)
        return llm
    return install


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "orders.csv"
    pd.DataFrame({
        "Order ID": ["A1", "A1", "A2", "A3"],
        "Region": ["West", "West", "East", "South"],
        "Sales": [100.0, 50.0, 200.0, 25.5],
        "Profit": [10.0, -5.0, 40.0, 2.5],
    }).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Input validation (the LLM must not be called)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["", "   ", None])
def test_blank_question_is_rejected(fake_llm, csv_file, question):
    llm = fake_llm("result = 1")
    out = run_analysis(csv_file, question, explain=False)
    assert out == {
        "answer": None,
        "table": None,
        "error": "Please enter a question.",
    }
    assert llm.calls == []


def test_empty_csv_is_rejected(fake_llm, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Order ID,Sales\n")
    llm = fake_llm("result = 1")
    out = run_analysis(str(path), "What are total sales?", explain=False)
    assert out == {"answer": None, "table": None, "error": "The CSV dataset is empty."}
    assert llm.calls == []


def test_missing_csv_returns_error(fake_llm, tmp_path):
    llm = fake_llm("result = 1")
    out = run_analysis(str(tmp_path / "nope.csv"), "What are total sales?", explain=False)
    assert out["answer"] is None
    assert out["table"] is None
    assert out["error"]
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------

def test_scalar_answer(fake_llm, csv_file):
    llm = fake_llm("result = round(df['Sales'].sum(), 2)")
    out = run_analysis(csv_file, "What are total sales?", explain=False)
    assert out == {"answer": "375.5", "table": None, "error": None}
    assert len(llm.calls) == 1


def test_grouped_answer_comes_back_as_rows(fake_llm, csv_file):
    fake_llm("result = df.groupby('Region')['Sales'].sum()")
    out = run_analysis(csv_file,"Total sales by region?", explain=False)
    assert out["error"] is None
    assert out["answer"] == "The analysis returned 3 row(s)."
    assert {"Region": "West", "Sales": 150.0} in out["table"]


def test_code_in_markdown_fences_still_runs(fake_llm, csv_file):
    llm = fake_llm("```python\nresult = df['Order ID'].nunique()\n```")
    out = run_analysis(csv_file, "How many unique orders?", explain=False)
    assert out == {"answer": "3", "table": None, "error": None}


def test_prompt_contains_question_and_schema(fake_llm, csv_file):
    llm = fake_llm("result = 1")
    run_analysis(csv_file, "Which region is most profitable?", explain=False)
    first_call = llm.calls[0]
    roles = [m["role"] for m in first_call]
    assert roles[0] == "system"
    assert "business data analysis assistant" in first_call[0]["content"]
    user_text = " ".join(m["content"] for m in first_call if m["role"] == "user")
    assert "Which region is most profitable?" in user_text
    for column in ("Order ID", "Region", "Sales", "Profit"):
        assert column in user_text


# ---------------------------------------------------------------------------
# Retries and failures
# ---------------------------------------------------------------------------

def test_failed_code_is_retried_with_the_error(fake_llm, csv_file):
    llm = fake_llm(
        "result = df['Revenue'].sum()",        # wrong column -> KeyError
        "result = round(df['Sales'].sum(), 2)",  # corrected
    )
    out = run_analysis(csv_file, "What are total sales?", explain=False)
    assert out == {"answer": "375.5", "table": None, "error": None}
    assert len(llm.calls) == 2
    feedback = llm.calls[1][-1]
    assert feedback["role"] == "user"
    assert "Your Python code failed" in feedback["content"]
    assert "Revenue" in feedback["content"]


def test_agent_gives_up_after_step_limit(fake_llm, csv_file):
    llm = fake_llm("result = 1 / 0")
    out = run_analysis(csv_file, "What are total sales?", explain=False)
    assert out["answer"] is None
    assert out["table"] is None
    assert "after 3 attempts" in out["error"]
    assert len(llm.calls) == 3


def test_code_without_result_counts_as_failure(fake_llm, csv_file):
    llm = fake_llm("total = df['Sales'].sum()", "result = df['Sales'].sum()")
    out = run_analysis(csv_file, "What are total sales?", explain=False)
    assert out == {"answer": "375.5", "table": None, "error": None}
    assert "result" in llm.calls[1][-1]["content"]


def test_blank_llm_reply_returns_error(fake_llm, csv_file):
    fake_llm("   ")
    out = run_analysis(csv_file, "What are total sales?", explain=False)
    assert out["answer"] is None
    assert out["table"] is None
    assert "RepeatedFormatError" in out["error"]


def test_llm_exception_returns_error(fake_llm, csv_file):
    fake_llm(ConnectionError("NRP is down"))
    out = run_analysis(csv_file, "What are total sales?", explain=False)
    assert out["answer"] is None
    assert out["table"] is None
    assert "ConnectionError" in out["error"]


# ---------------------------------------------------------------------------
# Environment adapter
# ---------------------------------------------------------------------------

def test_environment_success_and_failure():
    env = DataAnalysisEnvironment(pd.DataFrame({"Sales": [1.0, 2.0]}))

    ok = env.execute({"code": "result = df['Sales'].sum()"})

    assert ok == {
        "output": "3.0",
        "returncode": 0,
        "exception_info": "",
        "sandbox_broken": False,
    }

    bad = env.execute({"code": "result = df['Nope'].sum()"})

    assert bad["returncode"] == 1
    assert bad["output"] == ""
    assert "Nope" in bad["exception_info"]
    assert bad["sandbox_broken"] is False

# ---------------------------------------------------------------------------
# Explanation stage
# ---------------------------------------------------------------------------

def test_explanation_stage(fake_llm, csv_file):
    llm = fake_llm(
        "result = df['Sales'].sum()",
        "The total sales are $375.50.",
    )

    out = run_analysis(
        csv_file,
        "What are total sales?",
        explain=True,
    )

    assert out["error"] is None
    assert out["table"] is None
    assert "The total sales are $375.50." in out["answer"]
    assert out["answer"] == "The total sales are $375.50."
    assert "**Original calculation**" not in out["answer"]
    assert "**AI interpretation**" not in out["answer"]

    assert len(llm.calls) == 2


def test_explanation_failure_preserves_result(fake_llm, csv_file):
    fake_llm(
        "result = df['Sales'].sum()",
        ConnectionError("Explanation failed"),
    )

    out = run_analysis(
        csv_file,
        "What are total sales?",
        explain=True,
    )

    assert out == {
        "answer": "375.5",
        "table": None,
        "error": None,
    }

def test_followup_context_reaches_model(fake_llm, csv_file):
    llm = fake_llm(
        "result = df.loc[df['Region'] == 'East', 'Profit'].sum()"
    )

    context = [
        {
            "role": "user",
            "content": "Which region has the highest profit?",
        },
        {
            "role": "assistant",
            "content": "East has the highest profit.",
        },
    ]

    result = run_analysis(
        csv_file,
        "How much profit did it generate?",
        explain=False,
        context=context,
    )

    assert result["error"] is None
    assert result["answer"] == "40.0"

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "BEGIN CONVERSATION" in prompt
    assert "East has the highest profit." in prompt
    assert "How much profit did it generate?" in prompt


def test_context_reaches_explanation_model(fake_llm, csv_file):
    llm = fake_llm(
        "result = df.loc[df['Region'] == 'East', 'Profit'].sum()",
        "East generated $40.00 in profit.",
    )

    context = [
        {
            "role": "user",
            "content": "Which region performed best?",
        },
        {
            "role": "assistant",
            "content": "East had the highest profit.",
        },
    ]

    result = run_analysis(
        csv_file,
        "How much profit did it generate?",
        explain=True,
        context=context,
    )

    assert result["error"] is None
    assert (
        "East generated $40.00 in profit."
        in result["answer"]
    )

    assert result["answer"] == "East generated $40.00 in profit."
    assert "**Original calculation**" not in result["answer"]
    assert "**AI interpretation**" not in result["answer"]

    assert len(llm.calls) == 2

    explanation_prompt = llm.calls[1][-1]["content"]

    assert "BEGIN CONVERSATION" in explanation_prompt
    assert "East had the highest profit." in explanation_prompt
    assert "40.0" in explanation_prompt


def test_explanation_skipped_when_budget_is_low(
    monkeypatch,
):
    import time

    monkeypatch.setenv(
        "ANALYSIS_DEADLINE_MONOTONIC",
        str(time.monotonic() + 7),
    )

    def unexpected_model_call(*args, **kwargs):
        pytest.fail("Explanation model should not be called")

    monkeypatch.setattr(
        runner,
        "generate_response",
        unexpected_model_call,
    )

    last = {
        "success": True,
        "result": 375.5,
        "result_type": "scalar",
        "error": None,
    }

    answer = runner.explain_result(
        "What are total sales?",
        last,
        "375.5",
    )

    assert answer == "375.5"


def test_explanation_timeout_preserves_result(
    monkeypatch,
):
    monkeypatch.delenv(
        "ANALYSIS_DEADLINE_MONOTONIC",
        raising=False,
    )

    def slow_model(*args, **kwargs):
        raise TimeoutError("Model request timed out")

    monkeypatch.setattr(
        runner,
        "generate_response",
        slow_model,
    )

    last = {
        "success": True,
        "result": 375.5,
        "result_type": "scalar",
        "error": None,
    }

    answer = runner.explain_result(
        "What are total sales?",
        last,
        "375.5",
    )

    assert answer == "375.5"

# ---------------------------------------------------------------------------
# Result integrity
# ---------------------------------------------------------------------------

def test_list_result_skips_explanation(fake_llm, csv_file):
    llm = fake_llm(
        "result = df['Region'].unique().tolist()",
        "The region names are unavailable.",
    )

    out = run_analysis(
        csv_file,
        "List all distinct region names.",
        explain=True,
    )

    assert out["error"] is None
    assert out["table"] is None
    assert "West" in out["answer"]
    assert "East" in out["answer"]
    assert "South" in out["answer"]
    assert "unavailable" not in out["answer"]
    assert len(llm.calls) == 1


def test_empty_list_result(fake_llm, csv_file):
    llm = fake_llm(
        "result = df.loc[df['Sales'] > 1000000, "
        "'Region'].tolist()"
    )

    out = run_analysis(
        csv_file,
        "Find regions with transactions above one million.",
        explain=True,
    )

    assert out["error"] is None
    assert out["answer"] == (
        "No matching results were returned."
    )
    assert len(llm.calls) == 1


def test_dictionary_result_preserves_values(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = {'Total Sales': df['Sales'].sum(), "
        "'Total Profit': df['Profit'].sum()}",
        "The calculated values are unavailable.",
    )

    out = run_analysis(
        csv_file,
        "Calculate combined sales and profit metrics.",
        explain=True,
    )

    assert out["error"] is None
    assert "**Original calculation**" not in out["answer"]
    assert "**AI interpretation**" not in out["answer"]
    assert "unavailable" not in out["answer"].lower()
    assert "Total Sales" in out["answer"]
    assert "$375.50" in out["answer"]
    assert "Total Profit" in out["answer"]
    assert "$47.50" in out["answer"]
    assert len(llm.calls) == 2


def test_incorrect_scalar_explanation_preserves_value(
    fake_llm,
    csv_file,
):
    fake_llm(
        "result = df['Sales'].sum()",
        "Total sales are $999.00.",
    )

    out = run_analysis(
        csv_file,
        "What are total sales?",
        explain=True,
    )

    assert out["error"] is None
    assert out["answer"] == "375.5"
    assert "999" not in out["answer"]
    assert "**Original calculation**" not in out["answer"]
    assert "**AI interpretation**" not in out["answer"]


def test_null_result_skips_explanation(
    fake_llm,
    csv_file,
):
    llm = fake_llm("result = None")

    out = run_analysis(
        csv_file,
        "Calculate an unavailable metric.",
        explain=True,
    )

    assert out["error"] is None
    assert out["answer"] == (
        "The calculation did not return a value."
    )
    assert len(llm.calls) == 1


def test_table_survives_incorrect_explanation(
    fake_llm,
    csv_file,
):
    fake_llm(
        "result = df.groupby('Region')['Sales'].sum()",
        "The regional sales figures are unavailable.",
    )

    out = run_analysis(
        csv_file,
        "Calculate grouped regional revenue figures.",
        explain=True,
    )

    assert out["error"] is None
    assert out["table"] is not None
    assert {
        "Region": "West",
        "Sales": 150.0,
    } in out["table"]


def test_empty_dictionary_format():
    last = {
        "result": {},
        "result_type": "object",
    }

    assert runner.describe_plainly(last) == (
        "The calculation returned no values."
    )


def test_nested_dictionary_format():
    last = {
        "result": {
            "West": {
                "Sales": 150.0,
                "Profit": 5.0,
            },
        },
        "result_type": "object",
    }

    answer = runner.describe_plainly(last)

    assert "West" in answer
    assert "Sales" in answer
    assert "150.0" in answer
    assert "Profit" in answer
    assert "5.0" in answer


def test_large_result_is_truncated():
    last = {
        "result": ["x" * 1000] * 10,
        "result_type": "object",
    }

    answer = runner.describe_plainly(last)

    assert "[answer truncated]" in answer



def test_list_result_has_no_unnecessary_quotes(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = df['Region'].unique().tolist()"
    )

    out = run_analysis(
        csv_file,
        "List all distinct region names.",
        explain=True,
    )

    assert out["error"] is None
    assert "- West" in out["answer"]
    assert '"West"' not in out["answer"]
    assert len(llm.calls) == 1


def test_nested_records_become_supporting_table(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = {"
        "'Total Sales': df['Sales'].sum(), "
        "'regional_analysis': "
        "df.groupby('Region', as_index=False)"
        "['Sales'].sum().to_dict('records')"
        "}"
    )

    out = run_analysis(
        csv_file,
        "Calculate detailed regional sales figures.",
        explain=True,
    )

    assert out["error"] is None
    assert "$375.50" in out["answer"]
    assert out["table"] is not None
    assert len(out["table"]) == 3

    assert {
        "Region": "West",
        "Sales": 150.0,
    } in out["table"]

    assert len(llm.calls) == 1


def test_list_of_records_becomes_table(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = df.groupby('Region', as_index=False)"
        "['Profit'].sum().to_dict('records')"
    )

    out = run_analysis(
        csv_file,
        "Calculate detailed regional profit records.",
        explain=True,
    )

    assert out["error"] is None
    assert len(out["table"]) == 3
    assert len(llm.calls) == 1


def test_named_dictionary_formats_money():
    last = {
        "result": {
            "Total_Sales": 1234.567,
            "Total_Profit": -100.25,
        },
        "result_type": "object",
    }

    answer = runner.describe_plainly(last)

    assert "$1,234.57" in answer
    assert "-$100.25" in answer


def test_named_dictionary_formats_percentages():
    last = {
        "result": {
            "Profit_Margin_Pct": 0.5,
        },
        "result_type": "object",
    }

    answer = runner.describe_plainly(last)

    assert "0.50%" in answer
    assert "50.00%" not in answer


def test_structured_result_preserves_correlation():
    from agent.result_formatter import (
        describe_record_table,
    )

    records = [
        {
            "Category": "A",
            "Discount_Profit_Correlation": -0.4782,
        }
    ]

    value = {
        "overall_discount_profit_correlation": -0.2189,
        "category_analysis": records,
    }

    answer = describe_record_table(
        value,
        "category_analysis",
        records,
    )

    assert "-0.2189" in answer
    assert "Correlation does not establish causation." in answer


def test_multiple_tables_are_not_silently_discarded():
    from agent.result_formatter import extract_record_table

    value = {
        "sales": [{"Region": "West", "Sales": 150}],
        "profit": [{"Region": "West", "Profit": 5}],
    }

    name, table = extract_record_table(value)

    assert name is None
    assert table is None


def test_inconsistent_record_columns_are_rejected():
    from agent.result_formatter import extract_record_table

    value = [
        {"Region": "West", "Sales": 150},
        {"Region": "East", "Profit": 40},
    ]

    name, table = extract_record_table(value)

    assert name is None
    assert table is None


def test_structured_table_row_limit():
    from agent.result_formatter import (
        extract_record_table,
        describe_record_table,
    )

    value = [
        {"Region": str(index), "Sales": index}
        for index in range(250)
    ]

    name, table = extract_record_table(value)

    assert len(table) == 200

    answer = describe_record_table(
        value,
        name,
        table,
    )

    assert "Showing 200 of 250" in answer



def test_correlation_table_direct_answer():
    last = {
        "result": [
            {
                "Category": "Furniture",
                "Discount_Profit_Correlation": -0.4782,
            },
        ],
        "result_type": "table",
    }

    answer = runner.describe_plainly(last)

    assert "The analysis returned 1 row(s)." in answer
    assert "Correlation does not establish causation." in answer


def test_correlation_explanation_adds_disclaimer(
    fake_llm,
):
    fake_llm(
        "Discount and profit have a negative association."
    )

    last = {
        "result": [
            {
                "Category": "Furniture",
                "Discount_Profit_Correlation": -0.4782,
            },
        ],
        "result_type": "table",
    }

    answer = runner.explain_result(
        "Analyze discount and profit.",
        last,
        runner.describe_plainly(last),
    )

    assert "negative association" in answer
    assert "Correlation does not establish causation." in answer


def test_correlation_disclaimer_not_duplicated(
    fake_llm,
):
    fake_llm(
        "Discount and profit are negatively associated. "
        "Correlation does not establish causation."
    )

    last = {
        "result": [
            {
                "Discount_Profit_Correlation": -0.4782,
            },
        ],
        "result_type": "table",
    }

    answer = runner.explain_result(
        "Analyze discount and profit.",
        last,
        runner.describe_plainly(last),
    )

    assert answer.count(
        "Correlation does not establish causation."
    ) == 1


def test_correlation_explanation_failure_preserves_disclaimer(
    fake_llm,
):
    fake_llm(
        TimeoutError("Explanation timed out")
    )

    last = {
        "result": [
            {
                "Discount_Profit_Correlation": -0.4782,
            },
        ],
        "result_type": "table",
    }

    plain_answer = runner.describe_plainly(last)

    answer = runner.explain_result(
        "Analyze discount and profit.",
        last,
        plain_answer,
    )

    assert answer == plain_answer
    assert "Correlation does not establish causation." in answer


def test_non_correlation_table_has_no_disclaimer():
    last = {
        "result": [
            {
                "Region": "West",
                "Total_Sales": 150.0,
            },
        ],
        "result_type": "table",
    }

    answer = runner.describe_plainly(last)

    assert answer == "The analysis returned 1 row(s)."
    assert "causation" not in answer.lower()



def test_single_group_financial_summary():
    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Product": "Product A",
            "Total_Sales": 1500.0,
            "Total_Profit": -150.0,
            "Profit_Margin_Pct": -10.0,
            "Average_Group_Sales": 1000.0,
        },
    ]

    answer = summarize_financial_table(records)

    assert answer is not None
    assert "Product A" in answer
    assert "$1,500.00" in answer
    assert "-$150.00" in answer
    assert "-10.00%" in answer
    assert "$1,000.00" in answer
    assert "negative total profit" in answer


def test_discount_relationship_summary():
    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Category": "Category A",
            "Average_Discount_Pct": 20.0,
            "Profit_Margin_Pct": 5.0,
            "Discount_Profit_Correlation": -0.65,
        },
        {
            "Category": "Category B",
            "Average_Discount_Pct": 10.0,
            "Profit_Margin_Pct": 15.0,
            "Discount_Profit_Correlation": -0.25,
        },
    ]

    answer = summarize_financial_table(records)

    assert answer is not None
    assert "Category A" in answer
    assert "20.00%" in answer
    assert "5.00%" in answer
    assert "-0.6500" in answer
    assert "negative association" in answer
    assert (
        "Correlation does not establish causation."
        in answer
    )


def test_average_discount_summary():
    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Region": "North",
            "Average_Discount_Pct": 12.5,
        },
        {
            "Region": "South",
            "Average_Discount_Pct": 18.25,
        },
    ]

    answer = summarize_financial_table(records)

    assert answer is not None
    assert "North" in answer
    assert "South" in answer
    assert "12.50%" in answer
    assert "18.25%" in answer


def test_average_discount_summary_preserves_ties():
    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Region": "North",
            "Average_Discount_Pct": 20.0,
        },
        {
            "Region": "South",
            "Average_Discount_Pct": 20.0,
        },
        {
            "Region": "East",
            "Average_Discount_Pct": 10.0,
        },
    ]

    answer = summarize_financial_table(records)

    assert answer is not None
    assert "North" in answer
    assert "South" in answer
    assert "East" in answer
    assert "20.00%" in answer


def test_equal_average_discount_summary():
    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Region": "North",
            "Average_Discount_Pct": 15.0,
        },
        {
            "Region": "South",
            "Average_Discount_Pct": 15.0,
        },
    ]

    answer = summarize_financial_table(records)

    assert answer is not None
    assert "same average discount" in answer
    assert "15.00%" in answer


@pytest.mark.parametrize(
    "records",
    [
        [],
        None,
        ["North", "South"],
        [
            {
                "Region": "North",
                "Total_Sales": 100,
            },
        ],
        [
            {
                "Region": "North",
                "Total_Sales": 100,
                "Total_Profit": None,
            },
        ],
        [
            {
                "Region": "North",
                "Total_Sales": float("inf"),
                "Total_Profit": 20,
            },
        ],
    ],
)
def test_unsupported_financial_summary(records):
    from agent.result_formatter import (
        summarize_financial_table,
    )

    assert summarize_financial_table(records) is None


def test_financial_summary_does_not_modify_records():
    from copy import deepcopy

    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Product": "Product A",
            "Total_Sales": 1500.0,
            "Total_Profit": -150.0,
        },
    ]

    original = deepcopy(records)

    summarize_financial_table(records)

    assert records == original


def test_financial_summary_escapes_group_labels():
    from agent.result_formatter import (
        summarize_financial_table,
    )

    records = [
        {
            "Product": "*Product A*",
            "Total_Sales": 1500.0,
            "Total_Profit": -150.0,
        },
    ]

    answer = summarize_financial_table(records)

    assert r"\*Product A\*" in answer


def test_single_group_financial_result_skips_llm(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = df.groupby('Region', as_index=False)"
        ".agg(Total_Sales=('Sales', 'sum'), "
        "Total_Profit=('Profit', 'sum'))"
        ".query('Total_Sales > 175')"
    )

    out = run_analysis(
        csv_file,
        "Show sales and profit for regions with total sales above 175.",
        explain=True,
    )

    assert out["error"] is None
    assert out["table"] is not None
    assert len(out["table"]) == 1
    assert "East" in out["answer"]
    assert "$200.00" in out["answer"]
    assert "$40.00" in out["answer"]
    assert len(llm.calls) == 1


def test_discount_summary_skips_second_llm_request(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = df.groupby('Region', as_index=False)"
        ".agg(Total_Sales=('Sales', 'sum'), "
        "Total_Profit=('Profit', 'sum'))\n"
        "result['Average_Discount_Pct'] = "
        "result['Total_Sales'] * 0.1\n"
        "result['Profit_Margin_Pct'] = "
        "result['Total_Profit'] / "
        "result['Total_Sales'] * 100\n"
        "result['Discount_Profit_Correlation'] = -0.5"
    )

    out = run_analysis(
        csv_file,
        "Analyze discount and profitability by region.",
        explain=True,
    )

    assert out["error"] is None
    assert len(out["table"]) == 3
    assert (
        "Correlation does not establish causation."
        in out["answer"]
    )
    assert len(llm.calls) == 1


def test_recommendation_retains_explanation_route(
    fake_llm,
    csv_file,
):
    llm = fake_llm(
        "result = df.groupby('Region', as_index=False)"
        ".agg(Total_Sales=('Sales', 'sum'), "
        "Total_Profit=('Profit', 'sum'))"
        ".query('Total_Sales > 175')",
        "Investigate opportunities to improve performance.",
    )

    out = run_analysis(
        csv_file,
        "Suggest actions based on detailed regional results.",
        explain=True,
    )

    assert out["error"] is None
    assert len(llm.calls) == 2
    assert "Investigate opportunities" in out["answer"]



def test_financial_sales_threshold_skips_llm(
    fake_llm,
    csv_file,
):
    llm = fake_llm("result = 999")

    result = run_analysis(
        csv_file,
        "Which regions have total sales above $100?",
        explain=True,
    )

    assert result["error"] is None
    assert llm.calls == []

    assert result["table"] == [
        {
            "Region": "East",
            "Total_Sales": 200.0,
        },
        {
            "Region": "West",
            "Total_Sales": 150.0,
        },
    ]


def test_financial_margin_threshold_skips_llm(
    fake_llm,
    csv_file,
):
    llm = fake_llm("result = 999")

    result = run_analysis(
        csv_file,
        "Which regions have a profit margin below 5%?",
        explain=True,
    )

    assert result["error"] is None
    assert llm.calls == []

    assert len(result["table"]) == 1
    assert result["table"][0]["Region"] == "West"

    assert (
        result["table"][0]["Profit_Margin_Pct"]
        == 3.33
    )


def test_financial_route_preserves_explain_false(
    fake_llm,
    csv_file,
):
    llm = fake_llm("result = 123")

    result = run_analysis(
        csv_file,
        "Which regions have total sales above $100?",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "123"
    assert len(llm.calls) == 1



def test_discount_relationship_uses_zero_llm_requests(
    fake_llm,
    tmp_path,
):
    llm = fake_llm(
        AssertionError("LLM should not be called")
    )

    path = tmp_path / "discount_orders.csv"

    pd.DataFrame({
        "Category": ["Alpha", "Alpha", "Beta", "Beta"],
        "Sales": [100, 200, 300, 400],
        "Profit": [30, -10, 20, 60],
        "Discount": [0.1, 0.3, 0.05, 0.15],
    }).to_csv(path, index=False)

    result = run_analysis(
        str(path),
        "Analyze the relationship between discount "
        "and profitability across product categories.",
        explain=True,
    )

    assert result["error"] is None
    assert len(result["table"]) == 2
    assert llm.calls == []

    assert "Correlation does not establish causation" in (
        result["answer"]
    )


def test_average_discount_uses_zero_llm_requests(
    fake_llm,
    tmp_path,
):
    llm = fake_llm(
        AssertionError("LLM should not be called")
    )

    path = tmp_path / "discount_orders.csv"

    pd.DataFrame({
        "Region": ["North", "North", "South"],
        "Discount": [0.1, 0.3, 0.4],
    }).to_csv(path, index=False)

    result = run_analysis(
        str(path),
        "Show average discount by region.",
        explain=True,
    )

    assert result["error"] is None
    assert len(result["table"]) == 2
    assert llm.calls == []




def test_single_row_summary_preserves_calculated_values():
    from agent.result_formatter import summarize_single_row

    records = [
        {
            "Campaign": "Email",
            "Advertising_Spend": 500,
            "Conversions": 80,
            "Cost_Per_Conversion": 6.25,
        },
    ]

    answer = summarize_single_row(records)

    assert "Email" in answer
    assert "$500.00" in answer
    assert "$6.25" in answer
    assert "80" in answer
    assert "row(s)" not in answer


def test_single_row_summary_preserves_correlation_warning():
    from agent.result_formatter import summarize_single_row

    records = [
        {
            "Category": "Furniture",
            "Discount_Profit_Correlation": -0.4782,
        },
    ]

    answer = summarize_single_row(records)

    assert "Furniture" in answer
    assert "-0.4782" in answer
    assert "Correlation does not establish causation." in answer


def test_single_row_summary_rejects_multiple_rows():
    from agent.result_formatter import summarize_single_row

    records = [
        {"Department": "Engineering", "Salary": 100000},
        {"Department": "Marketing", "Salary": 75000},
    ]

    assert summarize_single_row(records) is None


def test_single_row_explanation_failure_preserves_values(
    fake_llm,
    csv_file,
):
    fake_llm(
        "result = pd.DataFrame(["
        "{'Campaign': 'Email', "
        "'Cost_Per_Conversion': 6.25}])",
        TimeoutError("Explanation timed out"),
    )

    out = run_analysis(
        csv_file,
        "Which campaign has the lowest cost per conversion?",
        explain=True,
    )

    assert out["error"] is None
    assert "Email" in out["answer"]
    assert "$6.25" in out["answer"]
    assert "row(s)" not in out["answer"]
    assert out["table"][0]["Cost_Per_Conversion"] == 6.25



@pytest.mark.parametrize(
    "value, explanation, expected",
    [
        (
            {"ROI_Pct": 500},
            "The return on investment is **500%**.",
            True,
        ),
        (
            {"Net_Income_Increase_Pct": 50},
            "Net income increased by **50%** "
            "from January to March.",
            True,
        ),
        (
            {"Total_Inventory_Value": 13430},
            "The total inventory value is **$13,430.00**.",
            True,
        ),
        (
            {"ROI_Pct": 500},
            "The return on investment is 500.",
            False,
        ),
        (
            {"ROI_Pct": 500},
            "The return on investment is 5000%.",
            False,
        ),
        (
            {"Net_Income": 3000},
            "Net income is $3,000 or $9,000.",
            False,
        ),
        (
            {"Net_Income": 3000},
            "Net income is 3,000%.",
            False,
        ),
        (
            {"Net_Income": -3000},
            "Net income is -$3,000.",
            True,
        ),
        (
            {"Net_Income": -3000},
            "Net income is $3,000.",
            False,
        ),
    ],
)


def test_single_metric_explanation_validation(
    value,
    explanation,
    expected,
):
    from agent.result_formatter import (
        validate_single_metric_explanation,
    )

    assert validate_single_metric_explanation(
        explanation,
        value,
    ) is expected


def test_single_metric_natural_fallback():
    from agent.result_formatter import (
        summarize_single_metric,
    )

    answer = summarize_single_metric(
        {"ROI_Pct": 500}
    )

    assert "500%" in answer
    assert answer.endswith(".")
    assert "Calculated values" not in answer


def test_named_percentage_explanation_accepted(
    fake_llm,
    csv_file,
):
    fake_llm(
        "result = {'Profit_Margin_Pct': "
        "round(df['Profit'].sum() / "
        "df['Sales'].sum() * 100, 2)}",
        "The profit margin is **12.65%**.",
    )

    out = run_analysis(
        csv_file,
        "Calculate the overall profit margin percentage.",
        explain=True,
    )

    assert out["error"] is None
    assert "12.65%" in out["answer"]
    assert "Original calculation" not in out["answer"]


def test_context_excludes_unlabeled_markdown_table():
    from agent.conversation_context import (
        build_conversation_context,
    )

    history = [
        {
            "role": "user",
            "content": "Show revenue by department.",
        },
        {
            "role": "assistant",
            "content": (
                "Revenue by department.\n\n"
                "| Department | Revenue |\n"
                "|:-----------|--------:|\n"
                "| Engineering | $500.00 |"
            ),
        },
    ]

    context = build_conversation_context(history)

    assert len(context) == 2

    assert context[1]["content"] == (
        "Revenue by department."
    )



def test_business_schema_reaches_analysis_model(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "marketing.csv"

    pd.DataFrame({
        "Campaign": ["Email", "Search"],
        "Advertising_Spend": [500, 1000],
        "Generated_Revenue": [3000, 4000],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "result = df['Generated_Revenue'].sum()"
    )

    result = run_analysis(
        str(path),
        "Calculate total campaign revenue.",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "7000"

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "Business schema hints" in prompt
    assert "advertising_spend: Advertising_Spend" in prompt
    assert "revenue: Generated_Revenue" in prompt
    assert "campaign: Campaign" in prompt

    assert len(llm.calls) == 1


def test_ambiguous_schema_reaches_analysis_model(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "ambiguous.csv"

    pd.DataFrame({
        "Sales": [1000],
        "Revenue": [1200],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "result = 'Cannot answer: clarify whether "
        "Sales or Revenue should be used.'"
    )

    result = run_analysis(
        str(path),
        "Calculate total revenue.",
        explain=False,
    )

    assert result["error"] is None

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "revenue: Sales, Revenue" in prompt
    assert "AMBIGUOUS" in prompt


def test_schema_failure_preserves_analysis(
    fake_llm,
    csv_file,
    monkeypatch,
):
    import agent.dataset_schema as schema_module

    def failed_profile(*args, **kwargs):
        raise RuntimeError(
            "Simulated profiling failure"
        )

    monkeypatch.setattr(
        schema_module,
        "profile_dataset",
        failed_profile,
    )

    llm = fake_llm(
        "result = df['Sales'].sum()"
    )

    result = run_analysis(
        csv_file,
        "What are total sales?",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "375.5"

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "Sales" in prompt
    assert "Business schema hints" not in prompt


def test_schema_profile_runs_once_per_analysis(
    fake_llm,
    csv_file,
    monkeypatch,
):
    import agent.dataset_schema as schema_module

    original = schema_module.profile_dataset
    calls = []

    def tracked_profile(dataframe):
        calls.append(1)

        return original(dataframe)

    monkeypatch.setattr(
        schema_module,
        "profile_dataset",
        tracked_profile,
    )

    llm = fake_llm(
        "result = df['Unknown_Column'].sum()",
        "result = df['Sales'].sum()",
    )

    result = run_analysis(
        csv_file,
        "What are total sales?",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "375.5"

    assert len(calls) == 1
    assert len(llm.calls) == 2


def test_invalid_revenue_is_blocked_before_model(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "quality.csv"

    pd.DataFrame({
        "Revenue": ["1000", "unknown", "2000"],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "result = 3000"
    )

    result = run_analysis(
        str(path),
        "Calculate total revenue.",
        explain=False,
    )

    assert result["error"] is None
    assert result["table"] is None

    assert "1 row(s)" in result["answer"]
    assert "2 valid numeric row(s)" in result["answer"]

    assert llm.calls == []

def test_formatted_numeric_text_analysis(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "formatted_revenue.csv"

    pd.DataFrame({
        "Revenue_USD": [
            "$1,000",
            "$2,000",
        ],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "values = df['Revenue_USD'].str.replace("
        "'$', '', regex=False).str.replace("
        "',', '', regex=False)\n"
        "result = float(pd.to_numeric(values).sum())"
    )

    result = run_analysis(
        str(path),
        "Calculate total revenue in USD.",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "3000.0"

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "formatted_numeric_text" in prompt


def test_unrelated_date_ambiguity_does_not_block_analysis(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "ambiguous_dates.csv"

    pd.DataFrame({
        "Order_Date": [
            "01/02/2026",
            "03/04/2026",
        ],
        "Revenue": [1000, 2000],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "result = float(df['Revenue'].sum())"
    )

    result = run_analysis(
        str(path),
        "Calculate total revenue.",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "3000.0"

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "ambiguous_date_convention" in prompt



def test_ambiguous_dates_block_model_generation(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "ambiguous_dates.csv"

    pd.DataFrame({
        "Order_Date": [
            "01/02/2026",
            "03/04/2026",
        ],
        "Revenue": [1000, 2000],
    }).to_csv(path, index=False)

    llm = fake_llm("result = 3000")

    result = run_analysis(
        str(path),
        "Calculate monthly revenue.",
        explain=False,
    )

    assert result["error"] is None
    assert result["table"] is None

    assert "MM/DD/YYYY" in result["answer"]
    assert "DD/MM/YYYY" in result["answer"]

    assert llm.calls == []


def test_mixed_discount_scale_blocks_model_generation(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "mixed_discounts.csv"

    pd.DataFrame({
        "Discount": [0.15, 20, 0.10, 30],
    }).to_csv(path, index=False)

    llm = fake_llm("result = 18.75")

    result = run_analysis(
        str(path),
        "Calculate average discount percentage.",
        explain=False,
    )

    assert result["error"] is None
    assert result["table"] is None

    assert "discount units" in result["answer"]

    assert llm.calls == []


def test_invalid_revenue_does_not_block_quantity(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "invalid_revenue.csv"

    pd.DataFrame({
        "Revenue": ["1000", "unknown", "2000"],
        "Quantity": [10, 20, 30],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "result = int(df['Quantity'].sum())"
    )

    result = run_analysis(
        str(path),
        "Calculate total quantity.",
        explain=False,
    )

    assert result["error"] is None
    assert result["answer"] == "60"

    assert len(llm.calls) == 1


def test_quality_hints_reach_model_for_explicit_exclusions(
    fake_llm,
    tmp_path,
):
    path = tmp_path / "partial_revenue.csv"

    pd.DataFrame({
        "Revenue": ["1000", "unknown", "2000"],
    }).to_csv(path, index=False)

    llm = fake_llm(
        "values = pd.to_numeric("
        "df['Revenue'], errors='coerce')\n"
        "valid = values.notna() & np.isfinite(values)\n"
        "result = {"
        "'Valid_Revenue': float(values[valid].sum()), "
        "'Excluded_Rows': int((~valid).sum())}"
    )

    result = run_analysis(
        str(path),
        "Calculate the sum of only the valid finite revenue "
        "values and report how many rows were excluded.",
        explain=False,
    )

    assert result["error"] is None
    assert len(llm.calls) == 1

    prompt = "\n".join(
        message["content"]
        for message in llm.calls[0]
        if message["role"] == "user"
    )

    assert "partially_numeric_text" in prompt
    assert "invalid numeric values=1" in prompt
    assert "3000" in result["answer"]
