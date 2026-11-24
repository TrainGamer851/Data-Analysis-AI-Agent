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

    def __call__(self, messages):
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

    assert out == {
        "answer": "The total sales are $375.50.",
        "table": None,
        "error": None,
    }

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