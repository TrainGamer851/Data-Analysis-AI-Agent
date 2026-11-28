import pandas as pd
import pytest
from agent import research, agent_runner
from agent.recommendations import try_recommendation


# Verify that research without csv.
@pytest.mark.parametrize(
    "question", ["Research inflation for US.", "Please research inflation for US."]
)
def test_research_without_csv(monkeypatch, question):
    monkeypatch.setattr(research, "fetch_indicator", lambda *a: (2024, 2.5))
    monkeypatch.setattr(agent_runner, "load_csv", lambda *a: pytest.fail("CSV read"))
    result = agent_runner.run_analysis(None, question)
    assert result["error"] is None and "2.50% annually" in result["answer"]


# Verify that missing csv still rejected.
def test_missing_csv_still_rejected():
    assert agent_runner.run_analysis(None, "Calculate revenue")["error"]


# Verify that combined evidence.
def test_combined_evidence(monkeypatch):
    monkeypatch.setattr(research, "fetch_indicator", lambda *a: (2024, 2.5))
    result = try_recommendation(
        pd.DataFrame({"Revenue": [100], "Advertising_Spend": [20]}),
        "Recommend actions based on marketing ROI with inflation for US.",
    )
    assert (
        "400.00%" in result["answer"] and "advertising-return proxy" in result["answer"]
    )
    assert (
        "2.50% annually" in result["answer"]
        and "same country and period" in result["answer"]
    )


# Verify that combined invalid dataset no network.
def test_combined_invalid_dataset_no_network(monkeypatch):
    monkeypatch.setattr(
        research, "fetch_indicator", lambda *a: pytest.fail("Unexpected network")
    )
    assert (
        "Cannot support"
        in try_recommendation(
            pd.DataFrame({"Revenue": [100]}),
            "Recommend actions based on ROAS with inflation for US",
        )["answer"]
    )


# Verify that combined network failure.
def test_combined_network_failure(monkeypatch):
    # Raise the configured test failure when the operation is called.
    def fail(*a):
        raise TimeoutError()

    monkeypatch.setattr(research, "fetch_indicator", fail)
    answer = try_recommendation(
        pd.DataFrame({"Revenue": [100], "Advertising_Spend": [20]}),
        "Recommend actions based on ROAS with inflation for US",
    )["answer"]
    assert "**5**" in answer and "No external value" in answer


# Verify that ui without csv.
def test_ui_without_csv(monkeypatch):
    from app import app

    monkeypatch.setattr(research, "fetch_indicator", lambda *a: (2024, 2.5))
    answer, chart = app.analyze_question_with_chart("Research inflation for US", None)
    assert "2.50% annually" in answer and chart is None


# Verify that controlled research null path.
def test_controlled_research_null_path(monkeypatch):
    from agent import analysis_controller as controller
    import json

    original = controller.subprocess.Popen
    seen = []

    # Start a mock worker process and capture its launch arguments.
    def launch(args, **kwargs):
        # Return the selected result when `'agent.analysis_controller' not in args`.
        if "agent.analysis_controller" not in args:
            return original(args, **kwargs)
        from pathlib import Path

        request = json.loads(Path(args[-2]).read_text())
        seen.append(request["csv_path"])
        return original(
            [
                controller.sys.executable,
                "-c",
                'import sys; from pathlib import Path; Path(sys.argv[1]).write_text(\'{"answer":"ok","table":null,"error":null}\')',
                args[-1],
            ],
            **kwargs
        )

    monkeypatch.setattr(controller.subprocess, "Popen", launch)
    monkeypatch.setattr(controller, "cleanup_docker", lambda *a: None)
    assert (
        controller.run_controlled_analysis(None, "Research inflation for US")["answer"]
        == "ok"
    )
    assert seen == [None]
