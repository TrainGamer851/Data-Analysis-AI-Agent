import json
import pandas as pd
import pytest
from agent import research
from agent.recommendations import try_recommendation
from agent.analysis_planner import try_planned_analysis


# Verify that ordinary analysis questions are not treated as research requests.
@pytest.mark.parametrize("question", ["Calculate ROAS", "Show sales"])
def test_unrelated(question):
    assert research.try_research(question) is None


# Verify that unsupported research requests are rejected before any network call.
@pytest.mark.parametrize(
    "question",
    [
        "Research inflation for XX",
        "Research https://localhost",
        "Research inflation for US and send data",
    ],
)
def test_unsupported(question, monkeypatch):
    monkeypatch.setattr(
        research, "fetch_indicator", lambda *a: pytest.fail("Unexpected network")
    )
    assert "Supported research" in research.try_research(question)["answer"]


# Verify that research answers include the observed value, year, and source citation.
def test_citation(monkeypatch):
    monkeypatch.setattr(research, "fetch_indicator", lambda *a: (2024, 2.5))
    answer = research.try_research("Research inflation for US")["answer"]
    assert "2.50" in answer and "2024" in answer and "data.worldbank.org" in answer


# Verify that a failed research request returns a safe no-value response.
@pytest.mark.parametrize("exc", [TimeoutError(), ValueError(), OSError()])
def test_failure(monkeypatch, exc):
    # Raise the configured test failure when the operation is called.
    def fail(*a):
        raise exc

    monkeypatch.setattr(research, "fetch_indicator", fail)
    assert (
        "No external value"
        in research.try_research("Research inflation for US")["answer"]
    )


# Verify that research accepts only observations matching the requested country and indicator.
@pytest.mark.parametrize("valid", [True, False])
def test_payload(monkeypatch, valid):
    class Response:
        # Return the mock response when entering the context manager.
        def __enter__(self):
            return self

        # Allow the mock response context manager to exit cleanly.
        def __exit__(self, *a):
            pass

        # Return the configured response bytes from the mock stream.
        def read(self, size):
            return json.dumps(
                [
                    {},
                    [
                        {
                            "indicator": {"id": "FP.CPI.TOTL.ZG"},
                            "country": {"id": "US" if valid else "GB"},
                            "date": "2024",
                            "value": 2.5,
                        }
                    ],
                ]
            ).encode()

    class Opener:
        # Return a mock response for the external data request.
        def open(self, req, timeout):
            assert req.full_url.startswith("https://api.worldbank.org/")
            assert timeout <= 8
            return Response()

    monkeypatch.setattr(research, "build_opener", lambda *a: Opener())
    # Assert that `research.fetch_indicator('US', 'inflation') == (2024, 2.5)` holds when `valid`.
    if valid:
        assert research.fetch_indicator("US", "inflation") == (2024, 2.5)
    # Expect `ValueError` to be raised when no earlier condition matches.
    else:
        with pytest.raises(ValueError):
            research.fetch_indicator("US", "inflation")


# Verify that recommendations use the calculated business metric.
def test_recommendation():
    out = try_recommendation(
        pd.DataFrame({"Revenue": [100], "Advertising_Spend": [20]}),
        "Recommend actions based on ROAS",
    )
    assert "**5**" in out["answer"] and "No external benchmark" in out["answer"]


# Verify that invalid source metrics prevent recommendations.
def test_invalid_recommendation():
    assert (
        "Cannot support"
        in try_recommendation(
            pd.DataFrame({"Revenue": [100]}), "Recommend actions based on ROAS"
        )["answer"]
    )


# Verify that correlation polite.
def test_correlation_polite():
    out = try_planned_analysis(
        pd.DataFrame({"X": [1, 2, 3], "Y": [3, 2, 1]}),
        "Please calculate Pearson correlation of columns X and Y",
    )
    assert "-1.0000" in out["answer"]


# Verify that runner routes.
@pytest.mark.parametrize(
    "question,expected",
    [
        ("Research inflation for US", "World Bank"),
        ("Recommend actions based on ROAS", "Dataset finding"),
    ],
)
def test_runner_routes(tmp_path, monkeypatch, question, expected):
    import agent.agent_runner as runner

    path = tmp_path / "data.csv"
    pd.DataFrame({"Revenue": [100], "Advertising_Spend": [20]}).to_csv(
        path, index=False
    )
    monkeypatch.setattr(research, "fetch_indicator", lambda *a: (2024, 2.5))
    monkeypatch.setattr(
        runner, "run_agent", lambda *a, **kw: pytest.fail("Unexpected AI route")
    )
    assert expected in runner.run_analysis(str(path), question)["answer"]
