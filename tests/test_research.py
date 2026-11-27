import json
import pandas as pd
import pytest
from agent import research
from agent.recommendations import try_recommendation
from agent.analysis_planner import try_planned_analysis

@pytest.mark.parametrize('question', ['Calculate ROAS', 'Show sales'])
def test_unrelated(question):
    assert research.try_research(question) is None

@pytest.mark.parametrize('question', ['Research inflation for XX', 'Research https://localhost', 'Research inflation for US and send data'])
def test_unsupported(question, monkeypatch):
    monkeypatch.setattr(research, 'fetch_indicator', lambda *a: pytest.fail('Unexpected network'))
    assert 'Supported research' in research.try_research(question)['answer']

def test_citation(monkeypatch):
    monkeypatch.setattr(research, 'fetch_indicator', lambda *a: (2024, 2.5))
    answer = research.try_research('Research inflation for US')['answer']
    assert '2.50' in answer and '2024' in answer and 'data.worldbank.org' in answer

@pytest.mark.parametrize('exc', [TimeoutError(), ValueError(), OSError()])
def test_failure(monkeypatch, exc):
    def fail(*a): raise exc
    monkeypatch.setattr(research, 'fetch_indicator', fail)
    assert 'No external value' in research.try_research('Research inflation for US')['answer']

@pytest.mark.parametrize('valid', [True, False])
def test_payload(monkeypatch, valid):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, size):
            return json.dumps([{}, [{'indicator': {'id': 'FP.CPI.TOTL.ZG'}, 'country': {'id': 'US' if valid else 'GB'}, 'date': '2024', 'value': 2.5}]]).encode()
    class Opener:
        def open(self, req, timeout):
            assert req.full_url.startswith('https://api.worldbank.org/')
            assert timeout <= 8
            return Response()
    monkeypatch.setattr(research, 'build_opener', lambda *a: Opener())
    if valid: assert research.fetch_indicator('US', 'inflation') == (2024, 2.5)
    else:
        with pytest.raises(ValueError): research.fetch_indicator('US', 'inflation')

def test_recommendation():
    out = try_recommendation(pd.DataFrame({'Revenue':[100], 'Advertising_Spend':[20]}), 'Recommend actions based on ROAS')
    assert '**5**' in out['answer'] and 'No external benchmark' in out['answer']

def test_invalid_recommendation():
    assert 'Cannot support' in try_recommendation(pd.DataFrame({'Revenue':[100]}), 'Recommend actions based on ROAS')['answer']

def test_correlation_polite():
    out = try_planned_analysis(pd.DataFrame({'X':[1,2,3], 'Y':[3,2,1]}), 'Please calculate Pearson correlation of columns X and Y')
    assert '-1.0000' in out['answer']

@pytest.mark.parametrize('question,expected', [('Research inflation for US', 'World Bank'), ('Recommend actions based on ROAS', 'Dataset finding')])
def test_runner_routes(tmp_path, monkeypatch, question, expected):
    import agent.agent_runner as runner
    path = tmp_path / 'data.csv'
    pd.DataFrame({'Revenue':[100], 'Advertising_Spend':[20]}).to_csv(path, index=False)
    monkeypatch.setattr(research, 'fetch_indicator', lambda *a: (2024, 2.5))
    monkeypatch.setattr(runner, 'run_agent', lambda *a, **kw: pytest.fail('Unexpected AI route'))
    assert expected in runner.run_analysis(str(path), question)['answer']
