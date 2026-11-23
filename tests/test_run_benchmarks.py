"""Tests for the benchmark scorer (evaluation/run_benchmarks.py). No LLM needed."""
import json
import os

import pytest

from evaluation.run_benchmarks import run_benchmark, score, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(ROOT, "evaluation", "benchmark_questions.json")


@pytest.fixture(scope="module")
def questions():
    with open(QUESTIONS_PATH) as fh:
        return {q["id"]: q for q in json.load(fh)}


def test_answer_key_scores_100_percent(questions):
    """Feeding each expected answer back in, as the agent would print it, must pass."""
    for qid, q in questions.items():
        correct, reason = score(q, str(q["expected"]))
        assert correct, f"{qid}: {reason}"


@pytest.mark.parametrize("qid, answer", [
    ("Q01", "2326534.3496"),                       # unrounded, within tolerance
    ("Q09", "[{'Segment': 'Consumer', 'Sales': 1170659.79}, "
            "{'Segment': 'Corporate', 'Sales': 715806.13}, "
            "{'Segment': 'Home Office', 'Sales': 440068.43}]"),  # Series as rows
    ("Q11", "[{'State': 'California', 'Sales': 457687.63}, {'State': 'New York', 'Sales': 1.0}]"),
    ("Q11", "California had the highest sales."),
    ("Q12", "[{'Customer Name': 'Sean Miller', 'Sales': 1}, {'Customer Name': 'Tamara Chand', 'Sales': 1}, "
            "{'Customer Name': 'Raymond Buch', 'Sales': 1}, {'Customer Name': 'Tom Ashbrook', 'Sales': 1}, "
            "{'Customer Name': 'Adrian Barton', 'Sales': 1}]"),
    ("Q14", "['Tables', 'Bookcases', 'Supplies']"),  # any order for a non-ranking list
    ("Q20", "11"),
    ("Q20", "[{'Order Date': 11, 'Sales': 1.0}]"),
    ("Q24", "The dataset does not contain a marketing spend column."),
])
def test_other_correct_formats_pass(questions, qid, answer):
    correct, reason = score(questions[qid], answer)
    assert correct, reason


@pytest.mark.parametrize("qid, answer", [
    ("Q01", "2326000"),
    ("Q07", "{'category': 'Furniture', 'total_sales': 741999.8}"),
    ("Q11", "[{'State': 'New York', 'Sales': 2.0}, {'State': 'California', 'Sales': 1.0}]"),
    ("Q12", "['Tamara Chand', 'Sean Miller', 'Raymond Buch', 'Tom Ashbrook', 'Adrian Barton']"),
    ("Q14", "['Bookcases', 'Tables']"),
    ("Q15", "0.1256"),                              # a fraction, not a percentage
    ("Q20", "10"),
    ("Q24", "12345.0"),                             # invented a number
    ("Q24", "Marketing spend was high."),
])
def test_wrong_answers_fail(questions, qid, answer):
    correct, _ = score(questions[qid], answer)
    assert not correct


def test_errors(questions):
    assert not score(questions["Q01"], None, "NRP_API_KEY is missing.")[0]
    # Giving up on an unanswerable question counts as declining...
    assert score(questions["Q24"], None, "Agent stopped: LimitsExceeded")[0]
    # ...but a crash (e.g. no API key) does not.
    assert not score(questions["Q24"], None, "NRP_API_KEY is missing.")[0]
    # The agent's code setting result = None comes back as the text "None": a decline.
    assert score(questions["Q24"], "None", None)[0]
    # For an answerable question, result = None is still wrong.
    assert not score(questions["Q01"], "None", None)[0]


def test_run_benchmark_and_summary(questions):
    picked = [questions["Q01"], questions["Q03"], questions["Q24"]]
    answers = {
        picked[0]["question"]: {"answer": "2326534.35", "error": None},
        picked[1]["question"]: {"answer": "1", "error": None},
        picked[2]["question"]: {"answer": None, "error": "Agent stopped: LimitsExceeded"},
    }
    results = run_benchmark(picked, "unused.csv", lambda csv, q: answers[q])
    assert [r["correct"] for r in results] == [True, False, True]

    summary = summarize(results)
    assert summary["overall"] == {"correct": 2, "total": 3, "accuracy": 66.7}
    assert summary["by_category"]["aggregation"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert summary["errors"] == 1
