"""Run the agent on every benchmark question and report accuracy (the Stage 4 number).

For each question in benchmark_questions.json this calls run_analysis() on the CSV,
scores the answer against the expected answer, and prints accuracy overall, by
category and by difficulty. Per-question results are saved as JSON.

Usage (from the repo root; needs NRP_API_KEY in .env):
    python evaluation/run_benchmarks.py
    python evaluation/run_benchmarks.py --ids Q01 Q07 Q24
    python evaluation/run_benchmarks.py --csv data/superstore_orders.csv \
        --questions evaluation/benchmark_questions.json --out evaluation/results

To rebuild the answer key itself, use evaluation/build_benchmarks.py.
"""
import argparse
import ast
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

# Allow `python evaluation/run_benchmarks.py` from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_CSV = os.path.join(ROOT, "data", "superstore_orders.csv")
DEFAULT_QUESTIONS = os.path.join(ROOT, "evaluation", "benchmark_questions.json")
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results")

# Small slack on top of each question's tolerance, for float rounding.
EPSILON = 1e-6

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

# Phrases that count as the agent declining an unanswerable question.
DECLINE_MARKERS = [
    "no ", "not ", "n't", "missing", "unavailable", "cannot", "unable",
    "does not exist", "not available", "not in the dataset", "no column",
]


# ---------------------------------------------------------------------------
# Parsing the agent's answer
# ---------------------------------------------------------------------------

def parse_answer(answer):
    """Turn run_analysis()'s answer text back into a Python value when possible.

    The agent returns str(result), e.g. "2326534.35", "['A', 'B']" or a list of
    row dicts for a DataFrame/Series. Free text stays a string.
    """
    if answer is None:
        return None
    if not isinstance(answer, str):
        return answer
    text = answer.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text


def _as_number(value):
    """Return value as a float if it is a real number (or numeric string), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").rstrip("%").strip()
        try:
            number = float(cleaned)
        except ValueError:
            return None
        return None if math.isnan(number) else number
    return None


def flatten(value):
    """Return (strings, numbers) found in value, in order. Dict keys are included."""
    strings, numbers = [], []

    def walk(item):
        if isinstance(item, dict):
            for key, sub in item.items():
                walk(key)
                walk(sub)
        elif isinstance(item, (list, tuple, set)):
            for sub in item:
                walk(sub)
        else:
            number = _as_number(item)
            if number is not None:
                numbers.append(number)
            elif isinstance(item, str):
                strings.append(item)
                # Numbers written inside free text, e.g. "Total sales were 2,326,534.35".
                for match in re.findall(r"-?\$?\d[\d,]*\.?\d*", item):
                    found = _as_number(match)
                    if found is not None:
                        numbers.append(found)

    walk(value)
    return strings, numbers


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _text_aliases(expected):
    """Equivalent spellings of an expected string (month names accept 11 / Nov)."""
    aliases = {_norm(expected)}
    if _norm(expected) in MONTHS:
        index = MONTHS.index(_norm(expected)) + 1
        aliases.update({str(index), f"{index:02d}", _norm(expected)[:3]})
    return aliases


def _number_matches(expected, numbers, tolerance):
    return any(abs(n - expected) <= tolerance + EPSILON for n in numbers)


def _string_present(expected, strings):
    aliases = _text_aliases(expected)
    return any(_norm(s) in aliases for s in strings)


# ---------------------------------------------------------------------------
# Scoring one question
# ---------------------------------------------------------------------------

def score(question, answer, error=None):
    """Return (correct: bool, reason: str) for one question."""
    answer_type = question["answer_type"]
    expected = question["expected"]
    tolerance = question.get("tolerance") or 0
    value = parse_answer(answer)

    if answer_type == "unanswerable":
        return _score_unanswerable(answer, value, error)

    if value is None:
        return False, f"no answer ({error})" if error else "no answer"

    strings, numbers = flatten(value)

    if answer_type == "number":
        if _number_matches(float(expected), numbers, tolerance):
            return True, "number matches"
        return False, f"expected {expected} (±{tolerance}), got numbers {numbers[:5]}"

    if answer_type == "text":
        return _score_text(expected, value, strings, numbers)

    if answer_type == "list":
        ordered = question.get("category") == "ranking"
        return _score_list(expected, strings, ordered)

    if answer_type == "dict":
        return _score_dict(expected, strings, numbers, tolerance)

    return False, f"unknown answer_type {answer_type!r}"


def _score_text(expected, value, strings, numbers):
    aliases = _text_aliases(expected)
    if isinstance(value, str):
        # A bare answer must equal the expected value; prose must mention it.
        if _norm(value) in aliases:
            return True, "text matches"
        if re.search(r"\b" + re.escape(_norm(expected)) + r"\b", _norm(value)):
            return True, "expected text found in answer"
        return False, f"expected {expected!r}, got {value!r}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ok = _norm(str(int(value)) if float(value).is_integer() else str(value)) in aliases
        return ok, "text matches" if ok else f"expected {expected!r}, got {value!r}"
    # A container (e.g. a sorted Series): the top label must be the expected one.
    first = _first_label(value)
    if first is not None and _norm(_label_text(first)) in aliases:
        return True, "top row matches"
    return False, f"expected {expected!r} as the top answer, got {first!r}"


def _label_text(label):
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label)


def _first_label(value):
    """The label of the top entry in a container answer.

    [{'State': 'California', 'Sales': 1.0}, ...] (a DataFrame/Series) -> 'California'
    [{'Month': 11, 'Sales': 1.0}, ...]                                -> 11
    {'California': 1.0, ...} (a dict)                                 -> 'California'
    {'state': 'California', 'sales': 1.0} (a dict)                    -> 'California'
    ['California', ...]                                               -> 'California'
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        head = value[0]
        if isinstance(head, dict):
            return next(iter(head.values()), None)
        return _first_label(head)
    if isinstance(value, dict):
        if not value:
            return None
        key, sub = next(iter(value.items()))
        if isinstance(sub, str) and _as_number(sub) is None:
            return sub
        return key
    return value


def _score_list(expected, strings, ordered):
    wanted = [_norm(e) for e in expected]
    seen = []
    for s in strings:
        n = _norm(s)
        if n in wanted and n not in seen:
            seen.append(n)
    if ordered:
        if seen == wanted:
            return True, "list matches in order"
        return False, f"expected {expected} in order, got {seen}"
    if set(seen) == set(wanted):
        return True, "list matches"
    return False, f"expected {sorted(expected)}, got {sorted(seen)}"


def _score_dict(expected, strings, numbers, tolerance):
    """Every expected value must appear in the answer.

    Keys like 'category' or 'total_sales' are just labels for the answer's parts and
    are not checked. Keys that are data labels (e.g. 'Consumer', 'Office Supplies')
    must appear in the answer too.
    """
    missing = []
    for key, val in expected.items():
        if not re.fullmatch(r"[a-z_]+", key) and not _string_present(key, strings):
            missing.append(key)
        number = _as_number(val)
        if number is not None:
            if not _number_matches(number, numbers, tolerance):
                missing.append(f"{key}={val}")
        elif not _string_present(val, strings):
            missing.append(f"{key}={val!r}")
    if not missing:
        return True, "all expected values found"
    return False, f"missing {missing}"


def _score_unanswerable(answer, value, error):
    """Correct when the agent declines instead of inventing a number."""
    if answer is not None and (value is None or (isinstance(value, str) and not value.strip())):
        # The code ran and deliberately set result = None (or "") - no value invented.
        return True, "agent returned no value (result = None)"
    if value is None:
        # The agent kept failing because the data is not there, and gave up.
        if error and error.startswith("Agent stopped"):
            return True, "agent declined (no valid result)"
        return False, f"run failed: {error}"
    if isinstance(value, str) and _as_number(value) is None:
        text = " " + _norm(value) + " "
        if any(marker in text for marker in DECLINE_MARKERS):
            return True, "agent said the data is not available"
        return False, f"answer did not decline: {value!r}"
    return False, f"invented an answer: {str(value)[:80]!r}"


# ---------------------------------------------------------------------------
# Running the benchmark
# ---------------------------------------------------------------------------

def run_benchmark(questions, csv_path, analyze):
    """Run analyze(csv_path, question) on each question and score it."""
    results = []
    for index, q in enumerate(questions, 1):
        start = time.perf_counter()
        try:
            output = analyze(csv_path, q["question"])
        except Exception as exc:  # run_analysis catches its own errors; this is a backstop
            output = {"answer": None, "error": f"{type(exc).__name__}: {exc}"}
        seconds = time.perf_counter() - start
        answer, error = output.get("answer"), output.get("error")
        correct, reason = score(q, answer, error)
        results.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "expected": q["expected"],
            "answer": answer,
            "error": error,
            "correct": correct,
            "reason": reason,
            "seconds": round(seconds, 2),
        })
        mark = "PASS" if correct else "FAIL"
        print(f"[{index:>2}/{len(questions)}] {q['id']} {mark} ({seconds:.1f}s) {reason}", flush=True)
    return results


def summarize(results):
    """Accuracy overall and broken down by category and difficulty."""
    def accuracy(rows):
        correct = sum(r["correct"] for r in rows)
        return {"correct": correct, "total": len(rows),
                "accuracy": round(100 * correct / len(rows), 1) if rows else 0.0}

    def by(field):
        groups = defaultdict(list)
        for r in results:
            groups[r[field]].append(r)
        return {name: accuracy(rows) for name, rows in sorted(groups.items())}

    return {
        "overall": accuracy(results),
        "by_category": by("category"),
        "by_difficulty": by("difficulty"),
        "errors": sum(1 for r in results if r["error"]),
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 2) if results else 0.0,
    }


def print_report(summary):
    overall = summary["overall"]
    print("\n" + "=" * 50)
    print(f"ACCURACY: {overall['correct']}/{overall['total']} = {overall['accuracy']}%")
    print("=" * 50)
    for title, key in (("By category", "by_category"), ("By difficulty", "by_difficulty")):
        print(f"\n{title}:")
        for name, stats in summary[key].items():
            print(f"  {name:<14} {stats['correct']:>2}/{stats['total']:<2}  {stats['accuracy']:>5}%")
    print(f"\nRuns that returned an error: {summary['errors']}")
    print(f"Average time per question: {summary['avg_seconds']}s")


def display_path(path):
    """Repo-relative path with forward slashes, so saved results hold no local usernames."""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:  # different drive on Windows: no relative path exists
        return os.path.basename(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="dataset CSV")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="benchmark JSON")
    parser.add_argument("--out", default=DEFAULT_OUT, help="folder for the results JSON")
    parser.add_argument("--ids", nargs="*", help="only run these question IDs, e.g. Q01 Q07")
    args = parser.parse_args(argv)

    with open(args.questions) as fh:
        questions = json.load(fh)
    if args.ids:
        wanted = {i.upper() for i in args.ids}
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            parser.error(f"no questions match {args.ids}")

    # Fail fast if the LLM is not configured, instead of 25 identical errors.
    from agent.llm_client import get_model_config
    try:
        model = get_model_config().get("model")
    except Exception as exc:
        sys.exit(f"LLM is not configured: {exc}\nSet NRP_API_KEY (see .env.example).")

    from agent.agent_runner import run_analysis

    print(f"Running {len(questions)} questions with model {model} on {args.csv}\n")
    results = run_benchmark(questions, args.csv, run_analysis)
    summary = summarize(results)
    print_report(summary)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out, f"benchmark_results_{stamp}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": model, "csv": display_path(args.csv), "run_at": stamp,
                   "summary": summary, "results": results}, fh, indent=2, default=str)
        fh.write("\n")
    print(f"\nSaved results to {out_path}")
    return summary


if __name__ == "__main__":
    main()
