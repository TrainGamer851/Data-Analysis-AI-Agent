"""Build evaluation/benchmark_questions.json with expected answers computed from the CSV.

Each question carries the pandas reference code used to compute its expected answer,
so anyone on the team can re-run it and check the answer key.

Usage (from the repo root):
    python evaluation/build_benchmarks.py data/superstore_orders.csv evaluation/benchmark_questions.json
"""

import json
import sys

import numpy as np
import pandas as pd

# (id, category, difficulty, question, answer_type, tolerance, reference_code)
# reference_code must assign `expected`. `df` is the raw CSV; dates are still strings.
# tolerance: absolute tolerance for numbers (also applied to each number in dict answers).
QUESTIONS = [
    # ---------- Easy: single aggregation ----------
    (
        "Q01",
        "aggregation",
        "easy",
        "What are the total sales across all orders?",
        "number",
        0.01,
        "expected = round(df['Sales'].sum(), 2)",
    ),
    (
        "Q02",
        "aggregation",
        "easy",
        "What is the total profit across all orders?",
        "number",
        0.01,
        "expected = round(df['Profit'].sum(), 2)",
    ),
    (
        "Q03",
        "aggregation",
        "easy",
        "How many unique orders are in the dataset?",
        "number",
        0,
        "expected = int(df['Order ID'].nunique())",
    ),
    (
        "Q04",
        "aggregation",
        "easy",
        "How many unique customers are there, counted by Customer ID?",
        "number",
        0,
        "expected = int(df['Customer ID'].nunique())",
    ),
    (
        "Q05",
        "aggregation",
        "easy",
        "What is the total quantity of items sold?",
        "number",
        0,
        "expected = int(df['Quantity'].sum())",
    ),
    (
        "Q06",
        "aggregation",
        "easy",
        "How many unique orders were shipped using Same Day shipping?",
        "number",
        0,
        "expected = int(df.loc[df['Ship Mode'] == 'Same Day', 'Order ID'].nunique())",
    ),
    # ---------- Medium: grouping and ranking ----------
    (
        "Q07",
        "grouping",
        "medium",
        "Which product category has the highest total sales, and what is that total?",
        "dict",
        0.01,
        "s = df.groupby('Category')['Sales'].sum()\n"
        "expected = {'category': s.idxmax(), 'total_sales': round(s.max(), 2)}",
    ),
    (
        "Q08",
        "grouping",
        "medium",
        "Which region has the highest total profit, and what is that total?",
        "dict",
        0.01,
        "s = df.groupby('Region')['Profit'].sum()\n"
        "expected = {'region': s.idxmax(), 'total_profit': round(s.max(), 2)}",
    ),
    (
        "Q09",
        "grouping",
        "medium",
        "What are the total sales for each customer segment?",
        "dict",
        0.01,
        "expected = df.groupby('Segment')['Sales'].sum().round(2).to_dict()",
    ),
    (
        "Q10",
        "grouping",
        "medium",
        "Which sub-category has the lowest total profit, and what is that total?",
        "dict",
        0.01,
        "s = df.groupby('Sub-Category')['Profit'].sum()\n"
        "expected = {'sub_category': s.idxmin(), 'total_profit': round(s.min(), 2)}",
    ),
    (
        "Q11",
        "grouping",
        "medium",
        "Which state or province has the highest total sales?",
        "text",
        None,
        "expected = df.groupby('State/Province')['Sales'].sum().idxmax()",
    ),
    (
        "Q12",
        "ranking",
        "medium",
        "Who are the top 5 customers by total sales, from highest to lowest?",
        "list",
        None,
        "expected = df.groupby('Customer Name')['Sales'].sum().nlargest(5).index.tolist()",
    ),
    (
        "Q13",
        "ranking",
        "medium",
        "Which product, grouped by Product Name, has the highest total quantity sold?",
        "text",
        None,
        "expected = df.groupby('Product Name')['Quantity'].sum().idxmax()",
    ),
    (
        "Q14",
        "grouping",
        "medium",
        "Which sub-categories have a negative total profit?",
        "list",
        None,
        "s = df.groupby('Sub-Category')['Profit'].sum()\n"
        "expected = sorted(s[s < 0].index.tolist())",
    ),
    # ---------- Medium: business ratios ----------
    (
        "Q15",
        "ratio",
        "medium",
        "What is the overall profit margin as a percentage of sales?",
        "number",
        0.01,
        "expected = round(df['Profit'].sum() / df['Sales'].sum() * 100, 2)",
    ),
    (
        "Q16",
        "ratio",
        "medium",
        "What is the profit margin percentage for each product category?",
        "dict",
        0.01,
        "g = df.groupby('Category')[['Sales', 'Profit']].sum()\n"
        "expected = (g['Profit'] / g['Sales'] * 100).round(2).to_dict()",
    ),
    (
        "Q17",
        "ratio",
        "medium",
        "What is the average order value (total sales divided by the number of unique orders)?",
        "number",
        0.01,
        "expected = round(df['Sales'].sum() / df['Order ID'].nunique(), 2)",
    ),
    # ---------- Medium/hard: dates and trends ----------
    (
        "Q18",
        "time",
        "medium",
        "Which year had the highest total sales, and what was that total?",
        "dict",
        0.01,
        "y = pd.to_datetime(df['Order Date'], format='%m/%d/%y').dt.year\n"
        "s = df.groupby(y)['Sales'].sum()\n"
        "expected = {'year': int(s.idxmax()), 'total_sales': round(s.max(), 2)}",
    ),
    (
        "Q19",
        "time",
        "hard",
        "What was the year-over-year sales growth percentage from 2025 to 2026?",
        "number",
        0.01,
        "y = pd.to_datetime(df['Order Date'], format='%m/%d/%y').dt.year\n"
        "s = df.groupby(y)['Sales'].sum()\n"
        "expected = round((s[2026] - s[2025]) / s[2025] * 100, 2)",
    ),
    (
        "Q20",
        "time",
        "hard",
        "In 2026, which month had the highest total sales?",
        "text",
        None,
        "d = pd.to_datetime(df['Order Date'], format='%m/%d/%y')\n"
        "s = df[d.dt.year == 2026].groupby(d.dt.month)['Sales'].sum()\n"
        "expected = pd.Timestamp(2026, int(s.idxmax()), 1).month_name()",
    ),
    (
        "Q21",
        "time",
        "hard",
        "What is the average number of days between order date and ship date?",
        "number",
        0.01,
        "o = pd.to_datetime(df['Order Date'], format='%m/%d/%y')\n"
        "sh = pd.to_datetime(df['Ship Date'], format='%m/%d/%y')\n"
        "expected = round((sh - o).dt.days.mean(), 2)",
    ),
    # ---------- Hard: filter + group, multi-step ----------
    (
        "Q22",
        "multi_step",
        "hard",
        "In the West region, which sub-category has the lowest total profit?",
        "text",
        None,
        "w = df[df['Region'] == 'West']\n"
        "expected = w.groupby('Sub-Category')['Profit'].sum().idxmin()",
    ),
    (
        "Q23",
        "multi_step",
        "hard",
        "What is the total profit on line items with a discount greater than 20%?",
        "number",
        0.01,
        "expected = round(df.loc[df['Discount'] > 0.2, 'Profit'].sum(), 2)",
    ),
    # ---------- Unanswerable: the agent should say the data is not available ----------
    (
        "Q24",
        "unanswerable",
        "medium",
        "What was the total marketing spend in 2025?",
        "unanswerable",
        None,
        "expected = 'The dataset has no marketing or advertising spend column.'",
    ),
    (
        "Q25",
        "unanswerable",
        "medium",
        "What is the average age of our customers?",
        "unanswerable",
        None,
        "expected = 'The dataset has no customer age or birth date column.'",
    ),
]


# Convert numpy/pandas types to plain JSON-safe Python values.
def to_plain(value):
    """Convert numpy/pandas types to plain JSON-safe Python values."""
    # Return the selected result when `isinstance(value, dict)`.
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    # Return the selected result when `isinstance(value, (list, tuple))`.
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    # Return the selected result when `isinstance(value, np.integer)`.
    if isinstance(value, np.integer):
        return int(value)
    # Return the selected result when `isinstance(value, np.floating)`.
    if isinstance(value, np.floating):
        return float(value)
    return value


# Run the module's command-line workflow.
def main(csv_path, out_path):
    df = pd.read_csv(csv_path)
    items = []
    # Process each `(qid, category, difficulty, question, answer_type, tol, code)` from `QUESTIONS`.
    for qid, category, difficulty, question, answer_type, tol, code in QUESTIONS:
        scope = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, scope)
        items.append(
            {
                "id": qid,
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "answer_type": answer_type,
                "expected": to_plain(scope["expected"]),
                "tolerance": tol,
                "reference_code": code,
            }
        )
    with open(out_path, "w") as fh:
        json.dump(items, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {len(items)} questions to {out_path}")


# Call `main` when `__name__ == '__main__'`.
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
