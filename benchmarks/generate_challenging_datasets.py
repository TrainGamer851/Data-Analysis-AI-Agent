from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "challenging_datasets"
)


def save_dataset(filename, data):
    """Save a reproducible benchmark dataset."""

    timer = perf_counter()

    df = pd.DataFrame(data)

    path = OUTPUT_DIR / filename

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    elapsed = perf_counter() - timer

    print(
        f"[BENCHMARK] Created {filename} | "
        f"Rows: {len(df):,} | "
        f"Columns: {len(df.columns)}"
    )

    print(
        f"[BENCHMARK] Columns: "
        f"{list(df.columns)}"
    )

    print(
        f"[BENCHMARK] Missing values: "
        f"{int(df.isna().sum().sum())}"
    )

    print(
        f"[BENCHMARK] File size: "
        f"{path.stat().st_size / 1024:.2f} KB"
    )

    print(
        f"[BENCHMARK] Generation completed "
        f"in {elapsed:.3f}s"
    )

    print("-" * 60)


def generate_ambiguous_revenue():
    save_dataset(
        "ambiguous_revenue.csv",
        {
            "Product_Type": [
                "Electronics",
                "Furniture",
                "Clothing",
                "Office Supplies",
            ],
            "Sales": [
                10000, 8000, 6000, 4000,
            ],
            "Revenue": [
                9000, 7000, 5500, 3500,
            ],
            "Net_Profit": [
                2000, 1000, 750, 250,
            ],
        },
    )


def generate_formatted_financials():
    save_dataset(
        "formatted_financials.csv",
        {
            "Department": [
                "Engineering",
                "Marketing",
                "Finance",
                "Operations",
            ],
            "Revenue_USD": [
                "$12,500.00",
                "$8,000.00",
                "$10,500.00",
                "$4,000.00",
            ],
            "Net_Profit_USD": [
                "$2,500.00",
                "($500.00)",
                "$1,500.00",
                "$750.00",
            ],
        },
    )


def generate_mixed_currencies():
    save_dataset(
        "mixed_currencies.csv",
        {
            "Region": [
                "United States",
                "Germany",
                "United Kingdom",
                "United States",
            ],
            "Revenue": [
                "USD 10000",
                "EUR 8000",
                "GBP 6000",
                "USD 5000",
            ],
            "Quantity": [
                100, 80, 60, 50,
            ],
        },
    )


def generate_invalid_numeric_values():
    save_dataset(
        "invalid_numeric_values.csv",
        {
            "Product": [
                "Laptop",
                "Monitor",
                "Keyboard",
                "Mouse",
                "Printer",
                "Headphones",
            ],
            "Revenue_USD": [
                "5000",
                "unknown",
                "3000",
                None,
                "2000",
                "inf",
            ],
            "Quantity": [
                10, 20, 30, 40, 50, 60,
            ],
        },
    )


def generate_ambiguous_dates():
    save_dataset(
        "ambiguous_dates.csv",
        {
            "Order_Date": [
                "01/02/2026",
                "03/04/2026",
                "05/06/2026",
                "07/08/2026",
            ],
            "Revenue_USD": [
                1000, 2000, 3000, 4000,
            ],
            "Net_Profit_USD": [
                200, 400, 600, 800,
            ],
        },
    )

    save_dataset(
        "conflicting_dates.csv",
        {
            "Order_Date": [
                "25/01/2026",
                "02/15/2026",
                "03/04/2026",
                "05/06/2026",
            ],
            "Revenue_USD": [
                1000, 2000, 3000, 4000,
            ],
        },
    )


def generate_mixed_discount_scales():
    save_dataset(
        "mixed_discount_scales.csv",
        {
            "Product_Type": [
                "Electronics",
                "Furniture",
                "Clothing",
                "Office Supplies",
            ],
            "Discount": [
                0.15, 20, 0.10, 30,
            ],
            "Revenue_USD": [
                10000, 8000, 6000, 4000,
            ],
        },
    )


def generate_unfamiliar_schema():
    save_dataset(
        "unfamiliar_schema.csv",
        {
            "Merchandise_Family": [
                "Electronics",
                "Electronics",
                "Furniture",
                "Furniture",
                "Clothing",
                "Clothing",
            ],
            "Billed_Value_USD": [
                12000, 8000,
                5000, 7000,
                4000, 6000,
            ],
            "Direct_Expenditure_USD": [
                9000, 5000,
                4000, 5500,
                3000, 4500,
            ],
            "Items_Dispatched": [
                120, 80, 50, 70, 40, 60,
            ],
        },
    )


def generate_large_business_dataset():
    row_count = 50000

    row_numbers = np.arange(row_count)

    categories = np.array([
        "Electronics",
        "Furniture",
        "Clothing",
        "Office Supplies",
    ])

    regions = np.array([
        "West",
        "East",
        "Central",
        "South",
    ])

    revenue = (
        (row_numbers % 100 + 1) * 10
    )

    profit = revenue // 5

    dates = (
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(
            row_numbers % 365,
            unit="D",
        )
    )

    save_dataset(
        "large_business_dataset.csv",
        {
            "Order_ID": [
                f"ORD-{i + 1:06d}"
                for i in row_numbers
            ],
            "Order_Date": dates.strftime(
                "%Y-%m-%d"
            ),
            "Product_Type": categories[
                row_numbers % 4
            ],
            "Region": regions[
                (row_numbers // 4) % 4
            ],
            "Revenue_USD": revenue,
            "Net_Profit_USD": profit,
            "Quantity": (
                row_numbers % 10 + 1
            ),
            "Discount_Pct": (
                row_numbers % 5 * 5
            ),
        },
    )

    print(
        "[BENCHMARK] Large dataset "
        "expected results:"
    )

    print(
        f"[BENCHMARK] Total revenue: "
        f"${revenue.sum():,.2f}"
    )

    print(
        f"[BENCHMARK] Total net profit: "
        f"${profit.sum():,.2f}"
    )

    print(
        f"[BENCHMARK] Net profit margin: "
        f"{profit.sum() / revenue.sum() * 100:.2f}%"
    )


def main():
    timer = perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[BENCHMARK] Starting challenging "
        "dataset generation"
    )

    generate_ambiguous_revenue()
    generate_formatted_financials()
    generate_mixed_currencies()
    generate_invalid_numeric_values()
    generate_ambiguous_dates()
    generate_mixed_discount_scales()
    generate_unfamiliar_schema()
    generate_large_business_dataset()

    elapsed = perf_counter() - timer

    print(
        "[BENCHMARK] All 9 challenging "
        "datasets generated"
    )

    print(
        f"[BENCHMARK] Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"[BENCHMARK] Total generation time: "
        f"{elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
