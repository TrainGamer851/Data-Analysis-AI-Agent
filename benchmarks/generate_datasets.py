from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"


def save_dataset(filename, data):
    df = pd.DataFrame(data)

    path = OUTPUT_DIR / filename

    df.to_csv(path, index=False)

    print(
        f"[BENCHMARK] Created {filename} | "
        f"Rows: {len(df)} | "
        f"Columns: {len(df.columns)}"
    )

    print(f"[BENCHMARK] Columns: {list(df.columns)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[BENCHMARK] Generating test datasets")

    # Dataset 1: Retail with unfamiliar column names.
    save_dataset(
        "retail_sales.csv",
        {
            "Product_Type": [
                "Clothing", "Clothing",
                "Electronics", "Electronics",
                "Home", "Home",
            ],
            "Revenue": [
                5000, 3000, 8000, 4000, 2000, 1000,
            ],
            "Net_Profit": [
                750, 450, 1200, 800, -100, 50,
            ],
            "Discount_Rate": [
                0.15, 0.25, 0.10, 0.05, 0.30, 0.20,
            ],
        },
    )

    # Dataset 2: Human resources.
    save_dataset(
        "human_resources.csv",
        {
            "Employee_ID": [
                "E001", "E002", "E003",
                "E004", "E005", "E006",
            ],
            "Department": [
                "Engineering", "Engineering",
                "Marketing", "Marketing",
                "Finance", "Finance",
            ],
            "Salary": [
                95000, 105000,
                70000, 80000,
                85000, 95000,
            ],
            "Years_Experience": [
                5, 7, 3, 5, 4, 6,
            ],
            "Satisfaction": [
                4.2, 4.5, 3.5, 4.0, 4.1, 4.3,
            ],
        },
    )

    # Dataset 3: Inventory management.
    save_dataset(
        "inventory.csv",
        {
            "SKU": [
                "P001", "P002", "P003",
                "P004", "P005", "P006",
            ],
            "Product": [
                "Laptop", "Monitor", "Keyboard",
                "Mouse", "Printer", "Headphones",
            ],
            "Stock_On_Hand": [
                5, 30, 12, 50, 3, 25,
            ],
            "Reorder_Level": [
                10, 20, 15, 20, 8, 15,
            ],
            "Unit_Cost": [
                800, 200, 40, 20, 150, 60,
            ],
        },
    )

    # Dataset 4: Marketing performance.
    save_dataset(
        "marketing.csv",
        {
            "Campaign": [
                "Search", "Social", "Email",
                "Display", "Referral",
            ],
            "Advertising_Spend": [
                1000, 1500, 500, 2000, 800,
            ],
            "Generated_Revenue": [
                4000, 4500, 3000, 2500, 2400,
            ],
            "Impressions": [
                20000, 50000, 10000, 80000, 15000,
            ],
            "Conversions": [
                100, 90, 80, 50, 60,
            ],
        },
    )

    # Dataset 5: Financial transactions.
    save_dataset(
        "financial_transactions.csv",
        {
            "Transaction_Date": [
                "2026-01-05", "2026-01-15",
                "2026-02-05", "2026-02-15",
                "2026-03-05", "2026-03-15",
            ],
            "Transaction_Type": [
                "Income", "Expense",
                "Income", "Expense",
                "Income", "Expense",
            ],
            "Amount": [
                5000, 3000,
                6000, 3500,
                7000, 4000,
            ],
        },
    )

    print("[BENCHMARK] Dataset generation completed")
    print(f"[BENCHMARK] Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()