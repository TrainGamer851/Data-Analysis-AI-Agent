import argparse
import json

import pandas as pd

from agent.dataset_schema import profile_dataset


# Run the module's command-line workflow.
def main():
    parser = argparse.ArgumentParser(description="Inspect a business CSV schema.")

    parser.add_argument(
        "csv_path",
        help="Path to the CSV dataset",
    )

    args = parser.parse_args()

    try:
        dataframe = pd.read_csv(args.csv_path)

    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            args.csv_path,
            encoding="latin1",
        )

    schema = profile_dataset(dataframe)

    print("\n" + "=" * 60)
    print("BUSINESS DATASET SCHEMA")
    print("=" * 60)

    print(
        json.dumps(
            schema,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


# Call `main` when `__name__ == '__main__'`.
if __name__ == "__main__":
    main()
