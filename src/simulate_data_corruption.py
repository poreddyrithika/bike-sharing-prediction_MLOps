"""Lab 5 extra: inject realistic faults into the raw data and confirm the schema catches them."""
import sys

import pandas as pd

from validate_data import validate_schema


def simulate_corrupted_data(df):
    """Simulate real-world data corruption across several columns."""
    print("[INFO] Injecting simulated faults into data...")
    corrupted_df = df.copy()

    # 1. Out-of-range numerical value (humidity is a 0-1 ratio)
    corrupted_df.loc[0, "hum"] = 1.9

    # 2. Invalid categorical value (season must be 1-4)
    corrupted_df.loc[1, "season"] = 7

    # 3. Invalid target (negative rental count)
    corrupted_df.loc[2, "cnt"] = -50

    # 4. Datatype mismatch (string inside a float column)
    corrupted_df["temp"] = corrupted_df["temp"].astype(object)
    corrupted_df.loc[3, "temp"] = "Very Hot"

    # 5. Inconsistent record (casual + registered != cnt)
    corrupted_df.loc[4, "registered"] = corrupted_df.loc[4, "registered"] + 500

    # 6. Schema inconsistency (unexpected extra column)
    corrupted_df["unexpected_column"] = 123

    # 7. Missing required attribute (dropping a column)
    corrupted_df = corrupted_df.drop(columns=["weathersit"])

    return corrupted_df


if __name__ == "__main__":
    data_path = "data/raw/day.csv"

    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")
        sys.exit(1)

    corrupted_df = simulate_corrupted_data(raw_df)

    print("\n--- Running Validation on Corrupted Data ---")
    passed = validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")

    # The corruption MUST be detected, so a "passed" validation is the failure here.
    if passed:
        print("[ERROR] Corrupted data was NOT detected by the schema!")
        sys.exit(1)
    print("[SUCCESS] All injected faults were detected by validation.")
