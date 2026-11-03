import os
import sys

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def _counts_consistent(df):
    """casual + registered must equal cnt (row-wise). Other checks report type problems."""
    try:
        return (df["casual"] + df["registered"]) == df["cnt"]
    except Exception:
        return True


def get_bike_schema():
    """Strict schema specification for the Bike Sharing dataset.

    `yr` accepts 0 (2011), 1 (2012) and 2 (2013) because incoming batches
    (Lab 13) continue the series into 2013.
    """
    return pa.DataFrameSchema(
        {
            "instant": Column(pa.Int, Check.ge(1), unique=True),
            "dteday": Column(
                pa.String,
                [Check(lambda s: pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").notna(),
                       error="dteday must be a YYYY-MM-DD date")],
                unique=True,
            ),
            "season": Column(pa.Int, Check.isin([1, 2, 3, 4])),
            "yr": Column(pa.Int, Check.isin([0, 1, 2])),
            "mnth": Column(pa.Int, Check.in_range(1, 12)),
            "holiday": Column(pa.Int, Check.isin([0, 1])),
            "weekday": Column(pa.Int, Check.in_range(0, 6)),
            "workingday": Column(pa.Int, Check.isin([0, 1])),
            "weathersit": Column(pa.Int, Check.in_range(1, 4)),
            "temp": Column(pa.Float, Check.in_range(0.0, 1.0)),
            "atemp": Column(pa.Float, Check.in_range(0.0, 1.0)),
            "hum": Column(pa.Float, Check.in_range(0.0, 1.0)),
            "windspeed": Column(pa.Float, Check.in_range(0.0, 1.0)),
            "casual": Column(pa.Int, Check.ge(0)),
            "registered": Column(pa.Int, Check.ge(0)),
            "cnt": Column(pa.Int, Check.ge(0)),
        },
        checks=[Check(_counts_consistent, error="casual + registered must equal cnt")],
        strict=True,
    )


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validate a DataFrame against the schema. Returns True when clean."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_bike_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    data_path = "data/raw/day.csv"
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")
        sys.exit(0 if passed else 1)
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)
