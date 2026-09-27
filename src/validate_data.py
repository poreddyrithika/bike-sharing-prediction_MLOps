import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def get_bike_schema():
    """Defines the strict schema specification for the Bike Sharing dataset."""
    return pa.DataFrameSchema({
        "instant": Column(pa.Int, Check.ge(1)),
        "dteday": Column(pa.String),
        "season": Column(pa.Int, Check.isin([1, 2, 3, 4])),
        "yr": Column(pa.Int, Check.isin([0, 1])),
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
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
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
        validate_schema(raw_df, output_report_name="baseline_validation.csv")
    else:
        print(f"[ERROR] Target file not found at: {data_path}")
