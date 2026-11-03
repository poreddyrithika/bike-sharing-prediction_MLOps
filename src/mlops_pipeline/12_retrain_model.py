"""Retrain a candidate V2 on the updated (post-drift) dataset; log it in MLflow (not registered yet)."""
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from common import build_preprocessor, clean_dataframe, load_config, regression_metrics, write_json
from mlflow_utils import configure_mlflow

UPDATED = Path("data/processed/updated_training_data.csv")
MODEL_V2, PREP_V2 = Path("models/retrained_model_v2.pkl"), Path("models/retrained_preprocessor_v2.pkl")
REPORT = Path("outputs/retraining/retraining_report.json")


def main():
    config = load_config()
    configure_mlflow(config)
    for p in (UPDATED, Path("data/processed/X_test_raw.csv"), Path("data/processed/y_test.npy")):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}. Run 12_create_updated_dataset.py first.")

    updated = pd.read_csv(UPDATED)
    X_train, y_train = clean_dataframe(updated.drop(columns=["cnt"]), config), updated["cnt"].to_numpy(dtype=float)
    X_test, y_test = clean_dataframe(pd.read_csv("data/processed/X_test_raw.csv"), config), np.load("data/processed/y_test.npy")
    evolved = pd.read_csv("data/monitoring/drift/evolved_labeled_data.csv")
    X_evo, y_evo = clean_dataframe(evolved.drop(columns=["cnt"]), config), evolved["cnt"].to_numpy(dtype=float)

    pre = build_preprocessor(config)
    model = GradientBoostingRegressor(**config["model"]).fit(pre.fit_transform(X_train), y_train)
    m_orig = regression_metrics(y_test, model.predict(pre.transform(X_test)))
    m_evo = regression_metrics(y_evo, model.predict(pre.transform(X_evo)))

    MODEL_V2.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_V2)
    joblib.dump(pre, PREP_V2)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="Lab12_Retraining_Model_V2") as run:
        mlflow.log_params({**config["model"], "model_family": "GradientBoostingRegressor",
                           "training_dataset": str(UPDATED), "training_rows": len(X_train)})
        mlflow.log_metrics({**m_orig, **{f"evolved_{k}": v for k, v in m_evo.items()}})
        mlflow.log_artifact(str(PREP_V2), artifact_path="preprocessing_pipeline_v2")
        run_id = run.info.run_id

    print("=" * 70 + "\nMODEL RETRAINING (V2)\n" + "=" * 70)
    print(f"Training rows: {len(X_train)}")
    print(f"V2 on ORIGINAL hold-out : RMSE={m_orig['rmse']:.2f} R2={m_orig['r2_score']:.4f}")
    print(f"V2 on EVOLVED data      : RMSE={m_evo['rmse']:.2f} R2={m_evo['r2_score']:.4f}")
    write_json(REPORT, {"mlflow_run_id": run_id, "training_dataset": str(UPDATED), "training_rows": len(X_train),
                        "metrics": m_orig, "evolved_metrics": m_evo})
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
