"""Stage 6 - Evaluate the freshly trained model on the fixed hold-out set."""
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import load_config, regression_metrics, write_json  # noqa: E402


def run_evaluation():
    print("[INFO] Starting ML Pipeline Model Evaluation...")
    config = load_config()

    model_path = config["paths"]["model"]
    processed_dir = config["data"]["processed_dir"]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        return False

    try:
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run 03_preprocessing.py first.")
        return False

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    residuals = y_test - y_pred
    abs_errors = np.abs(residuals)

    print("\n--- ML Pipeline: Model Evaluation ---")
    print(f"RMSE : {metrics['rmse']:.2f}")
    print(f"MAE  : {metrics['mae']:.2f}")
    print(f"MSE  : {metrics['mse']:.2f}")
    print(f"R2   : {metrics['r2_score']:.4f}")
    print("-------------------------------------\n")

    os.makedirs("artifacts", exist_ok=True)
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="Perfect prediction")
    ax.set_xlabel("Actual daily rentals")
    ax.set_ylabel("Predicted daily rentals")
    ax.set_title("Actual vs Predicted (hold-out)")
    ax.legend()
    fig.savefig("artifacts/actual_vs_predicted.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=25)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Days")
    ax.set_title("Residual distribution (hold-out)")
    fig.savefig("artifacts/residuals.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    report = {
        "stage": "ML Pipeline - Model Evaluation",
        "evaluated_on": "fixed chronological hold-out split",
        "test_rows": int(len(y_test)),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
        "error_summary": {
            "mean_residual": round(float(residuals.mean()), 4),
            "median_absolute_error": round(float(np.median(abs_errors)), 4),
            "p90_absolute_error": round(float(np.quantile(abs_errors, 0.9)), 4),
            "max_absolute_error": round(float(abs_errors.max()), 4),
            "negative_predictions": int((y_pred < 0).sum()),
        },
    }
    write_json(config["paths"]["evaluation_report"], report)
    print(f"[SUCCESS] Evaluation report saved to {config['paths']['evaluation_report']}")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_evaluation() else 1)
